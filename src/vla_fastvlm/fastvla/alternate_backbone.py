from __future__ import annotations

import gc
import time
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from .backbone import FastVLMFeatures
from .configuration_fastvla import FastVLAConfig


class SmolVLMBackbone:
    """Frozen MLX SmolVLM2 comparison backbone with the FastVLA feature contract."""

    def __init__(self, config: FastVLAConfig, *, load_model: bool = True) -> None:
        if config.backbone_family != "smolvlm":
            raise ValueError("SmolVLMBackbone requires backbone_family='smolvlm'")
        self.config = config
        self.model = None
        self.processor = None
        self._prompt_cache: dict[str, str] = {}
        if load_model:
            self.load()

    def load(self) -> None:
        from mlx_vlm import load

        model_path = self.config.base_model_path or self.config.base_model
        revision = None if self.config.base_model_path else self.config.base_revision
        self.model, self.processor = load(model_path, revision=revision)
        self.model.eval()
        self.model.freeze()

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            raise RuntimeError("SmolVLM backbone is not loaded")
        return int(self.model.config.text_config.hidden_size)

    def prompt(self, instruction: str) -> str:
        if instruction in self._prompt_cache:
            return self._prompt_cache[instruction]
        from mlx_vlm.prompt_utils import apply_chat_template

        rendered = apply_chat_template(
            self.processor,
            self.model.config,
            instruction,
            num_images=1,
            add_generation_prompt=True,
        )
        self._prompt_cache[instruction] = rendered
        return rendered

    def prepare(self, image: Image.Image | np.ndarray, instruction: str) -> dict[str, mx.array]:
        from mlx_vlm.utils import prepare_inputs

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        inputs = prepare_inputs(
            self.processor,
            images=[image],
            prompts=[self.prompt(instruction)],
            image_token_index=self.model.config.image_token_index,
            add_special_tokens=False,
            padding=False,
        )
        if inputs.get("pixel_values") is None:
            raise ValueError("SmolVLM preprocessing did not produce pixel_values")
        return {
            key: value
            for key, value in inputs.items()
            if key in {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "pixel_attention_mask",
                "image_sizes",
            }
        }

    def _encode_visual_tokens(self, pixel_values: mx.array) -> mx.array:
        batch, num_images, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch * num_images, channels, height, width)
        patch_mask = mx.ones((flat.shape[0], height, width), dtype=mx.bool_)
        output = self.model.vision_model(
            flat.transpose(0, 2, 3, 1),
            patch_attention_mask=patch_mask,
            output_hidden_states=True,
        )
        pooler_output = output[0] if isinstance(output, tuple) else output.pooler_output
        visual = self.model.connector(pooler_output.astype(flat.dtype))
        return visual.reshape(batch, num_images, visual.shape[1], visual.shape[2])

    @staticmethod
    def _hidden_from_embeddings(model, input_ids: mx.array, embeddings: mx.array) -> mx.array:
        from mlx_vlm.models.base import create_attention_mask

        hidden = embeddings.astype(model.language_model.norm.weight.dtype)
        mask = create_attention_mask(hidden, None)
        for layer in model.language_model.layers:
            hidden = layer(hidden, mask, None)
        return model.language_model.norm(hidden)

    def encode(self, image: Image.Image | np.ndarray, instruction: str) -> FastVLMFeatures:
        preprocessing_start = time.perf_counter()
        inputs = self.prepare(image, instruction)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        vision_start = time.perf_counter()
        visual = self._encode_visual_tokens(inputs["pixel_values"])
        mx.eval(visual)
        vision_seconds = time.perf_counter() - vision_start
        language_start = time.perf_counter()
        embeddings = self.model.get_input_embeddings(
            inputs["input_ids"],
            inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            cached_image_features=visual[:, 0],
        ).inputs_embeds
        hidden = self._hidden_from_embeddings(self.model, inputs["input_ids"], embeddings)
        global_hidden = hidden[:, -1, :]
        mx.eval(global_hidden)
        language_seconds = time.perf_counter() - language_start
        input_ids = np.asarray(inputs["input_ids"], dtype=np.int32)
        image_positions = np.flatnonzero(input_ids[0] == self.model.config.image_token_index)
        if image_positions.size == 0:
            raise ValueError("SmolVLM prompt contains no image tokens")
        visual_np = np.asarray(visual[0, 0].astype(mx.float32), dtype=np.float32)
        return FastVLMFeatures(
            input_ids=input_ids,
            attention_mask=np.asarray(inputs["attention_mask"], dtype=np.int8),
            visual_tokens=visual_np.astype(np.float16),
            global_hidden=np.asarray(
                global_hidden[0].astype(mx.float32), dtype=np.float32
            ),
            image_token_position=int(image_positions[0]),
            preprocessing_seconds=preprocessing_seconds,
            vision_seconds=vision_seconds,
            language_seconds=language_seconds,
        )

    def merge_visual_and_text_embeddings(
        self,
        input_ids: mx.array,
        visual_tokens: mx.array,
        *,
        image_token_position: int,
    ) -> mx.array:
        batch = input_ids.shape[0]
        if batch != 1:
            raise ValueError("SmolVLM comparison path currently uses batch size one")
        text = self.model.language_model.embed_tokens(input_ids)
        positions = np.flatnonzero(
            np.asarray(input_ids[0]) == self.model.config.image_token_index
        )
        count = visual_tokens.shape[-2] if visual_tokens.ndim == 3 else visual_tokens.shape[0]
        if positions.size != count or not np.all(np.diff(positions) == 1):
            raise ValueError("Expected one contiguous image-token block")
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens[None, ...]
        return mx.concatenate(
            [
                text[:, : positions[0], :],
                visual_tokens.astype(text.dtype),
                text[:, positions[-1] + 1 :, :],
            ],
            axis=1,
        )

    def hidden_from_cached_visual(
        self,
        input_ids: mx.array,
        visual_tokens: mx.array,
        *,
        image_token_position: int,
    ) -> mx.array:
        embeddings = self.merge_visual_and_text_embeddings(
            input_ids,
            visual_tokens,
            image_token_position=image_token_position,
        )
        return self._hidden_from_embeddings(self.model, input_ids, embeddings)

    def to_lora(self) -> None:
        raise NotImplementedError("The alternative VLM comparison is frozen-only")

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "model_type": "smolvlm",
            "hidden_size": None if self.model is None else self.hidden_size,
            "path": self.config.base_model_path or self.config.base_model,
            "revision": self.config.base_revision,
        }
