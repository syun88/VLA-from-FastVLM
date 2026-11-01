from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .configuration_fastvla import FastVLAConfig


@dataclass
class FastVLMFeatures:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    visual_tokens: np.ndarray | None
    global_hidden: np.ndarray | None
    image_token_position: int
    preprocessing_seconds: float
    vision_seconds: float
    language_seconds: float


class FastVLMBackbone:
    """MLX FastVLM feature extractor with an explicitly frozen base model."""

    def __init__(self, config: FastVLAConfig, *, load_model: bool = True) -> None:
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
        self._prewarm()

    def _prewarm(self) -> None:
        if self.model is None:
            return
        ids = mx.array(np.asarray([[1, 2]], dtype=np.int32))
        embeddings = self.model.language_model.model.embed_tokens(ids)
        mx.eval(embeddings)

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        return int(self.model.config.text_config.hidden_size)

    def prompt(self, instruction: str) -> str:
        if instruction in self._prompt_cache:
            return self._prompt_cache[instruction]
        from mlx_vlm.prompt_utils import apply_chat_template

        if self.model is None or self.processor is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        rendered = apply_chat_template(
            self.processor,
            self.model.config,
            instruction,
            num_images=1,
            add_generation_prompt=True,
        )
        if self.config.max_text_tokens and len(rendered) > self.config.max_text_tokens * 6:
            raise ValueError("Instruction prompt is unexpectedly long")
        self._prompt_cache[instruction] = rendered
        return rendered

    def prepare(self, image: Image.Image | np.ndarray, instruction: str) -> dict[str, mx.array]:
        if self.model is None or self.processor is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        from mlx_vlm.utils import prepare_inputs

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        if self.config.image_longest_edge:
            image = self._resize_longest_edge(image, self.config.image_longest_edge)
        rendered = self.prompt(instruction)
        inputs = prepare_inputs(
            self.processor,
            images=[image],
            prompts=[rendered],
            image_token_index=self.model.config.image_token_index,
            add_special_tokens=False,
            padding=False,
        )
        if inputs.get("pixel_values") is None:
            raise ValueError("FastVLM preprocessing did not produce pixel_values")
        return {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
        }

    @staticmethod
    def _resize_longest_edge(image: Image.Image, longest_edge: int) -> Image.Image:
        width, height = image.size
        scale = longest_edge / max(width, height)
        if scale == 1.0:
            return image
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(target, Image.Resampling.BICUBIC)

    def encode(self, image: Image.Image | np.ndarray, instruction: str) -> FastVLMFeatures:
        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        preprocessing_start = time.perf_counter()
        inputs = self.prepare(image, instruction)
        preprocessing_seconds = time.perf_counter() - preprocessing_start

        vision_start = time.perf_counter()
        visual_tokens, raw_visual_tokens = self._encode_visual_tokens(inputs["pixel_values"])
        mx.eval(visual_tokens, raw_visual_tokens)
        vision_seconds = time.perf_counter() - vision_start

        language_start = time.perf_counter()
        sequence_embeddings = self.merge_visual_and_text_embeddings(
            inputs["input_ids"], visual_tokens
        )
        hidden = self.model.language_model.model(
            inputs["input_ids"], input_embeddings=sequence_embeddings
        )
        global_hidden = hidden[:, -1, :]
        mx.eval(global_hidden)
        language_seconds = time.perf_counter() - language_start

        input_ids = np.asarray(inputs["input_ids"], dtype=np.int32)
        image_positions = np.flatnonzero(input_ids[0] == self.model.config.image_token_index)
        if image_positions.size != 1:
            raise ValueError(f"Expected one image token, found {image_positions.size}")
        return FastVLMFeatures(
            input_ids=input_ids,
            attention_mask=np.asarray(inputs["attention_mask"], dtype=np.int8),
            visual_tokens=np.asarray(
                visual_tokens[0].astype(mx.float32), dtype=np.float32
            ).astype(np.float16),
            global_hidden=np.asarray(
                global_hidden[0].astype(mx.float32), dtype=np.float32
            ),
            image_token_position=int(image_positions[0]),
            preprocessing_seconds=preprocessing_seconds,
            vision_seconds=vision_seconds,
            language_seconds=language_seconds,
        )

    def _encode_visual_tokens(self, pixel_values: mx.array) -> tuple[mx.array, mx.array]:
        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        model_dtype = self.model.language_model.model.embed_tokens.weight.dtype
        pixel_values = pixel_values.astype(self.model.vision_tower.vision_model.patch_embed.blocks[0].reparam_conv.weight.dtype)
        _, raw_features, _ = self.model.vision_tower(
            pixel_values.transpose(0, 2, 3, 1)
        )
        batch, height, width, channels = raw_features.shape
        flat = raw_features.reshape(batch, height * width, channels)
        projected = self.model.mm_projector(flat).astype(model_dtype)
        return projected, raw_features

    def merge_visual_and_text_embeddings(
        self,
        input_ids: mx.array,
        visual_tokens: mx.array,
        *,
        image_token_position: int | None = None,
    ) -> mx.array:
        """Replace the image placeholder with cached projected FastViTHD tokens.

        This is used by action-head and LoRA training after validating that FastViTHD
        and the multimodal projector are frozen. It avoids recomputing the image tower
        for every action-head update while retaining an end-to-end FastVLM text path.
        """

        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        if input_ids.shape[0] != 1:
            raise ValueError(
                "FastVLM MLX training currently requires batch size 1 because Qwen2 "
                "does not expose a padding mask in this pinned MLX-VLM revision"
            )
        if image_token_position is None:
            ids = np.asarray(input_ids, dtype=np.int32)
            positions = np.flatnonzero(ids[0] == self.model.config.image_token_index)
            if positions.size != 1:
                raise ValueError(f"Expected one image token, found {positions.size}")
            image_token_position = int(positions[0])
        text_embeddings = self.model.language_model.model.embed_tokens(input_ids)
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens[None, :, :]
        return mx.concatenate(
            [
                text_embeddings[:, :image_token_position, :],
                visual_tokens.astype(text_embeddings.dtype),
                text_embeddings[:, image_token_position + 1 :, :],
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
        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        embeddings = self.merge_visual_and_text_embeddings(
            input_ids,
            visual_tokens,
            image_token_position=image_token_position,
        )
        return self.model.language_model.model(input_ids, input_embeddings=embeddings)

    def to_lora(self) -> None:
        if self.model is None:
            raise RuntimeError("FastVLM backbone is not loaded")
        from mlx_vlm.trainer.lora import LoRaLayer
        from mlx_vlm.trainer.utils import (
            find_all_linear_names,
            set_module_by_name,
        )

        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        available = set(find_all_linear_names(self.model))
        unknown = set(targets).difference(available)
        if unknown:
            raise ValueError(f"LoRA target modules not found: {sorted(unknown)}")
        self.model.freeze()
        # CallableModuleList in FastVLM is not an nn.Module container, so freeze its
        # linear members explicitly when present.
        projector = getattr(self.model, "mm_projector", None)
        if isinstance(projector, (list, tuple)):
            for module in projector:
                if isinstance(module, nn.Module):
                    module.freeze()
        layers = list(self.model.language_model.named_modules())
        for name, module in layers:
            if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
                replacement = LoRaLayer(
                    module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                )
                set_module_by_name(self.model.language_model, name, replacement)
        self.model.config.lora = {
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "dropout": self.config.lora_dropout,
        }
        self.model.train()

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def diagnostics(self) -> dict[str, Any]:
        if self.model is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_type": str(self.model.config.model_type),
            "hidden_size": self.hidden_size,
            "image_token_index": int(self.model.config.image_token_index),
            "path": self.config.base_model_path or self.config.base_model,
            "revision": self.config.base_revision,
        }
