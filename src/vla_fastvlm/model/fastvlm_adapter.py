# src/vla_fastvlm/model/fastvlm_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
)

# torchvision はフォールバックで使用（PIL/NumPy混在にも強い）
try:
    import torchvision.transforms.functional as TF
    _HAS_TV = True
except Exception:
    _HAS_TV = False


Tensor = torch.Tensor
ImageLike = Union[Tensor, "numpy.ndarray", "PIL.Image.Image"]  # type: ignore


@dataclass
class FastVLMBackboneConfig:
    model_id: str = "apple/FastVLM-base"
    freeze_backbone: bool = True
    # "last_token" | "mean_pool"
    image_feature_pool: str = "last_token"
    # 最終的にビジョンタワーへ渡す正方形サイズ（model/processorから推定できなければこの値）
    fallback_image_size: int = 1024 #384
    # 強制サイズ（指定時は推定を上書き）
    # force_image_size: Optional[int] = None
    force_image_size: Optional[int] = 1024
    # 入力が uint8 のとき [0,1]→ImageNet 正規化まで行うか
    normalize_imagenet: bool = False
    # モデルへ渡す画像キーの優先順
    image_key_order: Tuple[str, ...] = ("images", "pixel_values", "pixel_values_vit")


class FastVLMBackbone(nn.Module):
    """
    LLaVA/FastVLM系の VLM を特徴抽出として使うバックボーン。
    ここで必ず (B,3,S,S) に揃えてから基盤モデルに渡すことで、
    途中の avg_pool2d(kernel_size=16) 等で 0x0 にならないよう保証する。
    """

    def __init__(self, config: FastVLMBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or FastVLMBackboneConfig()

        # ---- モデル本体
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            dtype=torch.float32,  # MPS安定性のため既定float32
        )
        if hasattr(self.model, "config"):
            self.model.config.output_hidden_states = True

        # hidden 次元の推定
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hs_list = getattr(self.model.config, "hidden_sizes", None)
            if isinstance(hs_list, (list, tuple)) and len(hs_list) > 0:
                hidden_size = int(hs_list[-1])
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config.")
        self.output_dim = int(hidden_size)

        # ---- Processor/Tokenizer 準備
        self.processor = None
        self.tokenizer = None
        self.image_processor = None

        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True)
        except Exception:
            self.processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        except Exception as e:
            if self.processor is not None and hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
            else:
                raise e

        if self.processor is not None and hasattr(self.processor, "image_processor"):
            self.image_processor = self.processor.image_processor
        else:
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.config.model_id, trust_remote_code=True
                )
            except Exception:
                self.image_processor = None  # 後で torchvision フォールバック

        # ---- 期待画像サイズの決定
        self.expected_size = self._resolve_expected_image_size()

        # 可能なら processor に明示セット（実装により無視される場合あり）
        ip = getattr(self, "image_processor", None)
        try:
            if ip is not None and hasattr(ip, "size"):
                desired = {"height": self.expected_size, "width": self.expected_size}
                try:
                    cur = ip.size
                except Exception:
                    cur = None
                if cur != desired:
                    ip.size = desired
        except Exception:
            pass

        # ---- 凍結
        if self.config.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        if not _HAS_TV and self.image_processor is None:
            raise RuntimeError(
                "No image processor found and torchvision is unavailable. "
                "Install torchvision or ensure the model repo provides an image processor."
            )

        print(f"[FastVLMBackbone] expected (S,S) = ({self.expected_size},{self.expected_size})")

    # -------------------- ヘルパ --------------------

    def _resolve_expected_image_size(self) -> int:
        if self.config.force_image_size is not None:
            return int(self.config.force_image_size)

        # 1) model.config.vision_config.image_size
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            vcfg = getattr(cfg, "vision_config", None)
            if vcfg is not None:
                img_size = getattr(vcfg, "image_size", None)
                if isinstance(img_size, (int, float)):
                    return int(img_size)
                if isinstance(img_size, (tuple, list)) and len(img_size) > 0:
                    return int(img_size[0])

        # 2) processor.image_processor.size
        ip = getattr(self, "image_processor", None)
        if ip is not None and hasattr(ip, "size"):
            size = getattr(ip, "size")
            if isinstance(size, dict):
                h = size.get("height") or size.get("shortest_edge") or size.get("max_height")
                if isinstance(h, (int, float)):
                    return int(h)
            if isinstance(size, (int, float)):
                return int(size)

        # 3) fallback
        return int(self.config.fallback_image_size)

    @staticmethod
    def _pool_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor], mode: str) -> torch.Tensor:
        """
        hidden: (B, T, H)
        attention_mask: (B, T) or None
        return: (B, H)
        """
        if mode == "mean_pool":
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.float().unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            return summed / denom

        # last_token
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1)
            idx = (lengths - 1).clamp_min(0)
            b, _, h = hidden.size()
            gather_idx = idx.view(b, 1, 1).expand(b, 1, h)
            return hidden.gather(dim=1, index=gather_idx).squeeze(1)
        return hidden[:, -1, :]

    def _prep_text(self, tasks: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer(
            tasks,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tok.items()}

    # -------- 画像前処理：必ず (B,3,S,S) を作る --------

    def _as_bchw(self, images: Union[List[ImageLike], ImageLike, Tensor]) -> Tensor:
        """
        いろいろな形式（PIL / NumPy / Tensor(BCHW/BHWC/CHW/HWC) / list）を BCHW float32 に正規化。
        値域はそのまま（後で正規化フラグに応じて処理）。
        """
        try:
            import numpy as np
            from PIL import Image as PILImage  # type: ignore
        except Exception:
            np = None
            PILImage = None  # type: ignore

        def _one_to_chw(x) -> Tensor:
            if isinstance(x, torch.Tensor):
                t = x
                if t.ndim == 3:
                    # CHW or HWC
                    if t.shape[0] in (1, 3):
                        return t.to(dtype=torch.float32)
                    else:
                        return t.permute(2, 0, 1).to(dtype=torch.float32)
                elif t.ndim == 2:
                    return t.unsqueeze(0).to(dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
            if np is not None and isinstance(x, np.ndarray):
                arr = x
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    return torch.from_numpy(arr).to(dtype=torch.float32)
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
                raise ValueError(f"Unsupported numpy array shape: {arr.shape}")
            if PILImage is not None and isinstance(x, PILImage):
                arr = torch.from_numpy(np.array(x))  # type: ignore
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    return arr.permute(2, 0, 1).to(dtype=torch.float32)
                raise ValueError(f"Unsupported PIL image shape: {arr.shape if hasattr(arr,'shape') else type(x)}")
            raise TypeError(f"Unsupported image type: {type(x)}")

        if isinstance(images, torch.Tensor) and images.ndim == 4:
            x = images
            # BHWC → BCHW
            if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                x = x.permute(0, 3, 1, 2)
            return x.to(dtype=torch.float32)

        if isinstance(images, torch.Tensor):  # CHW/HWC
            return _one_to_chw(images).unsqueeze(0)

        if isinstance(images, (list, tuple)):
            batch: List[Tensor] = []
            for img in images:
                chw = _one_to_chw(img)
                batch.append(chw)
            return torch.stack(batch, dim=0)

        # PIL/np 1枚
        chw = _one_to_chw(images)
        return chw.unsqueeze(0)

    def _resize_square(self, x_bchw: Tensor, S: int) -> Tensor:
        # 3ch 化
        if x_bchw.shape[1] == 1:
            x_bchw = x_bchw.repeat(1, 3, 1, 1)
        elif x_bchw.shape[1] > 3:
            x_bchw = x_bchw[:, :3]

        # 正方形リサイズ（歪みを許容：多くの FastViT/HD 系は square 前提）
        if x_bchw.shape[-2:] != (S, S):
            x_bchw = F.interpolate(x_bchw, size=(S, S), mode="bilinear", align_corners=False)
        return x_bchw

    def _maybe_normalize_imagenet(self, x_bchw: Tensor) -> Tensor:
        if not self.config.normalize_imagenet:
            return x_bchw
        if not _HAS_TV:
            # torchvision が無い環境は簡易実装
            mean = torch.tensor([0.485, 0.456, 0.406], device=x_bchw.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x_bchw.device).view(1, 3, 1, 1)
            return (x_bchw - mean) / std
        # 入力が 0..255 の可能性もあるので float32 で 0..1 に
        if x_bchw.max() > 1.5:
            x_bchw = x_bchw / 255.0
        # TF.normalize と等価
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return TF.normalize(x_bchw, mean=mean, std=std)

    def _prepare_images_tensor(self, images: Union[List[ImageLike], ImageLike, Tensor], device: torch.device) -> Tensor:
        """
        → (B,3,S,S) float32 tensor（正方形・高解像度を保証）
        """
        S = int(self.expected_size)
        x = self._as_bchw(images)             # (B,C,H,W)
        x = x.to(device="cpu")                 # CPUでリサイズ・正規化が安定
        x = self._resize_square(x, S)          # (B,3,S,S)
        x = self._maybe_normalize_imagenet(x)  # 正規化（必要な場合）
        return x.to(device)

    def _pack_image_inputs(self, x_bchw: Tensor, prefer_keys: Tuple[str, ...]) -> List[Dict[str, Tensor]]:
        """
        モデル実装の違いに合わせて複数のキー候補を用意
        """
        variants: List[Dict[str, Tensor]] = []
        for k in prefer_keys:
            variants.append({k: x_bchw})
        return variants

    # -------------------- Forward --------------------

    @torch.no_grad()
    def forward(
        self,
        images: Union[List[ImageLike], ImageLike, Tensor],
        tasks: List[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:  # (B, H=self.output_dim)
        if device is None:
            # モデルのデバイスに追随
            device = next(self.model.parameters()).device

        # 画像を (B,3,S,S) に固定
        img_bchw = self._prepare_images_tensor(images, device)

        # テキスト（タスク指示）
        text_inputs = self._prep_text(tasks, device)

        # 入力候補（images/pixel_values/pixel_values_vit）
        image_variants = self._pack_image_inputs(img_bchw, self.config.image_key_order)

        common: Dict[str, Any] = dict(
            output_hidden_states=True,
            return_dict=True,
        )

        last_out = None
        last_err: Optional[Exception] = None
        tried: List[str] = []

        for img_dict in image_variants:
            inputs = {**text_inputs, **img_dict}
            try:
                out = self.model(**inputs, **common)
                last_out = out
                last_err = None
                break
            except TypeError as e:
                last_err = e
                tried.append(list(img_dict.keys())[0])
                continue

        if last_out is None:
            raise TypeError(
                "VLM forward failed for all image-key variants. "
                f"Tried keys: {tried}. Last error: {repr(last_err)}"
            )

        outputs = last_out

        # (B,T,H) を取り出し
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_seq = outputs.last_hidden_state
        elif getattr(outputs, "hidden_states", None) is not None:
            hidden_seq = outputs.hidden_states[-1]
        else:
            raise RuntimeError("Backbone did not return hidden states. Ensure output_hidden_states=True.")

        attention_mask = text_inputs.get("attention_mask", None)
        pooled = self._pool_hidden(hidden_seq, attention_mask=attention_mask, mode=self.config.image_feature_pool)
        return pooled

    # 互換: 古い呼び出し `self.backbone(images, tasks, device=...)` に対応
    def backbone(self, images, tasks, device: Optional[torch.device] = None, **kwargs):
        return self.forward(images, tasks, device=device)
