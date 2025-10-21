# src/vla_fastvlm/model/fastvlm_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
)

# 画像前処理の最低限を自前で用意（画像プロセッサが無い時のフォールバック）
try:
    import torchvision.transforms as T
    _HAS_TV = True
except Exception:
    _HAS_TV = False


@dataclass
class FastVLMBackboneConfig:
    model_id: str = "apple/FastVLM-base"
    freeze_backbone: bool = True
    # 特徴プーリング: "last_token" | "mean_pool"
    image_feature_pool: str = "last_token"
    # 画像プロセッサが無い時のリサイズ解像度（多くのVLMが 336 or 384）
    fallback_image_size: int = 336


class FastVLMBackbone(nn.Module):
    """
    FastVLM / LLaVA 系 VLM を特徴抽出器として使うバックボーン。

    特徴:
    - AutoProcessor が tokenizer-only の場合に備え、tokenizer と image_processor を分離ロード
    - image_processor が見つからない場合は torchvision でリサイズ&標準化（フォールバック）
    - CausalLMOutput / BaseModelOutput の両方に対応（hidden_states or last_hidden_state）
    - モデルが受け付ける画像キー名を複数試行（pixel_values / images など）
    """

    def __init__(self, config: FastVLMBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or FastVLMBackboneConfig()

        # ---- モデル本体
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            dtype=torch.float32,  # macOS/MPS の安定性重視
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

        # ---- 前処理系（できるだけ賢くロード）
        self.processor = None
        self.tokenizer = None
        self.image_processor = None

        # 1) AutoProcessor を試す（成功しても tokenizer-only の可能性がある）
        try:
            proc = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True)
            self.processor = proc
        except Exception:
            self.processor = None

        # 2) tokenizer を確実に用意
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        except Exception as e:
            # AutoProcessor に tokenizer が入っているかも
            if self.processor is not None and hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
            else:
                raise e

        # 3) image_processor をできる限り探す
        #   - AutoProcessor に image_processor が含まれていれば利用
        #   - 無ければ AutoImageProcessor を試す
        if self.processor is not None and hasattr(self.processor, "image_processor"):
            self.image_processor = self.processor.image_processor
        else:
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.config.model_id, trust_remote_code=True
                )
            except Exception:
                self.image_processor = None  # 後で torchvision フォールバック

        # ---- 凍結オプション
        if self.config.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        # ---- torchvision フォールバック（必要時のみ使用）
        if self.image_processor is None and not _HAS_TV:
            # 画像プロセッサも torchvision も無いと画像前処理ができない
            raise RuntimeError(
                "No image processor found and torchvision is unavailable. "
                "Install torchvision or provide a valid image processor in the model repo."
            )

        # torchvision フォールバック用の transform
        if self.image_processor is None and _HAS_TV:
            size = self.config.fallback_image_size
            # 標準的な ImageNet 正規化（多くの VLM で互換）
            self._tv_transform = T.Compose([
                T.Resize((size, size)),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self._tv_transform = None

    # -------------------- ヘルパ --------------------

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
            mask = attention_mask.float().unsqueeze(-1)  # (B,T,1)
            summed = (hidden * mask).sum(dim=1)          # (B,H)
            denom = mask.sum(dim=1).clamp_min(1e-6)      # (B,1)
            return summed / denom

        # last_token
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1)       # (B,)
            idx = (lengths - 1).clamp_min(0)                 # (B,)
            b, _, h = hidden.size()
            gather_idx = idx.view(b, 1, 1).expand(b, 1, h)   # (B,1,H)
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

    def _prep_images(self, images: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        画像をモデルが受け取りそうな dict にして返す。
        可能なキー:
          - "pixel_values"
          - "images"（一部の trust_remote_code 実装）
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images as (B,C,H,W), got {images.shape}")

        if self.image_processor is not None:
            # AutoImageProcessor / processor.image_processor 経由
            # 多くの実装が PIL 変換不要で Tensor を受け取れる
            # （受け取れない場合は .tolist() 的な前処理が必要だが稀）
            out = self.image_processor(
                list(images),  # バッチ Tensor をリストで渡すと安定
                return_tensors="pt",
            )
            return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

        # torchvision フォールバック
        assert self._tv_transform is not None
        with torch.no_grad():
            x = images
            if x.dtype != torch.float32:
                x = x.float()
            # [0,255]→[0,1] の場合は適宜調整（ここではスケール未実施、ConvertImageDtypeが吸収）
            x = torch.stack([self._tv_transform(img.cpu()) for img in x], dim=0)  # (B,3,H,W)
        return {"pixel_values": x.to(device)}

    # -------------------- Forward --------------------

    def forward(
        self,
        images: torch.Tensor,   # (B,C,H,W)
        tasks: List[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:         # (B, H=self.output_dim)
        if device is None:
            device = images.device

        self.model.to(device)

        text_inputs = self._prep_text(tasks, device)
        image_inputs = self._prep_images(images, device)

        # 入力の組み立て
        common: Dict[str, Any] = dict(
            output_hidden_states=True,
            return_dict=True,
        )
        inputs = {**text_inputs, **image_inputs}

        # 受け付ける画像キー名がモデルごとに違う可能性に備えてフォールバック実行
        tried: List[str] = []
        def _try_call(**kw):
            out = self.model(**kw)
            return out

        # 優先順にキーバリアントを試す
        variants: List[Dict[str, Any]] = []

        # そのまま（pixel_values があるならまずは素直に）
        variants.append(inputs)

        # images キーに差し替え（trust_remote_code 系で稀に必要）
        if "pixel_values" in inputs and "images" not in inputs:
            v = dict(inputs)
            v["images"] = v.pop("pixel_values")
            variants.append(v)

        # pixel_values_vit キー（独自実装で見かけることがある）
        if "pixel_values" in inputs and "pixel_values_vit" not in inputs:
            v = dict(inputs)
            v["pixel_values_vit"] = v.pop("pixel_values")
            variants.append(v)

        # 実行
        last_out = None
        err: Optional[Exception] = None
        for cand in variants:
            try:
                out = _try_call(**cand, **common)
                last_out = out
                err = None
                break
            except TypeError as e:
                err = e
                tried.append(", ".join(sorted([k for k in cand.keys() if k not in text_inputs])))
                continue

        if last_out is None:
            raise TypeError(
                "VLM forward failed for all image-key variants. "
                f"Tried image key sets: {tried}. Last error: {repr(err)}"
            )

        outputs = last_out

        # 出力から (B,T,H) を取り出す
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_seq = outputs.last_hidden_state
        elif getattr(outputs, "hidden_states", None) is not None:
            hidden_seq = outputs.hidden_states[-1]
        else:
            raise RuntimeError(
                "Backbone did not return hidden states. Ensure output_hidden_states=True."
            )

        attention_mask = text_inputs.get("attention_mask", None)
        pooled = self._pool_hidden(hidden_seq, attention_mask=attention_mask, mode=self.config.image_feature_pool)
        return pooled
