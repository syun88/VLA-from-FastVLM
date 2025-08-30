#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: FastVLM の言語側潜在ベクトルを抽出し、Action Expert (MLP) に渡す最小例。
 - original.py は触らず、潜在抽出とMLPの配線だけ確認
 - まだ学習はしない（ランダム重みで forward のみ）
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Action Expert (MLP)
# -----------------------------
class ActionExpertMLP(nn.Module):
    """latent(d_model) -> continuous action(action_dim)"""
    def __init__(self, d_model: int, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, d_model]
        return self.net(z)


# -----------------------------
# FastVLM Loader
# -----------------------------
def load_fastvlm(mid: str, revision: str):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # 推論テストのみ: CUDAはfp16, それ以外はfp32
    dtype = torch.float16 if device == "cuda" else torch.float32  # MPSはfp32推奨

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        revision=revision,
    )
    return tok, model


# -----------------------------
# Inputs Builder (<image> を所定位置に差し込む)
# -----------------------------
def build_inputs(tok, model, pil_img: Image.Image, prompt: str):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": f"<image>\n{prompt}"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    pre, post = rendered.split("<image>", 1)

    pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids

    # 画像トークンID（将来互換）
    img_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    IMAGE_TOKEN_INDEX = -200 if img_token_id is None else int(img_token_id)
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)

    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # 画像前処理（モデル付属のprocessor）
    px = model.get_vision_tower().image_processor(
        images=pil_img, return_tensors="pt"
    )["pixel_values"].to(model.device, dtype=model.dtype)

    return input_ids, attention_mask, px


# -----------------------------
# Latent Encoder (hidden states を取得)
# -----------------------------
def encode_latent(model, input_ids, attention_mask, pixel_values) -> torch.Tensor:
    with torch.no_grad():  # ★ inference_mode は使わない
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    last_h = out.hidden_states[-1]   # [B, T, d_model]
    latent = last_h[:, -1, :]        # 末尾トークンを代表ベクトルに採用: [B, d_model]
    return latent


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="入力画像パス")
    ap.add_argument("--prompt", type=str, default="Describe this image.")
    ap.add_argument("--mid", type=str, default="apple/FastVLM-0.5B")
    ap.add_argument("--revision", type=str, default="main")  # 必要に応じて固定コミットへ
    ap.add_argument("--action-dim", type=int, default=8, help="出力アクション次元")
    ap.add_argument("--save-latent", type=str, default="", help="latent を .pt 保存するパス(任意)")
    args = ap.parse_args()

    # 1) モデル読み込み（original.py は別で保持）
    tok, model = load_fastvlm(args.mid, args.revision)

    # 2) 入力構築
    img_path = Path(args.image)
    assert img_path.exists(), f"Image not found: {img_path}"
    pil_img = Image.open(img_path).convert("RGB")
    input_ids, attn, px = build_inputs(tok, model, pil_img, args.prompt)

    # 3) 潜在ベクトル抽出
    latent = encode_latent(model, input_ids, attn, px)  # [1, d_model]
    # 念のため通常テンソル化（学習時の拡張にも安全）
    latent = latent.clone().detach()
    d_model = latent.shape[-1]
    print(f"[info] latent shape: {tuple(latent.shape)} (d_model={d_model})")

    # 4) Action Expert を通す（まだ学習なし）
    expert = ActionExpertMLP(d_model=d_model, action_dim=args.action_dim).to(model.device)
    expert.eval()
    with torch.no_grad():
        action = expert(latent)  # [1, action_dim]
    print(f"[info] action shape: {tuple(action.shape)}")
    print(f"[debug] action (unnormalized): {action.detach().cpu().numpy()}")

    # 5) 任意: 潜在を保存（今後の学習前処理に活用）
    if args.save_latent:
        out_path = Path(args.save_latent)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latent": latent.cpu(), "image": str(img_path), "prompt": args.prompt}, out_path)
        print(f"[info] latent saved to: {out_path}")

    print("[done] step2 OK.")


if __name__ == "__main__":
    main()
