#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済みExpertで推論（1枚画像→アクション）
"""
import argparse
from pathlib import Path
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_expert import ActionExpertMLP, build_inputs, encode_latent


def inverse_scale(x: np.ndarray, scaler_path: str | None = None) -> np.ndarray:
    """[-1,1]などに正規化されて学習した出力を実スケールへ戻す"""
    if not scaler_path:
        return x
    with open(scaler_path, "r") as f:
        sc = json.load(f)  # {"min":[...], "max":[...]} or {"mean":[...], "std":[...]}
    if "min" in sc and "max" in sc:
        mn, mx = np.array(sc["min"]), np.array(sc["max"])
        return (x + 1.0) * 0.5 * (mx - mn) + mn
    if "mean" in sc and "std" in sc:
        mu, st = np.array(sc["mean"]), np.array(sc["std"])
        return x * st + mu
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--ckpt", required=True)  # runs/exp1/best.pt
    ap.add_argument("--mid", default="apple/FastVLM-0.5B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--scaler", default="", help="学習時に使ったスケーラJSONへのパス（任意）")
    ap.add_argument("--clip", type=float, default=1.0, help="出力を[-clip, clip]に制限（Expertがtanhしない場合に有効）")
    args = ap.parse_args()

    # load VLM (frozen)
    tok = AutoTokenizer.from_pretrained(args.mid, trust_remote_code=True, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.mid,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # MPSはfp32
        device_map="auto",
        trust_remote_code=True,
        revision=args.revision,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # load expert
    ckpt = torch.load(args.ckpt, map_location=model.device)
    expert = ActionExpertMLP(d_model=ckpt["d_model"], action_dim=ckpt["action_dim"]).to(model.device)
    expert.load_state_dict(ckpt["expert"])
    expert.eval()

    # build input
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    ids, attn, px = build_inputs(tok, model, img, args.prompt)

    # latent -> action
    with torch.no_grad():
        z = encode_latent(model, ids, attn, px)     # [1, d_model]
        act = expert(z)                             # [1, A]
    act = act.detach().cpu().numpy()[0]            # (A,)

    # クリップ（Expert側でtanhしていない場合の保険）
    if args.clip > 0:
        act = np.clip(act, -args.clip, args.clip)

    # 逆正規化（指定があれば）
    act_real = inverse_scale(act, args.scaler) if args.scaler else act

    print("action (normalized):", act)
    if args.scaler:
        print("action (real):", act_real)


if __name__ == "__main__":
    main()
