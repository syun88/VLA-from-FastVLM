#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済みExpertで推論（1枚画像→アクション）
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_expert import ActionExpertMLP, build_inputs, encode_latent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--ckpt", required=True)  # runs/exp1/best.pt
    ap.add_argument("--mid", default="apple/FastVLM-0.5B")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    # load VLM (frozen)
    tok = AutoTokenizer.from_pretrained(args.mid, trust_remote_code=True, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.mid, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True, revision=args.revision,
    )
    for p in model.parameters(): p.requires_grad_(False); model.eval()

    # load expert
    ckpt = torch.load(args.ckpt, map_location=model.device)
    expert = ActionExpertMLP(d_model=ckpt["d_model"], action_dim=ckpt["action_dim"]).to(model.device)
    expert.load_state_dict(ckpt["expert"]); expert.eval()

    # build input
    img = Image.open(args.image).convert("RGB")
    ids, attn, px = build_inputs(tok, model, img, args.prompt)

    # latent -> action
    with torch.no_grad():
        z = encode_latent(model, ids, attn, px)         # [1, d_model]
        act = expert(z)                                  # [1, A]
    print("action:", act.detach().cpu().numpy())

if __name__ == "__main__":
    main()
