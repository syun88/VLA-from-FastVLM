#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: 行動模倣の最小学習スクリプト
 - FastVLMは凍結して hidden(潜在) だけ抽出
 - Action Expert(MLP) のみ学習（HuberLoss）
 - JSONLデータ: {"image": "path.jpg", "prompt": "text", "action": [..]}
   * action は連続値ベクトル（Δ関節角など）を想定（事前に[-1,1]正規化推奨）
"""

import os, json, math, random, argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Expert MLP（Step2と同じ）
# -----------------------------
class ActionExpertMLP(nn.Module):
    def __init__(self, d_model: int, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -----------------------------
# Dataset（JSONL）
# -----------------------------
class VLAJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        assert len(self.items) > 0, "Empty dataset"

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        o = self.items[idx]
        img = Image.open(o["image"]).convert("RGB")
        prompt = o.get("prompt", "Describe this image.")
        action = torch.tensor(o["action"], dtype=torch.float32)  # [action_dim]
        return img, prompt, action


# -----------------------------
# FastVLM loader & builders
# -----------------------------
def load_fastvlm(mid: str, revision: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=dtype, device_map="auto", trust_remote_code=True, revision=revision,
    )
    # 凍結（勾配不要）
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return tok, model

def build_inputs(tok, model, pil_img: Image.Image, prompt: str):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": f"<image>\n{prompt}"}],
        add_generation_prompt=True, tokenize=False,
    )
    pre, post = rendered.split("<image>", 1)
    pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    IMAGE_TOKEN_INDEX = -200 if img_token_id is None else int(img_token_id)
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)

    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attn = torch.ones_like(input_ids, device=model.device)

    px = model.get_vision_tower().image_processor(
        images=pil_img, return_tensors="pt"
    )["pixel_values"].to(model.device, dtype=model.dtype)
    return input_ids, attn, px

@torch.no_grad()
def encode_latent(model, input_ids, attn, px) -> torch.Tensor:
    out = model(
        input_ids=input_ids, attention_mask=attn, images=px,
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    last_h = out.hidden_states[-1]     # [B, T, d_model]
    latent = last_h[:, -1, :]          # [B, d_model]
    return latent


# -----------------------------
# Train loop
# -----------------------------
def train(args):
    # RNG
    torch.manual_seed(args.seed); random.seed(args.seed)

    # I/O
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Data
    train_ds = VLAJsonlDataset(args.train_jsonl)
    val_ds   = VLAJsonlDataset(args.val_jsonl) if args.val_jsonl else None
    # simple split fallback
    if not val_ds:
        n = len(train_ds); k = int(n*0.1)
        val_indices = set(random.sample(range(n), k))
        tr_items, va_items = [], []
        for i in range(n):
            (va_items if i in val_indices else tr_items).append(train_ds.items[i])
        train_ds.items, val_ds = tr_items, VLAJsonlDataset.__new__(VLAJsonlDataset)
        val_ds.items = va_items
    collate_keep = lambda batch: batch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_keep)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_keep)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    tok, vlm = load_fastvlm(args.mid, args.revision)
    # 1サンプルでd_modelを決定
    img0, pr0, _ = train_ds[0]
    ids0, attn0, px0 = build_inputs(tok, vlm, img0, pr0)
    z0 = encode_latent(vlm, ids0, attn0, px0)
    d_model = z0.shape[-1]
    expert = ActionExpertMLP(d_model=d_model, action_dim=args.action_dim).to(vlm.device)

    # Optim
    opt = torch.optim.AdamW(expert.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()  # Huber

    # Optional: latentキャッシュ
    cache_dir = Path(args.cache_latents) if args.cache_latents else None
    if cache_dir: cache_dir.mkdir(parents=True, exist_ok=True)

    def batch_to_latent(batch):
        imgs, prompts = [b[0] for b in batch], [b[1] for b in batch]
        actions = torch.stack([b[2] for b in batch], dim=0).to(vlm.device)  # [B, action_dim]
        latents = []
        for i, (img, pr) in enumerate(zip(imgs, prompts)):
            if cache_dir:
                key = f"{hash((train_ds.items.index({'image':img.filename,'prompt':pr,'action':None})))}"  # fragile; user can replace with stable key
            # 直接エンコード（まずは素直に）
            ids, attn, px = build_inputs(tok, vlm, img, pr)
            z = encode_latent(vlm, ids, attn, px)  # [1, d_model]
            latents.append(z)
        latents = torch.cat(latents, dim=0)  # [B, d_model]
        return latents, actions

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        expert.train()
        tr_loss = 0.0; n_tr = 0
        for batch in train_loader:
            latents, actions = batch_to_latent(batch)             # [B, d_model], [B, A]
            pred = expert(latents)                                # [B, A]
            loss = loss_fn(pred, actions)
            # ちょい正則化（出力の過大化抑制）
            loss = loss + args.l2_reg * (pred.pow(2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(expert.parameters(), args.clip)
            opt.step()
            tr_loss += loss.item() * actions.size(0); n_tr += actions.size(0)

        sched.step()
        tr_loss /= max(1, n_tr)

        # val
        expert.eval(); va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                latents, actions = batch_to_latent(batch)
                pred = expert(latents)
                loss = loss_fn(pred, actions)
                va_loss += loss.item() * actions.size(0); n_va += actions.size(0)
        va_loss /= max(1, n_va)

        print(f"[E{epoch:03d}] train={tr_loss:.4f}  val={va_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        # save
        ckpt = {
            "epoch": epoch,
            "expert": expert.state_dict(),
            "d_model": d_model,
            "action_dim": args.action_dim,
            "mid": args.mid, "revision": args.revision,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, out_dir / "best.pt")

    print(f"[done] best val loss = {best_val:.4f}  ckpt: {out_dir/'best.pt'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--out-dir", default="runs/exp1")
    ap.add_argument("--mid", default="apple/FastVLM-0.5B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--action-dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--l2-reg", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-latents", default="")  # 省略可
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
