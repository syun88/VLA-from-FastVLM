import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = "apple/FastVLM-0.5B"
REV = "main"  # 必要なら特定コミットに固定
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # MPSはfp32推奨

tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True, revision=REV)
model = AutoModelForCausalLM.from_pretrained(
    MID,
    dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    revision=REV,
)

# 画像トークンID（将来変更に備えて自動取得を試みる）
img_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
IMAGE_TOKEN_INDEX = -200 if img_token_id is None else img_token_id

messages = [{"role": "user", "content": "<image>\nDescribe this image in detail."}]
rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
pre, post = rendered.split("<image>", 1)
pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
attention_mask = torch.ones_like(input_ids, device=model.device)

img = Image.open("cat.jpg").convert("RGB")
px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
px = px.to(model.device, dtype=model.dtype)

with torch.inference_mode():
    out_ids = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=px,
        max_new_tokens=128,
    )

print(tok.decode(out_ids[0], skip_special_tokens=True))
