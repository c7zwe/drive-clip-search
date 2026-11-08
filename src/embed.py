"""下载图片，用 CLIP 提 512 维向量，写入 LanceDB。"""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 必须在 import 前设置
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import lancedb
import numpy as np
import open_clip
import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
DB_DIR = ROOT / "data" / "lancedb"

N_IMG = 120
BATCH = 16


def download_images():
    # 已存在的文件跳过，重复运行不会重复下载
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for page in (1, 2):
        resp = requests.get("https://picsum.photos/v2/list",
                            params={"page": page, "limit": 100}, timeout=30)
        items.extend(resp.json())
    items = items[:N_IMG]

    for k, it in enumerate(items, 1):
        out = IMG_DIR / f"{it['id']}.jpg"
        if not out.exists():
            out.write_bytes(requests.get(
                f"https://picsum.photos/id/{it['id']}/600/400.jpg", timeout=30
            ).content)
        if k % 20 == 0:
            print(f"  下载 {k}/{N_IMG}")
    return items


def main():
    items = download_images()

    # 第三个返回值才是推理用的 preprocess，前一个是训练用的（带随机裁剪）
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()

    vectors = []
    t0 = time.perf_counter()
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        imgs = torch.stack([
            preprocess(Image.open(IMG_DIR / f"{it['id']}.jpg").convert("RGB"))
            for it in chunk
        ])
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # 单位化后点积即余弦
        vectors.append(feats.numpy())

    vecs = np.vstack(vectors).astype(np.float32)
    cost = time.perf_counter() - t0
    print(f"向量 {vecs.shape}  {cost:.2f}s  {cost / len(items) * 1000:.0f} ms/张")

    rows = [
        {
            "img_id": str(it["id"]),
            "author": it["author"],
            "path": str(IMG_DIR / f"{it['id']}.jpg"),
            "vector": vecs[i].tolist(),
        }
        for i, it in enumerate(items)
    ]
    DB_DIR.mkdir(parents=True, exist_ok=True)
    table = lancedb.connect(str(DB_DIR)).create_table("images", data=rows, mode="overwrite")
    print(f"images 表写入 {table.count_rows()} 行，{len(rows[0]['vector'])} 维")


if __name__ == "__main__":
    main()
