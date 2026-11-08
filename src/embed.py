"""
embed：图片 → CLIP 512 维向量 → LanceDB（离线建索引）

图片来自 picsum.photos（公开、免注册）。换成真实路采数据时，
把这个 download_images() 换掉就行，后面的代码一行不用改。
"""

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像，必须在 import 之前
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
    """抓一批公开照片，已存在的自动跳过（幂等）"""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for page in (1, 2):
        resp = requests.get(
            "https://picsum.photos/v2/list",
            params={"page": page, "limit": 100},
            timeout=30,
        )
        items.extend(resp.json())
    items = items[:N_IMG]

    for k, it in enumerate(items, 1):
        out = IMG_DIR / f"{it['id']}.jpg"
        if not out.exists():
            img = requests.get(
                f"https://picsum.photos/id/{it['id']}/600/400.jpg", timeout=30
            ).content
            out.write_bytes(img)
        if k % 20 == 0:
            print(f"   下载 {k}/{N_IMG}")
    return items


def main():
    print("① 抓图片")
    items = download_images()

    print("\n② 加载 CLIP")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()

    print("\n③ 提向量")
    vectors = []
    t0 = __import__("time").perf_counter()
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        imgs = torch.stack([
            preprocess(Image.open(IMG_DIR / f"{it['id']}.jpg").convert("RGB")) for it in chunk
        ])
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # 归一化：让点积直接等于余弦
        vectors.append(feats.numpy())
        print(f"   {min(start + BATCH, len(items))}/{len(items)}")

    vecs = np.vstack(vectors).astype(np.float32)
    cost = __import__("time").perf_counter() - t0
    print(f"\n向量形状 {vecs.shape}   总耗时 {cost:.2f}s   平均 {cost / len(items) * 1000:.0f} ms/张")

    print("\n④ 写入 LanceDB")
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
    db = lancedb.connect(str(DB_DIR))
    table = db.create_table("images", data=rows, mode="overwrite")
    print(f"表 images 已写入 {table.count_rows()} 行，每行 {len(rows[0]['vector'])} 维")


if __name__ == "__main__":
    main()
