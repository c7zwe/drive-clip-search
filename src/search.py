"""文字检索：把查询编码成向量，在 LanceDB 中取 top-k，生成 HTML 结果页。

用法：python src/search.py [查询词 ...]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import lancedb
import open_clip
import torch

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "lancedb"
OUT_HTML = ROOT / "result.html"

DEFAULT_QUERIES = ["a dog", "a mountain", "a person", "water", "a city street"]
TOP_K = 8


def to_cosine(distance: float) -> float:
    # LanceDB 返回的 _distance 是 L2 距离的平方；单位向量下 d² = 2 - 2cos
    return 1 - distance / 2


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    table = lancedb.connect(str(DB_DIR)).open_table("images")
    print(f"库里 {table.count_rows()} 条向量")

    sections = []
    for q in queries:
        with torch.no_grad():
            t = model.encode_text(tokenizer([q]))
            t = t / t.norm(dim=-1, keepdim=True)  # 文本同样要单位化
        v = t.numpy()[0].astype("float32").tolist()

        rows = table.search(v).limit(TOP_K).to_list()
        print(f"  {q:<22} top1 cos = {to_cosine(rows[0]['_distance']):.3f}")

        cards = []
        for rank, r in enumerate(rows, 1):
            rel = Path(r["path"]).relative_to(ROOT).as_posix()
            cards.append(
                f'<figure class="card">'
                f'<img src="{rel}" alt="">'
                f'<figcaption><b>#{rank}</b> cos {to_cosine(r["_distance"]):.3f}</figcaption>'
                f'</figure>'
            )
        sections.append(
            f'<section><h2>{q}</h2><div class="grid">{"".join(cards)}</div></section>'
        )

    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>CLIP 检索结果</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
         margin: 0; padding: 32px; background: #fafafa; color: #171717; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 8px; }}
  .sub {{ color: #666; font-size: 14px; margin-bottom: 28px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 32px 0 12px;
        padding-left: 10px; border-left: 3px solid #0FDC78; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
  .card {{ margin: 0; background: #fff; border: 1px solid #e5e5e5;
           border-radius: 10px; overflow: hidden; }}
  .card img {{ width: 100%; height: 140px; object-fit: cover; display: block; }}
  figcaption {{ padding: 8px 10px; font-size: 12px; color: #666; }}
  figcaption b {{ color: #02743B; }}
</style></head><body>
<h1>CLIP 向量检索 · {table.count_rows()} 张图</h1>
<div class="sub">512 维图文向量，按余弦相似度排序</div>
{"".join(sections)}
</body></html>
"""

    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"结果页：{OUT_HTML}")


if __name__ == "__main__":
    main()
