"""
features：用 Ray 把「逐条处理」的活并行化，并对比三种跑法

场景：对每条 clip 做画面预处理，产出一个特征值。
真实的活是解码 jpeg / 读取 mp4 帧，这里用等价的 CPU 计算替代。
"""

import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import ray

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_CLIP = 10_000      # 处理多少条
CHUNK = 50           # 分块时每块多少条


def preprocess(clip_id: str) -> float:
    seed = zlib.crc32(clip_id.encode("utf-8"))   # 用 crc32 而非 hash()，保证各进程算出来一致
    rng = np.random.default_rng(seed)
    img = rng.random((224, 224, 3), dtype=np.float32)
    for _ in range(5):
        img = (img * 1.02 + 0.1) % 1.0
    return float(img.mean())


def preprocess_chunk(chunk_ids):
    return [preprocess(x) for x in chunk_ids]


def main():
    ids = pl.read_parquet(DATA / "index.parquet", columns=["clip_id"]).head(N_CLIP)["clip_id"].to_list()
    print(f"待处理 clip: {len(ids):,} 条\n")

    print("A 串行开始（这一段会慢，看进度）")
    t0 = time.perf_counter()
    serial = []
    for k, x in enumerate(ids):
        serial.append(preprocess(x))
        if (k + 1) % 2000 == 0:
            print(f"   进度 {k + 1:,} / {len(ids):,}   已用 {time.perf_counter() - t0:.1f}s")
    t_serial = time.perf_counter() - t0
    print(f"A 串行，1 个核               {t_serial:>6.2f}s")

    ray.init(include_dashboard=False, logging_level="ERROR")

    remote_one = ray.remote(preprocess)
    t0 = time.perf_counter()
    res_b = ray.get([remote_one.remote(x) for x in ids])
    t_b = time.perf_counter() - t0
    print(f"B Ray 逐条，{len(ids):,} 个任务      {t_b:>6.2f}s   加速 {t_serial / t_b:.1f}x")

    remote_chunk = ray.remote(preprocess_chunk)
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    t0 = time.perf_counter()
    res_c = ray.get([remote_chunk.remote(c) for c in chunks])
    t_c = time.perf_counter() - t0
    print(f"C Ray 分块，{len(chunks):,} 个任务     {t_c:>6.2f}s   加速 {t_serial / t_c:.1f}x")

    ray.shutdown()

    flat_c = [v for ch in res_c for v in ch]
    same = np.allclose(serial, flat_c) and np.allclose(serial, res_b)
    print(f"\n三种跑法结果完全一致: {same}")


if __name__ == "__main__":
    main()
