"""用 Ray 并行处理逐条 clip，对比串行 / 逐条派发 / 分块派发三种方式。"""

import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import ray

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N_CLIP = 10_000
CHUNK = 50


def preprocess(clip_id: str) -> float:
    # 用 crc32 而非内置 hash()，否则各进程算出的种子不一致
    seed = zlib.crc32(clip_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    img = rng.random((224, 224, 3), dtype=np.float32)
    for _ in range(5):
        img = (img * 1.02 + 0.1) % 1.0
    return float(img.mean())


def preprocess_chunk(chunk_ids):
    return [preprocess(x) for x in chunk_ids]


def main():
    ids = pl.read_parquet(DATA / "index.parquet", columns=["clip_id"]) \
        .head(N_CLIP)["clip_id"].to_list()
    print(f"待处理 clip: {len(ids):,}")

    t0 = time.perf_counter()
    serial = []
    for k, x in enumerate(ids):
        serial.append(preprocess(x))
        if (k + 1) % 2000 == 0:
            print(f"  串行 {k + 1:,}/{len(ids):,}  {time.perf_counter() - t0:.1f}s")
    t_serial = time.perf_counter() - t0
    print(f"A 串行             {t_serial:>6.2f}s")

    ray.init(include_dashboard=False, logging_level="ERROR")

    remote_one = ray.remote(preprocess)
    t0 = time.perf_counter()
    res_b = ray.get([remote_one.remote(x) for x in ids])
    t_b = time.perf_counter() - t0
    print(f"B Ray 逐条({len(ids)})   {t_b:>6.2f}s   {t_serial / t_b:.1f}x")

    remote_chunk = ray.remote(preprocess_chunk)
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    t0 = time.perf_counter()
    res_c = ray.get([remote_chunk.remote(c) for c in chunks])
    t_c = time.perf_counter() - t0
    print(f"C Ray 分块({len(chunks)})     {t_c:>6.2f}s   {t_serial / t_c:.1f}x")

    ray.shutdown()

    flat_c = [v for ch in res_c for v in ch]
    print(f"结果一致: {np.allclose(serial, flat_c) and np.allclose(serial, res_b)}")


if __name__ == "__main__":
    main()
