"""CSV / Parquet 的读取性能对比，含分区裁剪和小文件场景。"""

import os
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CSV = str(DATA / "index.csv").replace("\\", "/")
PQ = str(DATA / "index.parquet").replace("\\", "/")
PQ_PART = str(DATA / "index_partitioned").replace("\\", "/") + "/**/*.parquet"

con = duckdb.connect()


def bench(label, sql, runs=3):
    con.execute(sql).fetchall()  # 预热，避免把元数据读取算进首轮
    best = float("inf")
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = con.execute(sql).fetchall()
        best = min(best, (time.perf_counter() - t0) * 1000)
    print(f"{label:<14}{best:>9.0f} ms    {out[:2]}")
    return best


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)


section("count(*)")
bench("CSV", f"select count(*) from read_csv_auto('{CSV}')")
bench("Parquet", f"select count(*) from read_parquet('{PQ}')")

section("avg(speed_kmh)  —— 只读单列")
bench("CSV", f"select avg(speed_kmh) from read_csv_auto('{CSV}')")
bench("Parquet", f"select avg(speed_kmh) from read_parquet('{PQ}')")

section("filter city='上海' and weather='rainy'")
bench("CSV", f"select count(*) from read_csv_auto('{CSV}') "
             f"where city='上海' and weather='rainy'")
bench("Parquet", f"select count(*) from read_parquet('{PQ}') "
                 f"where city='上海' and weather='rainy'")

section("group by city")
bench("CSV", f"select city, round(avg(speed_kmh),2) from read_csv_auto('{CSV}') "
             f"group by city order by 2 desc")
bench("Parquet", f"select city, round(avg(speed_kmh),2) from read_parquet('{PQ}') "
                 f"group by city order by 2 desc")

# 分区表：命中 1 个文件，但要打开 155 个描述文件
section("分区裁剪  2026-01-15 + 上海")
bench("单文件", f"select count(*) from read_parquet('{PQ}') "
               f"where clip_date = DATE '2026-01-15' and city='上海'")
bench("分区表", f"select count(*) from read_parquet('{PQ_PART}') "
               f"where clip_date = DATE '2026-01-15' and city='上海'")

section("落盘体积")
csv_mb = os.path.getsize(DATA / "index.csv") / 1024**2
pq_mb = os.path.getsize(DATA / "index.parquet") / 1024**2
files = list((DATA / "index_partitioned").rglob("*.parquet"))
part_mb = sum(f.stat().st_size for f in files) / 1024**2
print(f"CSV                {csv_mb:>8.1f} MB")
print(f"Parquet            {pq_mb:>8.1f} MB   ({pq_mb / csv_mb:.0%} of CSV)")
print(f"Parquet 分区表      {part_mb:>8.1f} MB   ({len(files)} 个文件)")
