"""
bench_io：CSV vs Parquet 的 6 组对比实验

跑完你会看到：
  1. Parquet 数行数不用搬数据（只读文件尾部的目录页）
  2. 只读需要的列，不解析全部列
  3. 分区能跳过无关文件，但切太碎会因为「开门费」变慢（小文件问题）
"""

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
    con.execute(sql).fetchall()          # 预热：第一次要读文件元数据
    best = float("inf")
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = con.execute(sql).fetchall()
        best = min(best, (time.perf_counter() - t0) * 1000)
    print(f"{label:<12}{best:>9.0f} ms    {out[:2]}")
    return best


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)


section("实验1  全表 count(*)  —— Parquet 只读文件尾部的元数据")
bench("CSV", f"select count(*) from read_csv_auto('{CSV}')")
bench("Parquet", f"select count(*) from read_parquet('{PQ}')")

section("实验2  只读 1 列求均值 —— CSV 要解析全部 13 列")
bench("CSV", f"select avg(speed_kmh) from read_csv_auto('{CSV}')")
bench("Parquet", f"select avg(speed_kmh) from read_parquet('{PQ}')")

section("实验3  过滤 city='上海' and weather='rainy'")
bench("CSV", f"select count(*) from read_csv_auto('{CSV}') where city='上海' and weather='rainy'")
bench("Parquet", f"select count(*) from read_parquet('{PQ}') where city='上海' and weather='rainy'")

section("实验4  按城市分组求平均车速")
bench("CSV", f"select city, round(avg(speed_kmh),2) from read_csv_auto('{CSV}') group by city order by 2 desc")
bench("Parquet", f"select city, round(avg(speed_kmh),2) from read_parquet('{PQ}') group by city order by 2 desc")

section("实验5  只查「2026-01-15 + 上海」—— 分区表直接跳过无关文件")
bench("Parquet单文件", f"select count(*) from read_parquet('{PQ}') "
                    f"where clip_date = DATE '2026-01-15' and city='上海'")
bench("Parquet分区", f"select count(*) from read_parquet('{PQ_PART}') "
                   f"where clip_date = DATE '2026-01-15' and city='上海'")

section("实验6  落盘体积")
csv_mb = os.path.getsize(DATA / "index.csv") / 1024**2
pq_mb = os.path.getsize(DATA / "index.parquet") / 1024**2
files = list((DATA / "index_partitioned").rglob("*.parquet"))
part_mb = sum(f.stat().st_size for f in files) / 1024**2
print(f"CSV            {csv_mb:>10.1f} MB")
print(f"Parquet 单文件  {pq_mb:>10.1f} MB   压缩到 CSV 的 {pq_mb / csv_mb:.0%}")
print(f"Parquet 分区表  {part_mb:>10.1f} MB   {len(files)} 个文件")
