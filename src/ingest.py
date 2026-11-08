"""
ingest：生成「智驾 clip 元信息表」，写成 Parquet（分区 + 排序）

真实的数据平台里，一张元信息表要挂住上亿条 clip：
  一行 = 一段连续驾驶视频
  字段 = 城市 / 天气 / 光照 / 时间 / 车速 / 标注框数 / 各传感器文件在对象存储的位置

这里用随机数据模拟同样的 schema，方便在没有真实路采数据时也能复现全部实验。
"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

N = 1_000_000        # clip 条数
DAYS = 30
SEED = 42

CITIES = ["上海", "北京", "深圳", "广州", "杭州"]
WEATHERS = ["sunny", "cloudy", "rainy", "foggy"]
LIGHTINGS = ["day", "night"]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    city = rng.choice(CITIES, N).tolist()
    weather = rng.choice(WEATHERS, N, p=[0.55, 0.25, 0.15, 0.05]).tolist()
    lighting = rng.choice(LIGHTINGS, N, p=[0.7, 0.3]).tolist()

    epoch = pl.Series(rng.integers(0, DAYS * 86400, N), dtype=pl.Int64) + int(
        dt.datetime(2026, 1, 1).timestamp()
    )
    start_time = pl.from_epoch(epoch, time_unit="s")
    date_list = start_time.dt.date().to_list()

    clip_id = [f"clip_{i:07d}" for i in range(N)]

    # 路径里重复带了城市和日期 —— 这是后面「路径列冗余」实验的靶子
    cam_front = [f"s3://drive/{c}/{d:%Y%m%d}/{i}/cam_front.mp4"
                 for c, d, i in zip(city, date_list, clip_id)]
    lidar_top = [f"s3://drive/{c}/{d:%Y%m%d}/{i}/lidar_top.bin"
                 for c, d, i in zip(city, date_list, clip_id)]

    df = pl.DataFrame({
        "clip_id": clip_id,
        "city": city,
        "weather": weather,
        "lighting": lighting,
        "clip_date": start_time.dt.date(),
        "start_time": start_time,
        "duration_s": rng.integers(20, 60, N),
        "speed_kmh": np.round(np.abs(rng.normal(45, 18, N)), 1),
        "label_count": rng.poisson(15, N),
        "ego_x": np.round(rng.uniform(-20000, 20000, N), 2),
        "ego_y": np.round(rng.uniform(-20000, 20000, N), 2),
        "cam_front_path": cam_front,
        "lidar_top_path": lidar_top,
    })

    print(f"生成 {df.height:,} 条 clip")
    df.write_csv(DATA / "index.csv")
    df.write_parquet(DATA / "index.parquet", compression="zstd")
    df.write_parquet(DATA / "index_partitioned", partition_by=["clip_date", "city"])

    part_files = len(list((DATA / "index_partitioned").rglob("*.parquet")))
    print(f"index.csv            {(DATA / 'index.csv').stat().st_size / 1024**2:>8.1f} MB")
    print(f"index.parquet        {(DATA / 'index.parquet').stat().st_size / 1024**2:>8.1f} MB")
    print(f"index_partitioned    {part_files} 个文件")


if __name__ == "__main__":
    main()
