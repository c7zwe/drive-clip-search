# drive-clip-search

一个最小可跑的「智驾数据检索」玩具项目：**元信息表 + 并行处理 + 向量检索**。

用随机生成的路采元信息和公开照片，把数据平台里最核心的三件事走一遍，并且**每一步都有实测数据**，不是"书上说 Parquet 更快"。

---

## 它做了什么

| 模块 | 干什么 | 关键结论 |
|---|---|---|
| `src/ingest.py` | 造 100 万条 clip 元信息，写成 Parquet（分区 + 排序） | 列存 + 排序 + 分区，体积压到 CSV 的 14% |
| `src/bench_io.py` | CSV vs Parquet 的 6 组对比 | 读 1 列快 35 倍；分区切太碎会**变慢** |
| `src/features.py` | 用 Ray 把逐条处理并行化 | 1 万条任务：103s → 10.3s（**10 倍**） |
| `src/embed.py` | 图片 → CLIP 512 维向量 → LanceDB | CPU 上 24 ms/张 |
| `src/search.py` | 一句话搜图 → HTML 结果页 | 单次查询几十毫秒 |

---

## 实测数据（Windows 11 / i7-13650HX 14核20线程 / 16GB）

### 1. 列存 vs 行存（100 万行 × 13 列）

| 实验 | CSV | Parquet | 说明 |
|---|---|---|---|
| 全表 `count(*)` | 146 ms | **1 ms** | Parquet 只读文件尾部的元数据 |
| 只读 1 列求均值 | 146 ms | **4 ms** | CSV 要解析全部 13 列 |
| 过滤 城市+天气 | 153 ms | **3 ms** | 同上 |
| 按城市分组聚合 | 155 ms | **7 ms** | 同上 |
| 落盘体积 | 195.9 MB | **28.1 MB** | ZSTD 压缩 |

### 2. 小文件问题（分区不是越细越好）

同样查「某天 + 某城市」：

| | 打开几个文件 | 耗时 |
|---|---|---|
| 单文件 Parquet | 1 | **3 ms** |
| 按「日期 × 城市」分区 | 155 | **23 ms**（反而更慢） |

155 次文件打开的开销，超过了跳过无关数据省下的时间。**单文件控制在 100MB~1GB 才是工业界的常规做法。**

### 3. min/max 跳块只有数据有序才管用

| | 每块 city 的 min~max | 查询耗时 |
|---|---|---|
| 原始随机顺序 | 8 块全是 `上海 ~ 深圳` | 3.3 ms |
| 按城市排序后 | `上海~上海` … `深圳~深圳` | **2.4 ms** |

乱序数据每块的取值范围覆盖全部城市，**一块都跳不了**。排序或分区才能让 min/max 生效。

### 4. Ray 并行（1 万条 clip 的画面预处理）

| 方案 | 耗时 | 加速 |
|---|---|---|
| 串行，1 个核 | 103.16 s | 1x |
| Ray 逐条，10000 个任务 | 12.26 s | 8.4x |
| Ray 分块，200 个任务 | **10.33 s** | **10.0x** |

任务粒度太小会被调度开销吃掉：**分块比逐条快 16%**。
理论加速上限是物理核数（14），不是线程数（20）——剩下的是内存墙。

### 5. 向量检索

- CLIP ViT-B-32，512 维，CPU 上 **24 ms/张**
- 120 张图两两余弦：最小 0.160 / 最大 0.945 / 平均 0.437（分布健康）
- 暴力检索的成本是**内存带宽**，不是算力：100 万 × 512 维 = 2 GB 向量，每次查询都要搬一遍

---

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/ingest.py        # 造数据（约 1 分钟）
python src/bench_io.py      # 6 组 IO 对比
python src/features.py      # Ray 并行（约 2 分钟）
python src/embed.py         # 下载图片 + 提向量 + 建库
python src/search.py        # 查询，生成 result.html
```

也可以自己指定查询词：

```bash
python src/search.py "a rainy street" "a dog" "a tunnel"
```

> CLIP ViT-B-32 是拿**英文**图文对训练的，中文查询它看不懂。要中文搜图得换 Chinese-CLIP。

---

## 目录结构

```
drive-clip-search/
├── src/
│   ├── ingest.py       造元信息表 → Parquet（分区 + 排序）
│   ├── bench_io.py     CSV vs Parquet 六组实验
│   ├── features.py     Ray 串行 / 逐条 / 分块 三种跑法
│   ├── embed.py        图片 → CLIP 向量 → LanceDB
│   └── search.py       文字检索 → HTML
├── data/               生成物（不进 git）
├── requirements.txt
└── result.html         检索结果页（生成物，不进 git）
```

---

## 几个刻意的设计取舍

- **用随机数据而不是真实路采数据**：真实数据有版权和体积问题，而 schema 和查询模式是照着真实平台设计的。所有性能结论都来自真实测量，换个数据集量级不变。
- **不用 Milvus 而用 LanceDB**：Milvus 需要 Docker + 服务端，LanceDB 是嵌入式、`pip install` 就能跑。检索的概念（向量、归一化、top-k、L2 与余弦的关系）完全一致。
- **图片来自 picsum**：免注册的公开图库。换成 nuScenes 的路采帧，只需要替换 `download_images()`。

---

## 踩过的坑

1. **`open_clip.create_model_and_transforms` 返回的是 `(模型, 训练预处理, 推理预处理)`**，顺序容易记错。用训练版（带随机裁剪）做推理，每次算出来的向量都不一样。
2. **LanceDB 的 `_distance` 返回的是「L2 距离的平方」，不是距离本身**。单位向量下 `cos = 1 - _distance/2`，我第一次写成 `1 - _distance²/2`，结果全变成负分。
3. **类别不能直接映射成 0,1,2,3 塞进向量**。余弦会以为「上海(0) 和北京(1) 很近」，这是假信号。要么 one-hot，要么交给模型学。
4. **`write_bytes` 只收字节**。图片用 `.content` 直接写，文字要先 `.encode("utf-8")`。

---

## 局限

- 数据是合成的，不含真实的驾驶语义
- 图片和元信息表没有真正的关联（真实系统里应该先按元信息过滤，再做向量检索）
- 只用了暴力扫描，没建 ANN 索引（IVF / HNSW）
- 单机，没涉及多机调度和容错
