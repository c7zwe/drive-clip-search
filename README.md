# drive-clip-search

一个可运行的小项目，把「元信息表 → 并行处理 → 向量检索」这条链路走一遍。

数据是随机生成的模拟路采元信息，配公开图片。目的是在没有真实数据集的情况下，也能复现每一步的实测结果。

## 模块

| 文件 | 作用 |
|---|---|
| `src/ingest.py` | 生成 100 万条 clip 元信息，输出 CSV / Parquet / 分区 Parquet |
| `src/bench_io.py` | CSV 与 Parquet 的读取对比，含分区裁剪和小文件场景 |
| `src/features.py` | 用 Ray 并行处理逐条 clip，对比三种派发方式 |
| `src/embed.py` | 图片 → CLIP 512 维向量 → LanceDB |
| `src/search.py` | 文字检索，输出 HTML 结果页 |

## 实测结果

环境：Windows 11，i7-13650HX（14 核 20 线程），16 GB。

### 列存与行存（100 万行 × 13 列）

| 操作 | CSV | Parquet |
|---|---|---|
| `count(*)` | 146 ms | 1 ms |
| 单列求均值 | 146 ms | 4 ms |
| 过滤 城市 + 天气 | 153 ms | 3 ms |
| 按城市分组 | 155 ms | 7 ms |
| 落盘体积 | 195.9 MB | 28.1 MB（ZSTD） |

### 分区不是越细越好

同样查「某天 + 某城市」：

| | 打开文件数 | 耗时 |
|---|---|---|
| 单文件 Parquet | 1 | 3 ms |
| 按 日期 × 城市 分区 | 155 | 23 ms |

文件打开的开销超过了跳过无关数据省下的时间。单文件控制在 100 MB ~ 1 GB 比较常见。

### min/max 跳块需要数据有序

| | 每块 city 的 min~max | 查询耗时 |
|---|---|---|
| 原始随机顺序 | 8 块全部是 `上海 ~ 深圳` | 3.3 ms |
| 按城市排序后 | `上海~上海` … `深圳~深圳` | 2.4 ms |

乱序时每块的取值范围覆盖全部城市，一块都跳不了。

### Ray 并行（1 万条 clip）

| 方式 | 耗时 | 加速 |
|---|---|---|
| 串行 | 103.16 s | 1x |
| 逐条派发（10000 个任务） | 12.26 s | 8.4x |
| 分块派发（200 个任务） | 10.33 s | 10.0x |

分块比逐条快 16%，差的是调度开销。理论加速上限是物理核数 14，不是线程数 20。

### 向量检索

CLIP ViT-B-32，512 维，CPU 上约 24 ms/张。120 张图两两余弦：最小 0.160，最大 0.945，平均 0.437。

暴力检索瓶颈在内存带宽：100 万 × 512 维是 2 GB 向量，每次查询都要搬一遍。

## 运行

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/ingest.py
python src/bench_io.py
python src/features.py
python src/embed.py
python src/search.py "a rainy street" "a dog"
```

`search.py` 不带参数时使用内置的默认查询词。结果写入 `result.html`。

CLIP ViT-B-32 使用英文图文对训练，中文查询无法识别。

## 设计取舍

- 用随机数据而非真实路采数据：真实数据有版权和体积问题，而 schema 与查询模式按真实平台设计。性能结论来自实测，换数据集量级不变。
- 用 LanceDB 而非 Milvus：Milvus 需要 Docker 和服务端，LanceDB 是嵌入式库。向量检索相关的概念一致。
- 图片来自 picsum.photos：免注册的公开图库。换成 nuScenes 的路采帧只需要替换 `download_images()`。

## 已知问题

1. `open_clip.create_model_and_transforms` 返回 `(模型, 训练预处理, 推理预处理)`。误用训练版（带随机裁剪）会让同一张图每次算出不同向量。
2. LanceDB 的 `_distance` 是 L2 距离的平方，不是距离本身。单位向量下 `cos = 1 - _distance / 2`，写成 `1 - _distance² / 2` 会得到错误的相似度。
3. 类别字段不能直接映射成 0/1/2/3 放进向量，余弦会把它当作真实距离。应使用 one-hot 或由模型学习。
4. `Path.write_bytes` 只接受字节，文本需要先 `encode`。

## 局限

- 数据为合成，不含真实驾驶语义
- 元信息表与图片之间没有真正关联，真实系统应先按元信息过滤再做向量检索
- 未建 ANN 索引（IVF / HNSW），当前是暴力扫描
- 单机运行，未涉及多机调度与容错
