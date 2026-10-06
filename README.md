# PCDRec（Stage-0：SASRec CE 基线）

偏好一致性核验（PCS）+ 对齐蒸馏进 SASRec 的代码仓。**本批只交付 P0 前半**：用户 Beauty LOO 数据加载校验 + SASRec 全库 CE 可训可评。

- 主投 RecSys；**线上 0 LLM calls**
- 主数据：用户已处理的 Amazon Beauty LOO（**禁止重新下载 5-core**）
- **禁止编造实验结果**：`results/` 下的指标只能由训练/评估脚本写出；未跑通的主表标「待跑 / TODO」

## 环境

```bash
git clone https://github.com/ttstr/paper_material.git pcdrec && cd pcdrec
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 无 GPU 时
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
# 精确复现环境：pip install -r requirements-lock.txt（pip freeze 生成，Python 3.13.5）
```

当前本机探测（安装后实测）：

| 项 | 值 |
|---|---|
| Python | 3.13.5 |
| torch | 2.14.1+cpu |
| `torch.cuda.is_available()` | **False**（无 GPU） |

有 GPU 时安装对应 CUDA wheel，`configs/model/*.yaml` 中 `device: auto` 即可。

## 数据（不入库；一键从官方来源重建）

数据为 **RecBoard 的 `Amazon2014Beauty_550_LOU`**：由 **FreeRec 0.9.7** 原版 CLI 从 Zenodo 上的 RecBoard/FreeRec Atomic 包
（`Amazon2014Beauty.zip`，record 10995912，MD5 `ed0f0cfe2c44bbed899ca3da3aaf2918`）构建：

```
freerec make Amazon2014Beauty --root <dir> --kcore4user 5 --kcore4item 5 --splitting LOU
```

（用户/物品 5-core、评分阈值 0、按用户时间留一：最后一条 test、倒数第二条 valid。）FreeRec 输出的
`train/valid/test/item.txt` 即本仓的 `train/valid/test/item_meta.txt`。

**一键重建 + sha256 校验**（自动建独立 FreeRec 环境 `.freerec-venv`、下载 Zenodo 包并校验 MD5/SHA256、运行 FreeRec、审计切分、写 `manifest.json` / `build_info.json`，sha256 与参考值不一致即失败）：

```bash
bash scripts/00_build_beauty_from_raw.sh                       # 输出到 data/raw/amazon-beauty（默认）
PCDREC_DATA_DIR=/path/to/beauty bash scripts/00_build_beauty_from_raw.sh   # 自定义输出目录
bash scripts/00_build_beauty_from_raw.sh --archive /path/Amazon2014Beauty.zip # 离线：用本地 zip
```

本机已实测：从 Zenodo 下载 → FreeRec 0.9.7 → 四个文件与实验所用文件 **sha256 逐字节一致**（参考文件在 Windows 上生成，
记录分隔符为 CRLF；脚本默认 `--line-endings crlf` 复现之，`--line-endings lf` 输出 Linux 原生换行，两套参考哈希都内置校验）。
脚本：`scripts/00_build_beauty_from_raw.sh`（入口）+ `scripts/data_build/build_beauty_from_recboard.py`（仅标准库；改编自本项目作者最初的数据准备脚本
`build_recboard_data.py` / `prepare_data.py`，不重写任何过滤/切分逻辑）。依赖版本见 `requirements-freerec.txt`。

数据路径**可配置**：`configs/data/beauty.yaml` 中均为仓库相对路径（默认 `data/raw/amazon-beauty`、`data/processed/beauty`）；
环境变量 `PCDREC_DATA_DIR` 覆盖原始数据目录，`PCDREC_PROCESSED_DIR` 覆盖处理缓存目录。已有数据目录也可用
`PCDREC_DATA_DIR=/path bash scripts/00_link_user_data.sh` 软链到默认位置。

- 用户 22363；物品 12101；交互 198502（train 153776；valid/test 各 22363，每用户 1 条）
- ID 为 FreeRec 编码的连续整数（非原始 reviewer id / ASIN）；评测默认**全库排序**
- 文本字段：`title + categories + brand`（无 description）
- 加载时再次校验 `manifest.json` 的 sha256 与行数；处理后产物（本地，已 gitignore）：`data/processed/beauty/`

```bash
bash scripts/01_load_splits.sh
```

## 一键命令

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/src

# 1) 重建数据（或 PCDREC_DATA_DIR 指向已有目录）+ 校验/导出 processed
bash scripts/00_build_beauty_from_raw.sh
bash scripts/01_load_splits.sh

# 2) 单元测试
pytest -q

# 3) CPU smoke：1000 用户（写出真实 metrics；标注 subset）
MAX_USERS=1000 EPOCHS=10 BATCH=128 bash scripts/04_train_sasrec.sh

# 4) 评估 / 导出在线权重（无 LLM）
bash scripts/05_eval.sh results/checkpoints/sasrec_subset1000_best.pt
python -m pcdrec.export_online \
  --checkpoint results/checkpoints/sasrec_subset1000_best.pt \
  --out-dir results/online_export
```

### 全量训练（CPU，5 seeds）

```bash
# 单个 seed（CPU 上 4 线程最快：~23 s/epoch；8 线程反而更慢 ~35 s/epoch）
TQDM_DISABLE=1 FULL=1 EPOCHS=200 PATIENCE=20 BATCH=256 THREADS=4 SEED=42 TAG=sasrec_full_s42 \
  bash scripts/04_train_sasrec.sh > logs/sasrec_full_s42.log 2>&1

# 其余 seeds：2 条队列并行，各 3 线程（~26 s/epoch/进程，RSS ~0.9 GB/进程）
THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 43 45 > logs/queue_a.log 2>&1 &
THREADS=3 nohup bash scripts/run_sasrec_full_seeds.sh 44 46 > logs/queue_b.log 2>&1 &

# 汇总（只读 results/sasrec_full_s*_metrics.json，数字全部由脚本产出）
python scripts/06_tables.py
```

`04_train_sasrec.sh` 新增可选环境变量：`SEED` / `PATIENCE` / `THREADS` / `LR`（对应 `train.py` 的 `--seed/--patience/--threads/--lr`），全部写入结果 JSON 的 `config`。
每个 epoch 的 `train_sec/epoch_sec` 记录在 JSON `history` 中。

## 当前结果

**主表（SASRec-CE 基线，Beauty LOO 全量 22363 用户，全库 12101 物品排序，不过滤历史物品；CPU、无 GPU；5 seeds = 42–46，mean ± std）**：

- `results/main_table_sasrec_full.md`（可读表）
- `results/main_table_sasrec_full.csv`（每 seed 一行 + mean / std 行）
- 由 `scripts/06_tables.py` 从 `results/sasrec_full_s{42..46}_metrics.json` 自动生成；**README 不抄录数字，以表为准**。
- 训练日志：`logs/sasrec_full_s*.log`；checkpoint：`results/checkpoints/sasrec_full_s*_best.pt`（checkpoint 不入库）
- 硬件：8 核 CPU、~15 GB 内存、无 GPU（torch CPU）；seed 42 单独跑（4 线程），43–46 两进程并行（各 3 线程）。每 epoch 耗时、每 seed 耗时与总耗时同样由脚本写入汇总表。
- 重新生成：`python scripts/06_tables.py`

### 全量结果量级核查（排查记录）

seed=42 首次全量即落在公开 SASRec-CE（Beauty 5-core LOO、全库排序）报告的量级内（公开参考：phonism/genrec README SASRec(CE) N@10≈0.042 / R@10≈0.085；genrec 文档 N@10≈0.0375 / R@10≈0.069；sota2 汇总的 SASRec-SCE LOO N@10≈0.054），**因此未对模型 / 评测做数值性修正**。仍逐项核对了常见问题：

| 检查项 | 结论 |
|---|---|
| padding mask | 左 padding，`pad_id=n_items`；attention 屏蔽 pad key + 未来位置（causal）；pad 位置输出置零；loss `ignore_index` 忽略 pad 目标 |
| 位置编码 | 可学习 `pos_emb(L)`，左 padding 下最近物品恒在位置 L-1；用户表示取最右非 pad 位置 |
| 评测排除 padding item | 打分只用 `item_emb.weight[:-1]`（12101 个真实物品），pad 不参与排序 |
| 历史物品过滤 | **不过滤**（保持协议不变）；valid 用 train 历史，test 用 train+valid 历史 |
| 学习率 / dropout | Adam lr=1e-3、dropout 0.2（与 SASRec/RecBole/genrec 常用设置一致），收敛曲线正常（valid NDCG@10 先升后平/缓降，patience=20 早停；各 seed 的 best_epoch / epochs_run 见汇总表） |

唯一的代码改动是**效率**而非数值：训练时只在非 pad 位置计算全库 logits（`h[valid] @ E^T` + CE），与原先 `[B, L, n_items]` 上 `CE(ignore_index=-100, mean)` 数学等价，避免为大量 pad 位置分配 ~0.6 GB 的 logits。早停 patience 由 5 改为 20（写入 `configs/model/sasrec.yaml` 与结果 JSON），epoch 上限 200。

## 默认超参（SASRec CE）

2 layers，2 heads，`d=64`，`max_len=50`，dropout 0.2，全库 softmax CE，Adam lr=1e-3，batch 256，epoch 上限 200，早停看 valid **NDCG@10**（patience=20），seed=42（主表另跑 43–46）。

## 验收状态（本批）

| 项 | 状态 |
|---|---|
| 数据 sha256/行数/LOO 不变量 | 通过（`01_load_splits` + pytest） |
| `test_split` / `test_no_leakage` / `test_metrics` / `test_online_no_llm` | 通过（11 passed） |
| SASRec CE smoke 可训可评 | 通过（CPU，`max_users=1000`） |
| 全量主表（SASRec-CE，CPU，5 seeds） | 已跑，见 `results/main_table_sasrec_full.md` |
| LLM 离线 / PCS / 蒸馏 | **未实现**（目录 stub） |

Smoke 真实指标见脚本产物（勿手改）：

- `results/sasrec_subset1000_metrics.json`
- `results/sasrec_subset1000_metrics.csv`
- checkpoint：`results/checkpoints/sasrec_subset1000_best.pt`

## 尚未实现

- `llm_offline/`、`verify/`（PCS）、`encode/`、蒸馏损失、其它基线
- `configs/method/pcdrec.yaml` 仅为占位（`stage: 0`）

## 结果纪律

- 只信任 `results/*.json` / `*.csv` 中由 `train.py` / `evaluate.py` 写出的数字
- README **不编造 / 不抄录**指标；全量主表以 `scripts/06_tables.py` 产出的表为准
- 入库的结果证据仅限：`results/main_table_sasrec_full.{md,csv}`、`results/sasrec_full_s*_metrics.{json,csv}`、`logs/sasrec_full_s*.log`
- 不入库（见 `.gitignore`）：原始 TSV / `data/raw`（符号链接）/ `data/processed`、所有 checkpoint（`*.pt`）、`results/online_export*`、`.venv`、其它日志
