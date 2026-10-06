# PCDRec：偏好一致性核验 + 对齐蒸馏进 SASRec

偏好一致性核验（PCS）+ 对齐蒸馏进 SASRec 的代码仓（RecSys 投稿）。目前包含：

- 数据：从官方 Zenodo 包 + FreeRec 0.9.7 一键重建 RecBoard `Amazon2014Beauty_550_LOU`，sha256 逐字节校验
- 共享评测器（全库排序、不过滤历史物品）；非 LLM 基线 SASRec / GRU4Rec / BERT4Rec / SASRec-T / MF-BPR 的全量 5-seed 主表 + 配对显著性检验
- 离线 LLM 管线（画像 + 教师排序）、PCS 核验、蒸馏损失、学生训练、在线导出，全部有代码和单元测试；另有一次 CPU 小模型 / 小子集 pilot（**不是主表结果**）

约束：

- 主投 RecSys；**线上 0 LLM calls**
- **禁止编造实验结果**：`results/` 下的指标只能由训练/评估脚本写出；未跑通的部分标「待跑 / TODO」

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

### 全量基线（CPU，同一评测器，5 seeds）

所有方法共用 `pcdrec.train` + `pcdrec.evaluator`（全库 12101 物品排序，不过滤历史；valid 输入 = train 历史，test 输入 = train + valid 历史），
同容量（d = 64，max_len = 50），早停看 valid NDCG@10（patience 20，≤200 epoch；MF-BPR ≤300）。

| 方法 | 配置 | 训练目标 |
|---|---|---|
| SASRec | `configs/model/sasrec.yaml` | 全库 softmax CE（非 pad 位置） |
| GRU4Rec | `configs/model/gru4rec.yaml` | 1 层 GRU（hidden 64），全库 CE |
| BERT4Rec | `configs/model/bert4rec.yaml` | Cloze/MLM（mask ratio 0.2），评测时末尾追加 `[MASK]`（49 历史 + MASK） |
| SASRec-T | `configs/model/sasrec_t.yaml` | SASRec + 冻结 `all-MiniLM-L6-v2` 物品文本向量（Linear 384→64，与 id 嵌入相加），全库 CE |
| MF-BPR | `configs/model/mf_bpr.yaml` | BPR（每正例 1 个均匀负例），在框架内重实现 |

**调参预算（每个方法相同、很小）**：seed 42 上各试 2 个配置（序列模型 dropout ∈ {0.2, 0.5}；MF-BPR L2 ∈ {1e-5, 1e-4}），
按 valid NDCG@10 选一个，再跑 seeds 42–46。网格见 `configs/search/baselines.yaml`，选择结果由脚本写入 `results/search_summary.csv`。

```bash
# 文本向量缓存（SASRec-T 需要；写到 data/processed/beauty/text_emb/，不入库）
python -m pcdrec.encode.text_encoder
# 单个 run（例）
python -m pcdrec.train --model-config configs/model/gru4rec.yaml --seed 42 --threads 3 --tag gru4rec_d0.2_s42 --skip-process \
  --set hidden_dropout_prob=0.2
# 队列：每行 "<tag> <model> <seed> [--set k=v ...]"；auto_queue 在网格跑完后自动选配置并追加其余 seeds，维持 N 个 worker
bash scripts/auto_queue.sh      # N 由 logs/n_workers 指定（默认 2，每个 3 线程）
# 主表 + 配对检验（只读 results/*_metrics.json 与 results/per_user/*.npz）
python scripts/06_tables.py
# 旧的 SASRec-only 表
python scripts/06_tables.py --glob 'results/sasrec_full_s*_metrics.json' --out-stem results/main_table_sasrec_full
```

每个 run 写出 `results/<tag>_metrics.{json,csv}`（含每 epoch 耗时、配置、参数量）和 `results/per_user/<tag>.npz`（每用户 valid/test 排名，用于配对检验）。

### 离线 LLM → PCS → 蒸馏 → 在线导出

```bash
# 1) 离线 LLM：画像（K≤8 条带证据 id 的声明）+ 教师排序（P 次排列，M 个 Stage-0 候选含 GT）；JSON schema；可断点续跑的缓存
python -m pcdrec.llm_offline.run_offline --config configs/llm/qwen2.5-7b.yaml --out results/llm_qwen7b          # 论文默认：vLLM / GPU
python -m pcdrec.llm_offline.run_offline --config configs/llm/openai-compatible.yaml --out results/llm_api --n-users 200  # 密钥只读环境变量 PCDREC_LLM_API_KEY
# 2) PCS 核验（不把 LLM 当裁判：NLI 小模型 + 规则）：G / T / R / F、权重 w、负例 N1–N3、教师分布
python -m pcdrec.verify.run_verify --llm-dir results/llm_qwen7b
# 3) 学生（从 Stage-0 SASRec 初始化；L_rec + λ1·L_align(S-DPO, 冻结参考模型) + λ2·L_pref）
python -m pcdrec.distill.train_student --llm-dir results/llm_qwen7b --tag pcdrec_s42 --seed 42
python -m pcdrec.distill.train_student --llm-dir results/llm_qwen7b --tag a1_s42 --ablation configs/ablation/A1_naive_distill.yaml   # 消融开关
# 4) 在线导出（只保留 SASRec 主干）+ 检查（无 LLM import、导出模型复现学生指标、单请求延迟）
python -m pcdrec.export_online --student --checkpoint results/checkpoints/pcdrec_s42_best.pt --out-dir results/online_export
python scripts/check_online_export.py --export-dir results/online_export --student-metrics results/pcdrec_s42_metrics.json --out results
```

`run_offline` 对每次调用记录 prompt hash、模型名、解码参数、prompt/completion token 数、耗时；`ledger.json` 按阶段汇总。
`--cached-only` 不加载模型，只用已有缓存重建 profiles/rankings。消融开关 A1–A6 见 `configs/ablation/README.md`。

## 当前结果

**主表（非 LLM 基线，Beauty 全量 22363 用户，全库排序，5 seeds = 42–46，mean ± std）**：

- `results/main_table_beauty.md`（可读表：test 指标、所选配置、每 seed 训练耗时、调参记录、配对显著性检验）
- `results/main_table_beauty.csv`（每 seed 一行）、`results/significance_beauty.json`、`results/search_summary.csv`
- 由 `scripts/06_tables.py` 自动生成；**README 不抄录数字，以表为准**
- 训练日志：`logs/<tag>.log`；每用户排名：`results/per_user/*.npz`；checkpoint 不入库
- 硬件：8 核 CPU、~15 GB 内存、无 GPU；两个（后期三个）进程并行，各 3 线程（线程数记录在每个 run 的 JSON 中）

**CPU pilot（小模型、小子集 pilot，仅证明管线可运行，不是主表结果）**：`results/pilot/PILOT_REPORT.md`
（Qwen2.5-1.5B-Instruct、CPU、少量用户；含实测吞吐和全量 22363 用户在 CPU / 24 GB GPU 7B / API 上的成本估计，GPU 与 API 数字是标明出处的假设）。

### SASRec 量级核查（排查记录）


seed=42 首次全量即落在公开 SASRec-CE（Beauty 5-core LOO、全库排序）报告的量级内（公开参考：phonism/genrec README SASRec(CE) N@10≈0.042 / R@10≈0.085；genrec 文档 N@10≈0.0375 / R@10≈0.069；sota2 汇总的 SASRec-SCE LOO N@10≈0.054），**因此未对模型 / 评测做数值性修正**。仍逐项核对了常见问题：

| 检查项 | 结论 |
|---|---|
| padding mask | 左 padding，`pad_id=n_items`；attention 屏蔽 pad key + 未来位置（causal）；pad 位置输出置零；loss `ignore_index` 忽略 pad 目标 |
| 位置编码 | 可学习 `pos_emb(L)`，左 padding 下最近物品恒在位置 L-1；用户表示取最右非 pad 位置 |
| 评测排除 padding item | 打分只用 `item_emb.weight[:-1]`（12101 个真实物品），pad 不参与排序 |
| 历史物品过滤 | **不过滤**（保持协议不变）；valid 用 train 历史，test 用 train+valid 历史 |
| 学习率 / dropout | Adam lr=1e-3、dropout 0.2（与 SASRec/RecBole/genrec 常用设置一致），收敛曲线正常（valid NDCG@10 先升后平/缓降，patience=20 早停；各 seed 的 best_epoch / epochs_run 见汇总表） |

唯一的代码改动是**效率**而非数值：训练时只在非 pad 位置计算全库 logits（`h[valid] @ E^T` + CE），与原先 `[B, L, n_items]` 上 `CE(ignore_index=-100, mean)` 数学等价，避免为大量 pad 位置分配 ~0.6 GB 的 logits。早停 patience 由 5 改为 20（写入 `configs/model/sasrec.yaml` 与结果 JSON），epoch 上限 200。

## 默认超参

- SASRec CE：2 layers，2 heads，`d=64`，`max_len=50`，FFN 256，dropout 0.2（调参后见主表），全库 softmax CE，Adam lr=1e-3，batch 256，epoch 上限 200，早停看 valid **NDCG@10**（patience=20）
- 其它基线：同上（容量、优化器、早停一致），差异只在 `configs/model/*.yaml` 与上表
- PCDRec（`configs/method/pcdrec.yaml`）：M = 20，P = 3，h = 3，K ≤ 8，PCS = G^α·T^β·R^γ，w = sigmoid((PCS − θ)/T_w)，λ1 = λ2 = 0.1，S-DPO β = 1，m = 5 个负例，学生从 Stage-0 SASRec（seed 42）初始化

## 验收状态

| 项 | 状态 |
|---|---|
| 数据：Zenodo + FreeRec 0.9.7 一键重建，sha256 逐字节一致 | 通过（`scripts/00_build_beauty_from_raw.sh`） |
| 数据 sha256/行数/LOO 不变量 | 通过（`01_load_splits` + pytest） |
| 单元测试 | `pytest -q`（数量见 `supplementary/REPRODUCIBILITY.md`） |
| 非 LLM 基线全量主表（5 seeds，同一评测器）+ 配对检验 | 见 `results/main_table_beauty.md` |
| 离线 LLM / PCS / 损失 / 学生 / 在线导出 | 代码 + 单元测试完成；CPU pilot 见 `results/pilot/PILOT_REPORT.md`（**不是主表结果**） |
| 7B 教师全量离线运行、PCDRec 主结果、LLM 类基线（KAR / DLLM2Rec / RDRec 等） | **TODO**（需要 GPU 或 API 预算） |

Smoke 真实指标见脚本产物（勿手改）：`results/sasrec_subset1000_metrics.{json,csv}`

## 尚未实现

- LLM 类基线（KAR、DLLM2Rec、RDRec、UniSRec、Persona4Rec†、Ocean-feat†、在线 LLM 重排参考）
- A6（注入式）只部分接线；A7–A10、敏感性、分组分析、案例研究
- 7B 教师的全量离线产物与 PCDRec 主表结果

## 结果纪律

- 只信任 `results/` 下由 `train.py` / `evaluate.py` / `distill/train_student.py` / `verify/run_verify.py` / `llm_offline/run_offline.py` 写出的数字
- README **不编造 / 不抄录**指标；表格均由 `scripts/06_tables.py`、`scripts/pilot_report.py` 生成
- 入库的结果证据：主表与显著性（`results/main_table_beauty.*`、`results/significance_beauty.json`、`results/search_summary.csv`）、
  每 run 的 `results/*_s4[2-6]_metrics.{json,csv}`、`results/per_user/*.npz`、`logs/*_s4[2-6].log`、pilot 汇总（`results/pilot/` 下的汇总 JSON / 报告）
- 不入库（见 `.gitignore`）：原始数据（`.inter/.item/zip/TSV`）、`data/processed`（含文本向量缓存）、所有 checkpoint / 模型权重、
  在线导出、LLM 原始调用缓存（`calls.jsonl`，含物品文本）、`.venv`、其它日志
