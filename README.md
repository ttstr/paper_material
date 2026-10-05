# PCDRec（Stage-0：SASRec CE 基线）

偏好一致性核验（PCS）+ 对齐蒸馏进 SASRec 的代码仓。**本批只交付 P0 前半**：用户 Beauty LOO 数据加载校验 + SASRec 全库 CE 可训可评。

- 主投 RecSys；**线上 0 LLM calls**
- 主数据：用户已处理的 Amazon Beauty LOO（**禁止重新下载 5-core**）
- **禁止编造实验结果**：`results/` 下的指标只能由训练/评估脚本写出；未跑通的主表标「待跑 / TODO」

## 环境

```bash
cd /workspace/pcdrec
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 无 GPU 时
pip install -r requirements.txt
export PYTHONPATH=/workspace/pcdrec/src
```

当前本机探测（安装后实测）：

| 项 | 值 |
|---|---|
| Python | 3.13 |
| torch | 2.14.1+cpu |
| `torch.cuda.is_available()` | **False**（无 GPU） |

有 GPU 时安装对应 CUDA wheel，`configs/model/sasrec.yaml` 中 `device: auto` 即可。

## 数据路径（不入库）

原始 TSV **不复制进仓库**，只通过配置 / 符号链接指向：

```
/workspace/pcdrec-data/amazon-beauty/
  train.txt  valid.txt  test.txt  item_meta.txt  README.md  manifest.json
```

- 用户 22363；train 153776；valid/test 各 22363（每用户 1 条 LOO）
- meta 12101 物品；ID 从 0 连续重映射；评测默认**全库排序**
- 文本字段：`title + categories + brand`（无 description）
- 加载时校验 `manifest.json` 的 sha256 与行数

```bash
bash scripts/00_link_user_data.sh
bash scripts/01_load_splits.sh
```

处理后产物（本地，已 gitignore）：`data/processed/beauty/`。

## 一键命令

```bash
cd /workspace/pcdrec
source .venv/bin/activate
export PYTHONPATH=/workspace/pcdrec/src

# 1) 链接数据 + 校验/导出 processed
bash scripts/00_link_user_data.sh
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

### 全量训练（待跑 / TODO）

无 GPU 时全量 22363 用户 + 全库 softmax 很慢，**尚未作为主表结果提交**。有算力时：

```bash
FULL=1 EPOCHS=50 BATCH=256 TAG=sasrec_full bash scripts/04_train_sasrec.sh
```

主表 Beauty HR@10 / NDCG@10：**待跑**（勿手填）。

## 默认超参（SASRec CE）

2 layers，2 heads，`d=64`，`max_len=50`，dropout 0.2，全库 softmax CE，早停看 valid **NDCG@10**，seed=42。

## 验收状态（本批）

| 项 | 状态 |
|---|---|
| 数据 sha256/行数/LOO 不变量 | 通过（`01_load_splits` + pytest） |
| `test_split` / `test_no_leakage` / `test_metrics` / `test_online_no_llm` | 通过（11 passed） |
| SASRec CE smoke 可训可评 | 通过（CPU，`max_users=1000`） |
| 全量主表 | **待跑 / TODO** |
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
- README **不编造**指标；全量结果标「待跑」
