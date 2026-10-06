# Data Statement — PCDRec

## 1. Dataset and source
- **Amazon Beauty** (category "Beauty"), from the Amazon product review data released by McAuley et al. (SIGIR 2015, "Image-based Recommendations on Styles and Substitutes") and He & McAuley ("Ups and Downs", 2016). Public page for the 2014 version: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html (checked 2026-10-06 UTC+8). The page says this older version is kept "mainly ... for the sake of reproducing past results" and asks users to "cite one or both" of the two papers above.
- The experiments use a **pre-processed leave-one-out version supplied to the authors**: TSV files `train.txt`, `valid.txt`, `test.txt`, `item_meta.txt` plus a `manifest.json` with sha256 hashes.
  - Its size is **consistent with the public Beauty 5-core file**: train + valid + test = 153,776 + 22,363 + 22,363 = 198,502 interactions, which equals the 198,502 reviews listed for "Beauty 5-core" on the page above. Timestamps run from 2002-06 to 2014-07 (manifest `ts_min`/`ts_max`). TODO: confirm the exact provenance and preprocessing script of the supplied files.
- Statistics: 22,363 users; 12,101 items (metadata covers every interacted item); average sequence length 8.88 (train part 6.88; min 3, median 4, max 202); density 0.073%.

## 2. License / terms of use
- The dataset page asks users to cite the papers but **does not state an explicit license** (checked 2026-10-06). Common practice treats the data as research-only. Underlying content (reviews, product metadata) originates from Amazon. **TODO: verify the current terms** with the dataset maintainers before any public release, and do not redistribute.
- Consequence for this project: **the repository contains no raw or processed Amazon data.** `.gitignore` excludes `data/raw/`, `data/processed/`, `**/amazon-beauty/`, `train.txt`, `valid.txt`, `test.txt`, `item_meta.txt`, `*.tsv`, `*.pkl` and `*.parquet`. `data/raw/amazon-beauty` is only a local symlink created by `scripts/00_link_user_data.sh`.
- The repository may ship: configs, the sha256 manifest summary (hashes and row counts only), code, aggregate metrics (`results/*metrics*`, main table) and per-seed training logs. These contain no review text and no per-user records.

## 3. How to obtain the data yourself
1. Download the **Beauty 5-core** reviews and the **Beauty metadata** from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html (2014 version) and cite the papers listed there.
2. Convert the files into the four TSVs (tab-separated):
   - interactions `USER ITEM RATING TIMESTAMP`, with ids remapped to contiguous integers from 0;
   - leave-one-out per user: last interaction → `test.txt`, second-to-last → `valid.txt`, rest → `train.txt`; sort by timestamp and keep file order for ties; every user has ≥ 3 training interactions;
   - metadata `ITEM TITLE SALES_TYPE SALES_RANK CATEGORIES PRICE BRAND`.
   **TODO: the conversion script is not in the repository yet.** Add `scripts/00_prepare_amazon_beauty.py` (raw json → TSV + manifest) and check that its sha256 hashes match `manifest.json`.
3. Put the files in any directory, then `export PCDREC_DATA=/path/to/amazon-beauty`, update `configs/data/beauty.yaml` (`raw_dir`, `manifest`), and run `bash scripts/00_link_user_data.sh && bash scripts/01_load_splits.sh`. The loader checks sha256 hashes, row counts and LOO invariants.

## 4. Preprocessing used in the paper
- No further filtering or id remapping (already done in the supplied files). Sequences are built by timestamp.
- Item text = `title + categories + brand` (no description or review text in this version), truncated to about 128 tokens.
- Evaluation: full-catalog ranking over 12,101 items; history items are not filtered.
- Temporal hold-out for LLM inputs: profiles use only `i_1 .. i_{n-2-h}` (h = 3). The last h training items are consistency probes. Valid/test items **never** enter any LLM prompt (`tests/test_no_leakage.py`).

## 5. LLM usage and privacy
- LLMs are used **offline only** (profile extraction, teacher ranking, evaluation-only profiles). Serving makes **0 LLM calls/request** and reads no LLM artifacts (`tests/test_online_no_llm.py`).
- Planned teacher: open-weight Qwen2.5-7B-Instruct run locally with vLLM. **By default no closed API is used**, so no interaction data leaves the machine. If an API is ever used, document the provider, its data-retention terms and the exact fields sent (TODO).
- Prompts contain only public product metadata (title/categories/brand) of a user's own past items, keyed by anonymized integer ids. No reviewer names, review text or Amazon reviewer ids are used.
- Full LLM output caches are **not** released (size, and possible re-identification through long histories). Only a few de-identified examples and the JSON schema will be shared (TODO).
- No human-subjects study has been run. If the optional claim-annotation study is done, describe annotator recruitment, pay and any ethical review or exemption (TODO).
- Known biases: Amazon 5-core filtering over-represents active users and popular items. LLM profiles may inherit stereotypes from product categories (e.g., gendered beauty products). PCS grounding reduces unsupported claims but does not audit for bias (TODO: discuss in the paper's limitations).
