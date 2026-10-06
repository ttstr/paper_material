# Data Statement — PCDRec

## 1. Dataset and source
- **Amazon Beauty** (category "Beauty"), from the Amazon product review data (2014 version) released by McAuley et al. (SIGIR 2015, "Image-based Recommendations on Styles and Substitutes") and He & McAuley ("Ups and Downs", WWW 2016). Public page: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html (checked 2026-10-06 UTC+8); it asks users to "cite one or both" of the two papers.
- **Exact benchmark used: RecBoard `Amazon2014Beauty_550_LOU`**, built with the unmodified **FreeRec 0.9.7** CLI from the RecBoard/FreeRec Atomic archive (refs below):
  - input archive: `Amazon2014Beauty.zip` (`Amazon2014Beauty.inter` + `Amazon2014Beauty.item`) from Zenodo record 10995912 (DOI 10.5281/zenodo.10995912), URL `https://zenodo.org/records/10995912/files/Amazon2014Beauty.zip`, as registered in FreeRec's data registry; MD5 `ed0f0cfe2c44bbed899ca3da3aaf2918` (published by Zenodo), SHA256 `2895cc3f956a2844782cf2ac494da0efda578005674042ad5aa715ad9943c95f`;
  - command (as listed by RecBoard): `freerec make Amazon2014Beauty --root <dir> --kcore4user 5 --kcore4item 5 --splitting LOU`, i.e. iterative 5-core on users and items, rating threshold 0 (every rating is an interaction), leave-one-out per user by timestamp (last → test, second-to-last → valid, rest → train);
  - FreeRec output `train.txt / valid.txt / test.txt / item.txt` = our `train.txt / valid.txt / test.txt / item_meta.txt` (`item.txt` renamed).
  - References: RecBoard metadata https://github.com/MTandHJ/RecBoard/blob/master/benchmark/Amazon2014Beauty_550_LOU/meta.json ; FreeRec registry https://github.com/MTandHJ/freerec/blob/master/freerec/data/registry.json ; Zenodo https://zenodo.org/records/10995912 .
- **Provenance verified (2026-10-06 UTC+8):** rebuilding from the Zenodo archive with FreeRec 0.9.7 (pandas 2.3.3, numpy 2.5.0) on Linux reproduces all four files **byte-for-byte** (sha256 equal to `manifest.json`) once record terminators are written as CRLF; the reference files were produced on Windows, where pandas writes CRLF record terminators (newlines inside quoted titles stay LF). Content is identical either way; both CRLF and LF reference hashes are checked by the build script.
- Statistics: 22,363 users; 12,101 items (metadata covers every interacted item); 198,502 interactions (train 153,776 / valid 22,363 / test 22,363); average sequence length 8.88 (train part 6.88; min 3, median 4, max 202); density 0.073%. Timestamps 2002-06 to 2014-07.

## 2. License / terms of use
- The Zenodo record of the Atomic archive is labelled **CC BY 4.0** (checked 2026-10-06); this covers the repackaging. The underlying Amazon review data page asks users to cite the papers but **states no explicit license**; common practice treats it as research-only. **TODO: verify the current terms** with the dataset maintainers before any public release; we do not redistribute data.
- Consequence for this project: **the repository contains no raw or processed Amazon data** (no `.inter/.item`, no zip, no TSV). `.gitignore` excludes `data/raw/`, `data/processed/`, `data/recboard_build/`, `**/amazon-beauty/`, `*.zip`, `*.inter`, `*.item`, `train.txt`, `valid.txt`, `test.txt`, `item_meta.txt`, `*.tsv`, `*.pkl`, `*.parquet`, `*.npy`.
- The repository ships: code (including the data build script), configs, sha256 reference hashes, aggregate metrics, per-user ranks (integers, keyed by anonymised FreeRec user ids, no interaction records) and training logs.

## 3. How to obtain the data yourself (one command)
```bash
bash scripts/00_build_beauty_from_raw.sh            # -> data/raw/amazon-beauty
# or: PCDREC_DATA_DIR=/any/dir bash scripts/00_build_beauty_from_raw.sh
# offline: ... --archive /path/to/Amazon2014Beauty.zip
```
The script (i) creates an isolated FreeRec environment (`.freerec-venv`, pins in `requirements-freerec.txt`: freerec 0.9.7, pandas 2.3.3, numpy 2.5.0, polars 1.44.2, torchdata 0.7.0 `--no-deps`, CPU torch); (ii) downloads the archive from Zenodo and checks MD5/SHA256; (iii) runs the FreeRec command above; (iv) writes the four TSVs + `manifest.json` (sha256, rows, users, items, timestamp range) + `build_info.json` (versions, command, audit) and **fails if any sha256 differs from the reference**; (v) audits the split (one valid/test per user, no duplicate user–item pairs, per-user time order train ≤ valid ≤ test, sizes equal RecBoard metadata). Implementation: `scripts/data_build/build_beauty_from_recboard.py` (stdlib only), adapted from the original data-preparation scripts of this project's authors.
Then `bash scripts/01_load_splits.sh` validates the manifest again and builds the processed cache. Training code reads `configs/data/beauty.yaml` (`raw_dir: data/raw/amazon-beauty`, repo-relative) or the `PCDREC_DATA_DIR` environment variable.

## 4. Preprocessing used in the paper
- No further filtering or id remapping beyond FreeRec's (ids are FreeRec's contiguous integer encodings, not reviewer ids / ASINs). Sequences are built by timestamp; ties keep file order.
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
