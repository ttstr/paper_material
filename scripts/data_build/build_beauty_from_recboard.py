#!/usr/bin/env python
"""Rebuild the PCDRec Beauty data (RecBoard ``Amazon2014Beauty_550_LOU``) from the official source.

Pipeline (stdlib only; FreeRec runs in its own interpreter, see requirements-freerec.txt):
  1. obtain the RecBoard/FreeRec Atomic archive ``Amazon2014Beauty.zip`` from Zenodo
     (record 10995912, the URL registered in FreeRec's data registry) or a local copy,
     and verify its MD5 (published by Zenodo) and SHA256;
  2. run the unmodified FreeRec 0.9.7 CLI exactly as listed on RecBoard:
       freerec make Amazon2014Beauty --root <work> --kcore4user 5 --kcore4item 5 --splitting LOU
     (5-core users/items, rating threshold 0, leave-one-out per user);
  3. copy train/valid/test.txt and item.txt (-> item_meta.txt) to the output directory, optionally
     normalising record terminators (see --line-endings), audit the split, write manifest.json,
     and compare SHA256 with the reference files used for all PCDRec experiments.

About line endings: FreeRec writes with pandas ``to_csv``, whose record terminator is the
OS line separator. The reference files were produced on Windows (CRLF record terminators,
embedded newlines inside quoted titles left as LF). ``--line-endings crlf`` (default)
re-creates those bytes exactly from a Linux/macOS build; ``lf`` keeps FreeRec's native
Linux output (identical content; reference LF hashes are also checked).

Attribution: adapted from the original data-preparation scripts of this project's authors
(``build_recboard_data.py`` / ``prepare_data.py``), which call the FreeRec CLI rather than
re-implementing any filtering or splitting logic. Preprocessing itself is FreeRec's
(https://github.com/MTandHJ/freerec), benchmark definition is RecBoard's
(https://github.com/MTandHJ/RecBoard). Raw data: Amazon review data (2014), McAuley et al.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://zenodo.org/records/10995912/files/Amazon2014Beauty.zip"
REGISTRY_URL = "https://github.com/MTandHJ/freerec/blob/master/freerec/data/registry.json"
RECBOARD_META_URL = "https://github.com/MTandHJ/RecBoard/blob/master/benchmark/Amazon2014Beauty_550_LOU/meta.json"
ARCHIVE_NAME = "Amazon2014Beauty.zip"
ARCHIVE_MD5 = "ed0f0cfe2c44bbed899ca3da3aaf2918"
ARCHIVE_SHA256 = "2895cc3f956a2844782cf2ac494da0efda578005674042ad5aa715ad9943c95f"
ARCHIVE_MEMBERS = {"Amazon2014Beauty.inter", "Amazon2014Beauty.item"}
FREEREC_VERSION = "0.9.7"
# sha256 of freerec/data/preprocessing/base.py in freerec 0.9.7 (the filtering / LOU logic)
FREEREC_PREPROC_SHA256 = "621169008a68c03dbbec33cd80348356f7fa0bd2947b04683c1761d581fd4ed5"
DATASET_DIR = "Amazon2014Beauty_550_LOU"
FREEREC_ARGS = ["make", "Amazon2014Beauty", "--kcore4user", "5", "--kcore4item", "5", "--splitting", "LOU"]
FILE_MAP = {"train.txt": "train.txt", "valid.txt": "valid.txt", "test.txt": "test.txt", "item.txt": "item_meta.txt"}
# Reference hashes of the files used in all PCDRec experiments.
REFERENCE_SHA256 = {
    "crlf": {
        "train.txt": "73b8377e495a161f8e6809d8f2536becca33da0da03b956e2adae71c6bdc9399",
        "valid.txt": "5f489eec1fb4c5b69a833622ebf15c4f19c835450f545a306e2eeb134bd54e74",
        "test.txt": "138cc95a042712a71732344ded53cf8eb390769f2f6f4f6592cbde1ff71f268b",
        "item_meta.txt": "85b6c3efc958c1c57858eebcb2a1db12d1323d359ffebde493733a0143183b36",
    },
    "lf": {
        "train.txt": "88dc9bad08e059ad3ee1af6f6d7bf96086377689563f4c4f762b2a786d3c785d",
        "valid.txt": "f85f92445f22ecf672bab38b78d8b8eeee70e85252ad4dda3c0a58a764859ce5",
        "test.txt": "cff49e92f7fde52705b973906e23dfa90d3187b8090784d7d6862b0e9f19c7e1",
        "item_meta.txt": "390cac34a8df789892348eb8abb09a660fb5bdb81acb051151b8ca276701c179",
    },
}
EXPECTED_SIZES = {"n_users": 22363, "n_items": 12101, "n_interactions": 198502}


def digest(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def obtain_archive(cache_dir: Path, archive: Path | None, url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / ARCHIVE_NAME
    if archive is not None:
        src = Path(archive)
        if src.resolve() != target.resolve():
            shutil.copyfile(src, target)
        log(f"using local archive copy -> {target}")
    elif target.exists() and digest(target, "md5") == ARCHIVE_MD5:
        log(f"reusing cached archive {target}")
    else:
        part = target.with_suffix(".zip.part")
        log(f"downloading {url}")
        with urllib.request.urlopen(url + "?download=1", timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.replace(target)
    md5, sha = digest(target, "md5"), digest(target)
    if md5 != ARCHIVE_MD5 or sha != ARCHIVE_SHA256:
        raise SystemExit(f"archive checksum mismatch: md5={md5} sha256={sha}")
    with zipfile.ZipFile(target) as z:
        if set(z.namelist()) != ARCHIVE_MEMBERS or z.testzip() is not None:
            raise SystemExit(f"unexpected archive content: {z.namelist()}")
    log(f"archive OK (md5={md5})")
    return target


def inspect_freerec(python: str) -> dict:
    code = (
        "import json,hashlib,pathlib,importlib.metadata as m,freerec;"
        "p=pathlib.Path(freerec.__file__).parent/'data'/'preprocessing'/'base.py';"
        "print(json.dumps({'freerec_version':m.version('freerec'),'pandas_version':m.version('pandas'),"
        "'numpy_version':m.version('numpy'),'polars_version':m.version('polars'),"
        "'preprocessing_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
    )
    r = subprocess.run([python, "-B", "-c", code], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("FreeRec interpreter not usable (see scripts/00_build_beauty_from_raw.sh):\n" + r.stderr)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    if info["freerec_version"] != FREEREC_VERSION:
        raise SystemExit(f"need freerec=={FREEREC_VERSION}, found {info['freerec_version']}")
    if info["preprocessing_source_sha256"] != FREEREC_PREPROC_SHA256:
        log("WARNING: freerec preprocessing source hash differs from the verified 0.9.7 file")
    return info


def run_freerec(python: str, work_dir: Path, archive: Path) -> Path:
    raw = work_dir / "Amazon2014Beauty"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:  # always restore pristine inputs
        for name in sorted(ARCHIVE_MEMBERS):
            with z.open(name) as s, open(raw / name, "wb") as t:
                shutil.copyfileobj(s, t)
    out = work_dir / "Processed" / DATASET_DIR
    if out.exists():
        shutil.rmtree(out)
    cmd = [python, "-B", "-X", "utf8", "-u", "-m", "freerec", FREEREC_ARGS[0], FREEREC_ARGS[1],
           "--root", str(work_dir), *FREEREC_ARGS[2:]]
    log("running: freerec " + " ".join(FREEREC_ARGS[:2] + ["--root", "<work_dir>"] + FREEREC_ARGS[2:]))
    with open(work_dir / "freerec_make.log", "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=work_dir)
    if p.returncode or not all((out / n).exists() for n in FILE_MAP):
        raise SystemExit(f"freerec make failed (rc={p.returncode}); see {work_dir / 'freerec_make.log'}")
    return out


def to_crlf(data: bytes) -> bytes:
    """Convert record terminators LF -> CRLF, leaving newlines inside quoted CSV fields untouched."""
    out = bytearray()
    quoted = False
    for b in data:
        if b == 0x22:  # '"' (escaped "" toggles twice -> no net change)
            quoted = not quoted
        if b == 0x0A and not quoted:
            out += b"\r\n"
        else:
            out.append(b)
    return bytes(out)


def to_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r.fieldnames or []), list(r)


def audit(out_dir: Path) -> tuple[dict, dict]:
    """Manifest entries + split audit (logic adapted from the authors' prepare_data.py)."""
    manifest, parts = {}, {}
    for name in ("train.txt", "valid.txt", "test.txt"):
        cols, rows = read_rows(out_dir / name)
        trip = [(int(r["USER"]), int(r["ITEM"]), float(r["TIMESTAMP"])) for r in rows]
        parts[name] = trip
        ts = [t for _, _, t in trip]
        manifest[name] = {"path": name, "sha256": digest(out_dir / name), "rows": len(rows), "cols": cols,
                          "users": len({u for u, _, _ in trip}), "items": len({i for _, i, _ in trip}),
                          "ts_min": min(ts), "ts_max": max(ts)}
    cols, rows = read_rows(out_dir / "item_meta.txt")
    manifest["item_meta.txt"] = {"path": "item_meta.txt", "sha256": digest(out_dir / "item_meta.txt"),
                                 "rows": len(rows), "cols": cols, "items": len({int(r["ITEM"]) for r in rows})}
    tr, va, te = parts["train.txt"], parts["valid.txt"], parts["test.txt"]
    allp = tr + va + te
    n_users = max(u for u, _, _ in allp) + 1
    n_items = max(i for _, i, _ in allp) + 1
    if len({u for u, _, _ in va}) != n_users or len(va) != n_users or len(te) != n_users:
        raise SystemExit("valid/test must contain exactly one interaction per user")
    keys = [(u, i) for u, i, _ in allp]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate user-item pairs / overlapping splits")
    last_train: dict[int, float] = {}
    for u, _, t in tr:
        last_train[u] = max(last_train.get(u, t), t)
    vt = {u: t for u, _, t in va}
    tt = {u: t for u, _, t in te}
    if any(last_train[u] > vt[u] or vt[u] > tt[u] for u in vt):
        raise SystemExit("split violates per-user time order")
    stats = {"n_users": n_users, "n_items": n_items, "n_interactions": len(allp), "n_train": len(tr),
             "n_valid": len(va), "n_test": len(te), "split_overlap": 0,
             "time_order_check": "passed (train <= valid <= test per user)"}
    counts: dict[int, int] = {}
    for u, _, _ in tr:
        counts[u] = counts.get(u, 0) + 1
    stats["min_train_per_user"] = min(counts.values())
    for k, v in EXPECTED_SIZES.items():
        if stats[k] != v:
            raise SystemExit(f"{k}={stats[k]} differs from RecBoard metadata ({v})")
    return manifest, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True, help="where train/valid/test/item_meta.txt + manifest.json go")
    ap.add_argument("--work-dir", required=True, help="scratch dir for the archive and FreeRec output")
    ap.add_argument("--freerec-python", default=sys.executable, help="interpreter with freerec==0.9.7 installed")
    ap.add_argument("--archive", default=None, help="use a local Amazon2014Beauty.zip instead of downloading")
    ap.add_argument("--url", default=SOURCE_URL)
    ap.add_argument("--line-endings", choices=["crlf", "lf"], default="crlf")
    ap.add_argument("--force", action="store_true", help="overwrite existing files in --out-dir that differ")
    ap.add_argument("--no-verify", action="store_true", help="do not fail on reference sha256 mismatch")
    args = ap.parse_args(argv)

    out_dir, work_dir = Path(args.out_dir), Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_freerec(args.freerec_python)
    log(f"freerec {info['freerec_version']}, pandas {info['pandas_version']}, numpy {info['numpy_version']}")
    archive = obtain_archive(work_dir, Path(args.archive) if args.archive else None, args.url)
    produced = run_freerec(args.freerec_python, work_dir, archive)

    stage = work_dir / "staged"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    conv = to_crlf if args.line_endings == "crlf" else to_lf
    for src, dst in FILE_MAP.items():
        (stage / dst).write_bytes(conv(to_lf((produced / src).read_bytes())))
    manifest, stats = audit(stage)
    ref = REFERENCE_SHA256[args.line_endings]
    match = {n: manifest[n]["sha256"] == ref[n] for n in ref}
    for n, ok in match.items():
        log(f"{n:14s} sha256={manifest[n]['sha256']}  reference_match={ok}")
    if not all(match.values()) and not args.no_verify:
        raise SystemExit("sha256 differs from the PCDRec reference files (use --no-verify to keep anyway)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in list(FILE_MAP.values()):
        dst = out_dir / name
        if dst.exists() and digest(dst) != manifest[name]["sha256"] and not args.force:
            raise SystemExit(f"{dst} exists with different content; pass --force to overwrite")
    for name in list(FILE_MAP.values()):
        dst = out_dir / name
        if not (dst.exists() and digest(dst) == manifest[name]["sha256"]):
            shutil.copyfile(stage / name, dst)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build = {
        "dataset": DATASET_DIR,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": args.url, "registry_url": REGISTRY_URL, "recboard_metadata_url": RECBOARD_META_URL,
        "archive_md5": ARCHIVE_MD5, "archive_sha256": ARCHIVE_SHA256,
        "freerec_command": "freerec " + " ".join(FREEREC_ARGS[:2] + ["--root", "<work_dir>"] + FREEREC_ARGS[2:]),
        **info, "python": sys.version.split()[0], "line_endings": args.line_endings,
        "file_map": FILE_MAP, "sha256_reference_match": match, "audit": stats,
    }
    (out_dir / "build_info.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
    log(f"wrote {out_dir}/{{train,valid,test,item_meta}}.txt, manifest.json, build_info.json")
    log("all reference sha256 match" if all(match.values()) else "WARNING: sha256 mismatch kept (--no-verify)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
