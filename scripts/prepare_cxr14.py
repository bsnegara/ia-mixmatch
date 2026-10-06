"""Prepare ChestX-ray14 for IA-MixMatch: 256x256 grayscale images, labels and data splits.

Inputs (from the NIH release, https://nihcc.app.box.com/v/ChestXray-NIHCC):
  - the image archives (images_001.tar.gz ... images_012.tar.gz) or an extracted folder of PNGs
  - Data_Entry_2017.csv and test_list.txt

Outputs:
  - <img_out>/<image>.png             112,120 grayscale images, 256x256 (LANCZOS)
  - <split_dir>/train.csv, val.csv, test.csv
  - <split_dir>/labeled_XX.csv, unlabeled_XX.csv   (XX = 02, 05, 10, 15, 20; nested subsets)
  - <split_dir>/prevalence.csv

By default the splits are rebuilt from splits/cxr14_split_assignment.csv.gz, which records the exact
assignment used in the paper. Without that file (or with --regenerate) they are recomputed with the
same procedure and seed, which gives the same result for the same Data_Entry_2017.csv.

Examples:
  python scripts/prepare_cxr14.py --raw_dir /data/nih --img_out data/cxr14_256
  python scripts/prepare_cxr14.py --meta_dir /data/nih --skip_images        # splits only
"""
import argparse, glob, io, os, re, sys, tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iamm import FINDINGS, HEAD_T, TAIL_T  # noqa: E402

SPLIT_SEED = 0          # the split is frozen; training seeds vary separately
VAL_FRAC = 0.10         # 10% of the train_val patients form the validation set
RATIOS = [0.02, 0.05, 0.10, 0.15, 0.20]
SIZE = 256
PAT = re.compile(r"(\d{8}_\d{3}\.png)$")


# ----------------------------------------------------------------- images
def _resize_one(args):
    src, dst = args
    if os.path.exists(dst):
        return 0
    im = Image.open(src) if isinstance(src, str) else Image.open(io.BytesIO(src))
    im = im.convert("L")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS, reducing_gap=2.0)
    tmp = dst + ".tmp.png"
    im.save(tmp, format="PNG")
    os.replace(tmp, dst)
    return 1


def resize_images(raw_dir, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    pngs = [p for p in glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True) if PAT.search(p)]
    jobs = [(p, os.path.join(out_dir, PAT.search(p).group(1))) for p in pngs]
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            n = sum(ex.map(_resize_one, jobs, chunksize=256))
        print(f"resized {n:,} new images from extracted PNGs")
    for tgz in sorted(glob.glob(os.path.join(raw_dir, "images_*.tar.gz"))):
        n = 0
        with tarfile.open(tgz) as tf:
            for m in tf:
                hit = PAT.search(m.name)
                if m.isfile() and hit:
                    n += _resize_one((tf.extractfile(m).read(), os.path.join(out_dir, hit.group(1))))
        print(f"{os.path.basename(tgz)}: {n:,} new images")
    total = len(glob.glob(os.path.join(out_dir, "*.png")))
    print(f"images in {out_dir}: {total:,} (expected 112,120)")


# ----------------------------------------------------------------- labels and splits
def build_labels(csv_path):
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({"image": df["Image Index"], "patient_id": df["Patient ID"], "view": df["View Position"]})
    tags = df["Finding Labels"].str.split("|")
    for c in FINDINGS:
        out[c] = tags.apply(lambda t: int(c in t))
    return out


def regenerate(df, test_list):
    """Same procedure as used for the paper: official test split, 10% of patients for validation,
    and one fixed random order of the training images so that labeled subsets are nested."""
    test_names = set(open(test_list).read().split())
    test = df[df.image.isin(test_names)].reset_index(drop=True)
    trainval = df[~df.image.isin(test_names)]
    rng = np.random.default_rng(SPLIT_SEED)
    pats = trainval.patient_id.unique()
    rng.shuffle(pats)
    val_p = set(pats[: int(round(VAL_FRAC * len(pats)))])
    val = trainval[trainval.patient_id.isin(val_p)].reset_index(drop=True)
    train = trainval[~trainval.patient_id.isin(val_p)]
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    train["order"] = np.arange(len(train))
    return train, val, test


def from_assignment(df, path):
    a = pd.read_csv(path)
    d = df.merge(a[["image", "split", "order"]], on="image", how="inner", validate="one_to_one")
    assert len(d) == len(a) == len(df), "assignment file does not match Data_Entry_2017.csv"
    train = d[d.split == "train"].sort_values("order").drop(columns="split").reset_index(drop=True)
    train["order"] = train["order"].astype(int)
    val = d[d.split == "val"].drop(columns=["split", "order"])
    test = d[d.split == "test"].drop(columns=["split", "order"])
    # keep the row order of Data_Entry_2017.csv for val and test, as in the original files
    return train, val.reset_index(drop=True), test.reset_index(drop=True)


def write_splits(train, val, test, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    assert not set(train.patient_id) & set(val.patient_id)
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(val.patient_id) & set(test.patient_id)
    train.to_csv(f"{split_dir}/train.csv", index=False)
    val.to_csv(f"{split_dir}/val.csv", index=False)
    test.to_csv(f"{split_dir}/test.csv", index=False)
    rows = []
    for r in RATIOS:
        n, pct = int(round(r * len(train))), int(round(r * 100))
        train.iloc[:n].to_csv(f"{split_dir}/labeled_{pct:02d}.csv", index=False)
        train.iloc[n:].to_csv(f"{split_dir}/unlabeled_{pct:02d}.csv", index=False)
        print(f"{pct:>2}% -> labeled {n:,} | unlabeled {len(train) - n:,}")
    for c in FINDINGS:
        p = train[c].mean()
        rows.append({"finding": c, "group": "head" if p > HEAD_T else ("tail" if p < TAIL_T else "medium"),
                     "prev_train_%": 100 * p, "prev_val_%": 100 * val[c].mean(), "prev_test_%": 100 * test[c].mean()})
    pd.DataFrame(rows).sort_values("prev_train_%", ascending=False).to_csv(f"{split_dir}/prevalence.csv", index=False)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", help="folder with images_*.tar.gz or extracted PNGs")
    ap.add_argument("--meta_dir", help="folder with Data_Entry_2017.csv and test_list.txt (default: --raw_dir)")
    ap.add_argument("--img_out", default="data/cxr14_256")
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--assignment", default="splits/cxr14_split_assignment.csv.gz")
    ap.add_argument("--regenerate", action="store_true", help="recompute the split instead of reading --assignment")
    ap.add_argument("--skip_images", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    meta = a.meta_dir or a.raw_dir
    if not meta:
        ap.error("give --meta_dir (and --raw_dir unless --skip_images)")
    if not a.skip_images:
        if not a.raw_dir:
            ap.error("--raw_dir is required unless --skip_images")
        resize_images(a.raw_dir, a.img_out, a.workers)
    df = build_labels(os.path.join(meta, "Data_Entry_2017.csv"))
    if os.path.exists(a.assignment) and not a.regenerate:
        print(f"using the published split assignment: {a.assignment}")
        train, val, test = from_assignment(df, a.assignment)
    else:
        print("recomputing the split with seed", SPLIT_SEED)
        train, val, test = regenerate(df, os.path.join(meta, "test_list.txt"))
    write_splits(train, val, test, a.split_dir)


if __name__ == "__main__":
    main()
