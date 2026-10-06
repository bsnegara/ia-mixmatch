"""Prepare the CheXpert validation set for external evaluation.

CheXpert is distributed by the Stanford AIMI Center under its own research use agreement, so no CheXpert
images or labels are included in this repository. Download "CheXpert-v1.0 batch 1 (validate & csv)"
via https://stanfordmlgroup.github.io/competitions/chexpert/ and point
--src to the extracted folder or to the zip file itself.

Outputs:
  <out_dir>/images/valid_<patient>_<study>_<view>.png   frontal images, 256x256 grayscale (LANCZOS)
  <out_dir>/valid.csv   image name + the 14 ChestX-ray14 label columns; only the five shared findings
                        (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion -> Effusion)
                        carry labels, the other columns are 0 and are not used in evaluation.

Example:
  python scripts/prepare_chexpert.py --src "CheXpert-v1.0 batch 1 (validate & csv).zip" --out_dir data/chexpert
"""
import argparse, io, os, re, sys, zipfile

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iamm import FINDINGS  # noqa: E402

SIZE = 256
SHARED = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
          "Edema": "Edema", "Effusion": "Pleural Effusion"}


def key(path):
    """'.../valid/patientX/study1/view1_frontal.jpg' -> 'valid_patientX_study1_view1_frontal.png'"""
    m = re.search(r"(valid|test)/.+", path)
    return (m.group(0) if m else path).replace("/", "_").rsplit(".", 1)[0] + ".png"


class Source:
    """Reads files either from a zip archive or from an extracted folder."""
    def __init__(self, src):
        self.zip = zipfile.ZipFile(src) if zipfile.is_zipfile(src) else None
        self.root = None if self.zip else src
        self.names = self.zip.namelist() if self.zip else [
            os.path.relpath(os.path.join(d, f), src).replace(os.sep, "/")
            for d, _, fs in os.walk(src) for f in fs]

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        with open(os.path.join(self.root, name), "rb") as f:
            return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="CheXpert batch 1 zip or extracted folder")
    ap.add_argument("--out_dir", default="data/chexpert")
    a = ap.parse_args()
    src = Source(a.src)
    os.makedirs(f"{a.out_dir}/images", exist_ok=True)

    csv = [n for n in src.names if os.path.basename(n).lower() == "valid.csv"]
    assert csv, "valid.csv not found in --src"
    raw = pd.read_csv(io.BytesIO(src.read(csv[0])))
    pcol = [c for c in raw.columns if c.lower() == "path"][0]
    raw = raw[raw[pcol].str.contains("frontal", case=False)].copy()

    imgs = {key(n): n for n in src.names if "/valid/" in n and "frontal" in n.lower()
            and n.lower().endswith((".jpg", ".png"))}
    for k, n in imgs.items():
        dst = f"{a.out_dir}/images/{k}"
        if not os.path.exists(dst):
            im = Image.open(io.BytesIO(src.read(n))).convert("L")
            im.resize((SIZE, SIZE), Image.Resampling.LANCZOS, reducing_gap=2.0).save(dst)

    raw["image"] = raw[pcol].apply(key)
    out = pd.DataFrame({"image": raw["image"]})
    for f in FINDINGS:
        out[f] = (raw[SHARED[f]].fillna(0) == 1).astype(int) if f in SHARED else 0
    out = out[out.image.apply(lambda n: os.path.exists(f"{a.out_dir}/images/{n}"))]
    out.to_csv(f"{a.out_dir}/valid.csv", index=False)
    print(f"{len(out)} frontal images; positives: " + ", ".join(f"{f} {int(out[f].sum())}" for f in SHARED))


if __name__ == "__main__":
    main()
