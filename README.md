# IA-MixMatch: Imbalance-aware MixMatch for semi-supervised multi-label chest X-ray classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23178791.svg)](https://doi.org/10.5281/zenodo.23178791)

Code and data splits for the paper

> B. S. Negara, M. Irsyad, and Darwan, "Imbalance-aware MixMatch for semi-supervised multi-label chest X-ray
> classification," *IAES International Journal of Artificial Intelligence (IJ-AI)*, under review.

Semi-supervised learning can reduce the annotation burden of chest X-ray classification, but its pseudo-labels
under-predict rare findings. We show that sharpening rare findings with a lower temperature suppresses them
further, and that sharpening with any temperature shrinks their average target. IA-MixMatch adds two
components to MixMatch:

- **Mean-preserving sharpening (MPS)**: after Bernoulli sharpening, a per-finding logit shift found by bisection
  makes the average target of each finding in a batch equal to its average guess. It has no hyperparameter and
  no feedback loop.
- **Adaptive class-aware weighting (ACW)**: tracks the learning status of each finding and polarity with
  exponential moving averages and softly down-weights unreliable soft targets instead of discarding them.

## Main result

Mean AUROC (%) on the official ChestX-ray14 test split (25,596 images), DenseNet-121, mean ± std over three seeds.

| Method | 2% | 5% | 10% | 15% | 20% |
|---|---|---|---|---|---|
| Supervised (BCE) | 65.35 ± 0.89 | 69.53 ± 0.37 | 73.17 ± 0.39 | 75.50 ± 0.15 | 76.65 ± 0.18 |
| MixMatch | 65.94 ± 0.12 | 69.66 ± 0.95 | 74.33 ± 0.19 | 76.31 ± 0.28 | 76.96 ± 0.11 |
| FreeMatch | – | 69.11 ± 0.26 | – | – | 76.63 ± 0.15 |
| **IA-MixMatch** | **66.68 ± 0.62** | **70.71 ± 0.61** | **74.61 ± 0.36** | **76.70 ± 0.32** | **77.74 ± 0.22** |

On the external CheXpert validation set (202 frontal images, five shared findings), IA-MixMatch trained with
20% labels reaches a mean AUROC of 82.6, compared with 81.4 for MixMatch and 81.1 for supervised training.

## Repository layout

```
iamm/                  method and utilities
  ssl.py               Bernoulli sharpening, MPS (class PCS, mode "mps"), ACW, MixUp, losses
  data.py, augment.py  dataset, GPU augmentation (no horizontal flip)
  model.py             DenseNet-121 with grayscale input
  metrics.py           AUROC, AUPRC, ECE per finding and prevalence group, pseudo-label statistics
  analysis.py          DeLong test, paired bootstrap, figures
  tests.py             unit tests
train.py               all methods, baselines and ablations (resumes automatically from checkpoints)
simulate.py            numerical analysis of Section 2 (Figure 1, CPU only)
scripts/
  prepare_cxr14.py     256x256 images, labels and data splits for ChestX-ray14
  prepare_chexpert.py  CheXpert validation images and labels for external evaluation
  run_experiments.sh   every training run reported in the paper
  analyze.py           tables, significance tests and figures from the run outputs
splits/                exact data split used in the paper (see splits/README.md)
```

In the code, MPS is implemented by the class `PCS` with `--pcs_mode mps` (the default); `--method ia_mixmatch`
turns on MPS and ACW.

## Installation

```bash
git clone https://github.com/bsnegara/ia-mixmatch.git
cd ia-mixmatch
pip install -r requirements.txt
python -m iamm.tests          # unit tests, CPU is enough
```

The experiments were run on Google Colab with PyTorch 2 and one NVIDIA L4 GPU (24 GB), using automatic mixed precision.

## Data

**ChestX-ray14.** Download the images, `Data_Entry_2017.csv` and `test_list.txt` from the
[NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC), then run

```bash
python scripts/prepare_cxr14.py --raw_dir /path/to/nih --img_out data/cxr14_256 --split_dir splits
```

This resizes the 112,120 images to 256x256 grayscale PNG and writes the training (78,146), validation (8,378) and
official test (25,596) splits together with the nested labeled subsets of 2%, 5%, 10%, 15% and 20%. The split is
read from `splits/cxr14_split_assignment.csv.gz`, so it is identical to the one used in the paper.

**CheXpert** (external validation only). The CheXpert research use agreement does not allow redistribution, so no
CheXpert data are included. Download "CheXpert-v1.0 batch 1 (validate & csv)" from the
[Stanford AIMI Center](https://stanfordmlgroup.github.io/competitions/chexpert/) and run

```bash
python scripts/prepare_chexpert.py --src "CheXpert-v1.0 batch 1 (validate & csv).zip" --out_dir data/chexpert
```

## Training

```bash
python train.py --method ia_mixmatch --ratio 5 --seed 0 --iters 4000 --eval_every 500 --tag p2
python train.py --method mixmatch    --ratio 5 --seed 0 --iters 4000 --eval_every 500 --tag p2
python train.py --method supervised  --ratio 5 --seed 0 --iters 4000 --eval_every 500 --tag p2
```

Each run writes `config.json`, training and validation logs, `best.pt`, test predictions and `test_metrics.json` to
`runs/<run name>/`. The default recipe is AdamW (learning rate 1e-4, weight decay 0.01), 300 warm-up iterations
with cosine decay, batches of 32 labeled and 32 unlabeled images, K = 2, T = 0.5, MixUp alpha = 0.75 and
lambda_U = 1 with a linear ramp-up over the first 25% of training. The number of iterations is 4,000 for 2% and 5%
labels, 6,000 for 10%, and 8,000 for 15% and 20%.

## Reproducing the paper

```bash
bash scripts/run_experiments.sh main          # Table 1
bash scripts/run_experiments.sh losses        # class-balanced and asymmetric losses
bash scripts/run_experiments.sh ablation      # Table 3
bash scripts/run_experiments.sh sensitivity   # Figure 4
python simulate.py                            # Figure 1

python scripts/analyze.py summary             # mean +- std of all runs
python scripts/analyze.py significance        # bootstrap CIs, DeLong with Holm correction
python scripts/analyze.py chexpert            # Table 2
python scripts/analyze.py figures             # Figures 3 and 4
python scripts/analyze.py gradcam             # Figure 5
```

The full set of 120 runs takes roughly 55 to 60 GPU hours on an NVIDIA L4. IA-MixMatch adds below 1% to the training
time of MixMatch (0.307 vs. 0.305 s per iteration).

## Citation

Code archive: B. S. Negara, M. Irsyad, and Darwan, *IA-MixMatch* (v1.0.1), Zenodo, 2026, doi: [10.5281/zenodo.23178792](https://doi.org/10.5281/zenodo.23178792).
The DOI [10.5281/zenodo.23178791](https://doi.org/10.5281/zenodo.23178791) always resolves to the latest version.

Paper:

```bibtex
@article{negara2026iamixmatch,
  title   = {Imbalance-aware {MixMatch} for semi-supervised multi-label chest {X}-ray classification},
  author  = {Negara, Benny Sukma and Irsyad, Muhammad and Darwan},
  journal = {IAES International Journal of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

## License and data use

The code is released under the MIT License. ChestX-ray14 and CheXpert remain subject to the terms of their
providers; the split files in `splits/` contain only image names and split assignments.

## Acknowledgments

This work was supported by the Institute for Research and Community Service (LPPM) of Universitas Islam Negeri
Sultan Syarif Kasim Riau. Contact: Benny Sukma Negara (bsnegara@uin-suska.ac.id).
