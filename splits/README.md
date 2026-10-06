# Data splits

`cxr14_split_assignment.csv.gz` records the exact ChestX-ray14 split used in the paper, one row per image:

| column | meaning |
|---|---|
| `image` | image file name (`Image Index` in `Data_Entry_2017.csv`) |
| `split` | `train`, `val` or `test` |
| `order` | position in the fixed random order of the training set (-1 for `val` and `test`) |

- **test**: the official patient-level test list of the NIH release (25,596 images).
- **val**: 10% of the remaining patients, drawn with seed 0 (8,378 images).
- **train**: all other images (78,146). The labeled subset with ratio *r* consists of the training images with
  `order < round(r * 78146)`, so the subsets are nested: 2% (1,563) ⊂ 5% ⊂ 10% ⊂ 15% ⊂ 20% (15,629). All other
  training images are used as unlabeled data.

`python scripts/prepare_cxr14.py` turns this file and `Data_Entry_2017.csv` into the CSV files read by
`train.py` (`train.csv`, `val.csv`, `test.csv`, `labeled_XX.csv`, `unlabeled_XX.csv`). No patient appears in more
than one of train, val and test.

The controlled-imbalance subsets of Figure 4 are derived from the 10% subset inside `train.py` (`--imb k`) with
the fixed seed `--imb_seed 1234`.
