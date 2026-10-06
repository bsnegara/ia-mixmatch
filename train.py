"""Training script for all methods.

Examples
  python train.py --method supervised --ratio 5 --seed 0
  python train.py --method mixmatch    --ratio 5 --seed 0
  python train.py --method ia_mixmatch --ratio 5 --seed 0
  python train.py --method mixmatch --pcs --ratio 5          # ablation: PCS only
  python train.py --method mixmatch --lw_temp --ratio 5      # ablation: label-wise temperature
  python train.py --method freematch --ratio 5
  python train.py --method supervised --full                 # sanity check with 100% labels
  python train.py --method ia_mixmatch --ratio 5 --benchmark 40
Runs resume automatically from <out_root>/<run>/ckpt_last.pt.
"""
import argparse, json, os, random, time, collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from iamm import FINDINGS
from iamm.data import CXRDataset, infinite_loader, eval_loader, prevalence, groups_from_train
from iamm.augment import weak_aug, strong_aug, eval_resize
from iamm.model import CXRNet
from iamm.ssl import (PCS, ACW, sharpen_bernoulli, labelwise_temperature, mixup,
                      asymmetric_loss, class_balanced_bce)
from iamm.metrics import summarize, pseudo_label_stats

MIXMATCH_FAMILY = {"mixmatch", "ia_mixmatch"}
THRESH_FAMILY = {"fixmatch", "freematch"}


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--method", required=True,
                   choices=["supervised", "mixmatch", "ia_mixmatch", "fixmatch", "freematch"])
    a.add_argument("--ratio", type=int, default=5, help="labeled percentage: 2, 5, 10, 15, 20")
    a.add_argument("--full", action="store_true", help="supervised on the full training set")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--img_dir", default="data/cxr14_256")
    a.add_argument("--split_dir", default="splits")
    a.add_argument("--out_root", default="runs")
    a.add_argument("--tag", default="")
    # optimisation
    a.add_argument("--iters", type=int, default=8000)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--ubs", type=int, default=32)
    a.add_argument("--lr", type=float, default=1e-4)
    a.add_argument("--wd", type=float, default=1e-2, help="decoupled weight decay (AdamW)")
    a.add_argument("--opt", default="adamw", choices=["adamw", "adam"])
    a.add_argument("--sched", default="cosine", choices=["cosine", "const"])
    a.add_argument("--warmup", type=int, default=300, help="linear LR warm-up iterations")
    a.add_argument("--ema", type=float, default=0.999, help="EMA decay of model weights")
    a.add_argument("--size", type=int, default=224)
    a.add_argument("--loss", default="bce", choices=["bce", "asl", "cb"], help="supervised loss")
    # MixMatch
    a.add_argument("--K", type=int, default=2)
    a.add_argument("--T", type=float, default=0.5)
    a.add_argument("--alpha", type=float, default=0.75)
    a.add_argument("--lambda_u", type=float, default=None,
                   help="default: 1")
    a.add_argument("--rampup", type=float, default=0.25, help="fraction of iterations for linear ramp-up")
    # IA-MixMatch components (ia_mixmatch turns on --pcs and --acw)
    a.add_argument("--pcs", action="store_true")
    a.add_argument("--acw", action="store_true")
    a.add_argument("--acw_hard", action="store_true")
    a.add_argument("--lw_temp", action="store_true", help="ablation: label-wise temperature")
    a.add_argument("--beta", type=float, default=1.0)
    a.add_argument("--pcs_mode", default="mps", choices=["mps", "naive", "integral"])
    a.add_argument("--eta", type=float, default=0.0, help="strength of distribution alignment in PCS")
    a.add_argument("--kappa", type=float, default=0.002, help="PCS gain (integral mode only)")
    a.add_argument("--pcs_warmup", type=int, default=1000,
                   help="iterations before the PCS shift starts integrating (m_hat is tracked meanwhile)")
    a.add_argument("--allow_cpu", action="store_true", help="allow running without a GPU (very slow)")
    a.add_argument("--pcs_naive", action="store_true",
                   help="ablation: align raw guesses only (no mean preservation after sharpening)")
    a.add_argument("--m", type=float, default=0.999, help="EMA momentum of PCS/ACW statistics")
    a.add_argument("--target_mult_tail", type=float, default=1.0)
    a.add_argument("--imb", type=float, default=1.0,
                   help="controlled imbalance: keep each labeled image with a tail finding with prob 1/imb")
    a.add_argument("--imb_seed", type=int, default=1234, help="fixed seed of the controlled-imbalance subset")
    a.add_argument("--fix_thr", type=float, default=0.95)
    # bookkeeping
    a.add_argument("--eval_every", type=int, default=1000)
    a.add_argument("--ckpt_every", type=int, default=500)
    a.add_argument("--log_every", type=int, default=50)
    a.add_argument("--pl_window", type=int, default=500)
    a.add_argument("--max_eval", type=int, default=0, help=">0: limit eval images (smoke tests)")
    a.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    a.add_argument("--benchmark", type=int, default=0, help=">0: time N iterations and exit")
    a.add_argument("--force", action="store_true", help="rerun even if DONE exists")
    args = a.parse_args()
    if args.method == "ia_mixmatch":
        args.pcs = args.acw = True
    if args.acw_hard:
        args.acw = True
    if args.pcs_naive:
        args.pcs, args.pcs_mode, args.eta = True, "naive", 1.0
    if args.lambda_u is None:
        args.lambda_u = 1.0   # pilot: lambda_U = 1 best for the MixMatch family; FixMatch/FreeMatch use 1
    return args


def check_loss(a):
    if a.method in MIXMATCH_FAMILY and a.loss not in ("bce", "cb"):
        raise SystemExit(f"--loss {a.loss} is not supported for {a.method} (use bce or cb)")
    if a.method == "freematch" and a.loss != "bce":
        raise SystemExit("--loss must be bce for freematch")


def run_name(a):
    if a.full:
        return f"supervised_full_{a.loss}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    n = a.method + (f"_{a.loss}" if a.loss != "bce" else "")
    if a.method == "mixmatch":
        n += "".join(s for s, on in [("_pcsnaive" if a.pcs_naive else "_pcs", a.pcs), ("_acwhard" if a.acw_hard else "_acw", a.acw),
                                     ("_lwt", a.lw_temp)] if on)
    if a.method == "ia_mixmatch" and a.acw_hard:
        n += "_acwhard"
    if a.pcs and not a.pcs_naive:
        if a.pcs_mode != "mps":
            n += f"_{a.pcs_mode}"
        if a.pcs_mode == "integral" and a.kappa != 0.002:
            n += f"_k{a.kappa:g}"
        if a.eta != 0.0:
            n += f"_eta{a.eta:g}"
    if a.target_mult_tail != 1.0:
        n += f"_tm{a.target_mult_tail:g}"
    if a.T != 0.5:
        n += f"_T{a.T:g}"
    if a.imb != 1.0:
        n += f"_imb{a.imb:g}"
    n += f"_r{a.ratio:02d}_s{a.seed}"
    return n + (f"_{a.tag}" if a.tag else "")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, dev, size, limit=0):
    was = model.training
    model.eval()
    P, Y, n = [], [], 0
    for x, y, _ in loader:
        x = eval_resize(x.to(dev, non_blocking=True).float().div_(255), size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            P.append(torch.sigmoid(model(x).float()).cpu())
        Y.append(y)
        n += len(y)
        if limit and n >= limit:
            break
    model.train(was)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = get_args()
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not args.allow_cpu:
        raise SystemExit("No GPU detected. Run on a CUDA GPU, or pass --allow_cpu to run on the CPU (very slow).")
    torch.backends.cudnn.benchmark = True
    check_loss(args)
    name = run_name(args)
    out = os.path.join(args.out_root if not args.benchmark else "/content/bench_runs", name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/DONE") and not args.force and not args.benchmark:
        print(f"[skip] {name} already finished (use --force to rerun)."); return
    print(f"== {name} | device {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")

    # ---------------- data ----------------
    sd = args.split_dir
    lab_csv = f"{sd}/train.csv" if args.full else f"{sd}/labeled_{args.ratio:02d}.csv"
    unl_csv = f"{sd}/unlabeled_{args.ratio:02d}.csv"
    ssl = args.method != "supervised"
    groups = groups_from_train(f"{sd}/train.csv")
    if args.imb != 1.0:   # controlled imbalance: drop part of the labeled images with a tail finding
        import pandas as pd
        df = pd.read_csv(lab_csv)
        tail_cols = [FINDINGS[i] for i in groups["tail"]]
        pos = df[tail_cols].sum(axis=1).values > 0
        keep = ~pos | (np.random.default_rng(args.imb_seed).random(len(df)) < 1.0 / args.imb)
        lab_csv = os.path.join(out, "labeled_imb.csv")
        df[keep].to_csv(lab_csv, index=False)
        print(f"[imb x{args.imb:g}] labeled: {len(df)} -> {int(keep.sum())} images | tail positives per finding: "
              + ", ".join(f"{c}: {int(df[c].sum())}->{int(df.loc[keep, c].sum())}" for c in tail_cols))
    prior_l = torch.tensor(prevalence(lab_csv), dtype=torch.float32, device=dev)
    lab_ds = CXRDataset(lab_csv, args.img_dir)
    n_pos = torch.tensor(lab_ds.labels.sum(0), dtype=torch.float32, device=dev)
    lab_it = infinite_loader(lab_ds, args.bs, args.workers, args.seed)
    unl_it = infinite_loader(CXRDataset(unl_csv, args.img_dir), args.ubs, args.workers,
                             args.seed + 1) if ssl else None
    val_dl = eval_loader(CXRDataset(f"{sd}/val.csv", args.img_dir), 128, args.workers)
    C = len(FINDINGS)

    # ---------------- model ----------------
    model = CXRNet(C).to(dev).to(memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    ema.eval()
    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    def lr_factor(i):
        if i < args.warmup:
            return (i + 1) / args.warmup
        if args.sched == "const":
            return 1.0
        prog = (i - args.warmup) / max(1, args.iters - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    pcs = acw = None
    T = args.T
    if args.pcs:
        target = prior_l.clone()
        if args.target_mult_tail != 1.0:
            target[groups["tail"]] *= args.target_mult_tail
        pcs = PCS(prior_l, mode=args.pcs_mode, eta=args.eta, kappa=args.kappa, m=args.m, target=target)
    if args.acw or args.method == "freematch":
        acw = ACW(C, dev, m=args.m, hard=args.acw_hard or args.method == "freematch")
    if args.lw_temp:
        T = labelwise_temperature(prior_l, args.T, args.beta)

    # ---------------- resume ----------------
    start, best = 0, -1.0
    ck = f"{out}/ckpt_last.pt"
    if os.path.exists(ck) and not args.benchmark:
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        if s.get("sched"): sched.load_state_dict(s["sched"])
        if pcs and s.get("pcs"): pcs.load_state_dict(s["pcs"])
        if acw and s.get("acw"): acw.load_state_dict(s["acw"])
        start, best = s["it"], s["best"]
        torch.set_rng_state(s["rng"]["torch"].cpu()); np.random.set_state(s["rng"]["numpy"])
        random.setstate(s["rng"]["python"])
        print(f"[resume] from iteration {start}, best val AUROC {best:.4f}")
    if not args.benchmark:
        json.dump(vars(args), open(f"{out}/config.json", "w"), indent=1)

    def checkpoint(it):
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "sched": sched.state_dict(), "pcs": pcs.state_dict() if pcs else None,
                     "acw": acw.state_dict() if acw else None, "it": it, "best": best,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                             "python": random.getstate()}}, ck)

    pl_buf = collections.deque(maxlen=args.pl_window)
    hist = collections.defaultdict(float)
    n_hist, t_last = 0, time.time()
    total = args.benchmark if args.benchmark else args.iters
    t_bench = None
    model.train()

    for it in range(start, total):
        if args.benchmark and it == 10:
            torch.cuda.synchronize(); t_bench = time.time()
        xl, yl, _ = next(lab_it)
        xl = xl.to(dev, non_blocking=True).float().div_(255)
        yl = yl.to(dev, non_blocking=True)
        ramp = min(1.0, it / max(1, args.rampup * args.iters))
        lu = torch.zeros((), device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            if not ssl:
                logits = model(weak_aug(xl, args.size)).float()
                if args.loss == "bce":
                    lx = F.binary_cross_entropy_with_logits(logits, yl)
                elif args.loss == "asl":
                    lx = asymmetric_loss(logits, yl)
                else:
                    lx = class_balanced_bce(logits, yl, n_pos)
                loss = lx
            elif args.method in MIXMATCH_FAMILY:
                xu, yu, _ = next(unl_it)
                xu = xu.to(dev, non_blocking=True).float().div_(255)
                B = xl.shape[0]
                with torch.no_grad():
                    u_augs = [weak_aug(xu, args.size) for _ in range(args.K)]
                    qbar = torch.sigmoid(model(torch.cat(u_augs)).float()).view(args.K, -1, C).mean(0)
                    if pcs:
                        qt, q = pcs.targets(qbar, T)
                        pcs.update(qbar, q, integrate=it >= args.pcs_warmup)
                    else:
                        qt = qbar
                        q = sharpen_bernoulli(qt, T)
                    if acw:
                        acw.update(qt)
                        w = acw.weights(qt)
                    else:
                        w = torch.ones_like(q)
                X = torch.cat([weak_aug(xl, args.size)] + u_augs)
                Y = torch.cat([yl] + [q] * args.K)
                W = torch.cat([torch.ones_like(yl)] + [w] * args.K)
                Xm, Ym, Wm, _ = mixup(X, Y, W, args.alpha)
                logits = model(Xm).float()
                if args.loss == "cb":   # class-balanced labeled loss (on mixed labeled targets)
                    lx = class_balanced_bce(logits[:B], Ym[:B], n_pos)
                else:
                    lx = F.binary_cross_entropy_with_logits(logits[:B], Ym[:B])
                lu = (Wm[B:] * (torch.sigmoid(logits[B:]) - Ym[B:]) ** 2).mean()
                loss = lx + args.lambda_u * ramp * lu
                pl_buf.append((qt.cpu(), q.cpu(), w.cpu(), yu))
            else:  # fixmatch / freematch
                xu, yu, _ = next(unl_it)
                xu = xu.to(dev, non_blocking=True).float().div_(255)
                B, Bu = xl.shape[0], xu.shape[0]
                logits = model(torch.cat([weak_aug(xl, args.size), weak_aug(xu, args.size),
                                          strong_aug(xu, args.size)])).float()
                ll, lw_, ls = logits[:B], logits[B:B + Bu], logits[B + Bu:]
                lx = F.binary_cross_entropy_with_logits(ll, yl)
                with torch.no_grad():
                    pw = torch.sigmoid(lw_.detach())
                    yhat = (pw >= 0.5).float()
                    if args.method == "freematch":
                        acw.update(pw)
                        mask = acw.weights(pw)
                    else:
                        mask = (torch.maximum(pw, 1 - pw) >= args.fix_thr).float()
                lu = (F.binary_cross_entropy_with_logits(ls, yhat, reduction="none") * mask).mean()
                loss = lx + args.lambda_u * lu
                pl_buf.append((pw.cpu(), yhat.cpu(), mask.cpu(), yu))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update_parameters(model)

        hist["loss"] += loss.item(); hist["lx"] += lx.item(); hist["lu"] += float(lu)
        n_hist += 1
        step = it + 1
        if args.benchmark:
            continue
        if step % args.log_every == 0:
            dt = (time.time() - t_last) / n_hist
            row = {k: v / n_hist for k, v in hist.items()}
            row.update(it=step, s_per_it=dt, ramp=ramp, lr=opt.param_groups[0]["lr"])
            if pl_buf:
                row["mean_w"] = float(torch.stack([b[2].mean() for b in pl_buf]).mean())
            if pcs:
                row["pcs_shift_tail"] = float(pcs.shift()[groups["tail"]].mean())
                row["pcs_delta"] = [round(float(v), 3) for v in pcs.shift()]
                row["pcs_mp_tail"] = float(pcs.last_mp[groups["tail"]].mean())
                row["pcs_mraw"] = [float(f"{v:.2e}") for v in pcs.m_raw]
            if acw:
                thr = acw.thresholds()
                row["thr_pos_tail"] = float(thr[1, groups["tail"]].mean())
                row["thr_neg_mean"] = float(thr[0].mean())
            with open(f"{out}/train_log.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"it {step:>6} | loss {row['loss']:.4f} lx {row['lx']:.4f} lu {row['lu']:.4f} "
                  f"| {dt:.3f} s/it", flush=True)
            hist.clear(); n_hist, t_last = 0, time.time()
        if step % args.eval_every == 0 or step == args.iters:
            yv, pv = predict(ema, val_dl, dev, args.size, args.max_eval)
            res = summarize(yv, pv, groups)
            if pl_buf:
                cat = [np.concatenate([b[i].numpy() for b in pl_buf]) for i in range(4)]
                res["pseudo_labels"] = pseudo_label_stats(*cat, groups)
            res["it"] = step
            if res["mean_auroc"] > best:
                best = res["mean_auroc"]
                atomic_save(ema.module.state_dict(), f"{out}/best.pt")
            res["best"] = best
            with open(f"{out}/val_log.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"[val] it {step} | AUROC {res['mean_auroc']:.4f} (best {best:.4f}) "
                  f"| tail AUROC {res.get('tail_auroc', float('nan')):.4f}", flush=True)
            checkpoint(step)
        elif step % args.ckpt_every == 0:
            checkpoint(step)

    if args.benchmark:
        torch.cuda.synchronize()
        n = args.benchmark - 10
        spi = (time.time() - t_bench) / n
        mem = torch.cuda.max_memory_allocated() / 1e9
        res = {"method": args.method, "gpu": torch.cuda.get_device_name(0), "s_per_it": spi,
               "peak_mem_gb": mem, "hours_per_run": spi * args.iters / 3600, "iters": args.iters}
        os.makedirs("/content/bench", exist_ok=True)
        json.dump(res, open(f"/content/bench/{name}.json", "w"))
        print(json.dumps(res))
        return

    # ---------------- final test with best EMA model ----------------
    ema.module.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    test_dl = eval_loader(CXRDataset(f"{sd}/test.csv", args.img_dir), 128, args.workers)
    yt, pt = predict(ema, test_dl, dev, args.size, args.max_eval)
    yv, pv = predict(ema, val_dl, dev, args.size, args.max_eval)
    np.save(f"{out}/test_pred.npy", pt.astype(np.float16))
    np.save(f"{out}/val_pred.npy", pv.astype(np.float16))
    if not os.path.exists(f"{args.split_dir}/test_labels.npy") and not args.max_eval:
        np.save(f"{args.split_dir}/test_labels.npy", yt.astype(np.uint8))
    res = summarize(yt, pt, groups, FINDINGS)
    res["best_val_auroc"] = best
    json.dump(res, open(f"{out}/test_metrics.json", "w"), indent=1)
    print(f"[test] AUROC {res['mean_auroc']:.4f} | AUPRC {res['mean_auprc']:.4f} | "
          + " | ".join(f"{g} {res.get(g + '_auroc', float('nan')):.4f}" for g in groups))
    open(f"{out}/DONE", "w").write(time.ctime())
    # the resume checkpoint (~130 MB) is no longer needed once a run is finished; best.pt is kept
    for f in (ck, ck + ".tmp"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
