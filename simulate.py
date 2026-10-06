"""E1: numerical check of Section 2 (CPU only).
(a) Bernoulli sharpening scales the logit; label-wise temperature suppresses guesses below 0.5.
(b) Self-training drift of a rare finding: a scalar model bias is trained on a labeled BCE term plus
    the MixMatch L2 term towards (corrected) sharpened targets; we track the targets of true positives."""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sig = lambda z: 1 / (1 + np.exp(-z))
def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def mp_sharpen(p, T):
    """Mean-preserving Bernoulli sharpening over one batch (1-D array)."""
    z, mu, lo, hi = lg(p) / T, p.mean(), -40.0, 40.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if sig(z + mid).mean() > mu: hi = mid
        else: lo = mid
    return sig(z + (lo + hi) / 2)


def drift(variant, prev, rng, steps=30000, B=32, T=0.5, r_c=0.1, kappa=0.002, m=0.999,
          lam=1.0, lr=0.5, n=400_000, n_lab=2000, sep=2.0, log_every=500):
    y = rng.random(n) < prev
    s = np.where(y, rng.normal(sep, 1, n), rng.normal(0, 1, n))   # fixed discriminative score
    lab = rng.choice(n, n_lab, replace=False)
    a = lg(prev) - 0.5 * sep                                       # initial bias
    delta, m_hat = 0.0, prev
    curve = []
    for t in range(steps):
        il, iu = rng.choice(lab, B), rng.integers(0, n, B)
        ql, qu = sig(s[il] + a), sig(s[iu] + a)
        if variant == "standard":
            tg = sig(lg(qu) / T)
        elif variant == "label-wise temperature":
            tg = sig(lg(qu) / (T * r_c))
        elif variant == "MPS":                           # mean-preserving sharpening
            tg = mp_sharpen(qu, T)
        elif variant == "naive prior correction":      # align raw guesses, then sharpen
            m_hat = m * m_hat + (1 - m) * qu.mean()
            tg = sig((lg(qu) + lg(prev) - lg(m_hat)) / T)
        else:                                            # PCS: calibrate shift on sharpened targets
            tg = sig(lg(sig(lg(qu) + delta)) / T)
            m_hat = m * m_hat + (1 - m) * tg.mean()
            delta = float(np.clip(delta + kappa * (lg(prev) - lg(m_hat)), -8, 8))
        a += lr * ((y[il] - ql).mean() + lam * 4 * ((tg - qu) * qu * (1 - qu)).mean())
        if t % log_every == 0 or t == steps - 1:
            q = sig(s + a)
            if variant == "standard": tt = sig(lg(q) / T)
            elif variant == "label-wise temperature": tt = sig(lg(q) / (T * r_c))
            elif variant == "naive prior correction": tt = sig((lg(q) + lg(prev) - lg(m_hat)) / T)
            elif variant == "MPS":
                idx = rng.permutation(n)[:32000].reshape(-1, B)
                tt, yy = np.concatenate([mp_sharpen(q[b], T) for b in idx]), np.concatenate([y[b] for b in idx])
                curve.append((t, tt[yy].mean(), (tt[yy] >= 0.5).mean(), tt.mean(), q.mean()))
                continue
            else: tt = sig(lg(sig(lg(q) + delta)) / T)
            curve.append((t, tt[y].mean(), (tt[y] >= 0.5).mean(), tt.mean(), q.mean()))
    return np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--prev", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = 0.5
    print("(a) Example of Sec. 2.2 (p=0.4, T=0.5, r_c=0.1):")
    print(f"    standard sharpening    : {sig(lg(0.4) / T):.4f}")
    print(f"    label-wise temperature : {sig(lg(0.4) / (T * 0.1)):.2e}")
    print(f"    odds of p=0.01 after sharpening with T=0.5: {0.01/0.99:.1e} -> {(0.01/0.99)**2:.1e}")

    variants = ["standard", "label-wise temperature", "naive prior correction", "PCS", "MPS"]
    label = {"standard": "standard", "label-wise temperature": "label-wise temperature",
             "naive prior correction": "distribution alignment", "PCS": "integral shift", "MPS": "MPS (ours)"}
    res = {v: drift(v, a.prev, np.random.default_rng(a.seed)) for v in variants}
    print(f"\n(b) Drift of a rare finding (prevalence {a.prev:.2%}), final state:")
    print(f"    {'variant':<24}{'mean target, positives':>24}{'recall of positives':>21}"
          f"{'mean target, all':>18}{'mean prediction':>17}")
    for v in variants:
        _, pm, pr, tm, qm = res[v][-1]
        print(f"    {label[v]:<24}{pm:>24.3f}{pr:>21.3f}{tm:>18.4f}{qm:>17.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    p = np.linspace(0.001, 0.999, 500)
    ax[0].plot(p, p, "k:", label="no sharpening")
    ax[0].plot(p, sig(lg(p) / T), label=f"uniform $T$={T}")
    for rr in (0.3, 0.1):
        ax[0].plot(p, sig(lg(p) / (T * rr)), label=f"label-wise, $r_c$={rr}")
    ax[0].axvline(0.5, color="grey", lw=0.5)
    ax[0].set(xlabel="guessed probability $p$", ylabel="sharpened target",
              title="(a) Sharpening scales the logit")
    ax[0].legend(fontsize=8)
    for v in variants:
        ax[1].plot(res[v][:, 0], res[v][:, 1], label=label[v], lw=2.2 if v == "MPS" else 1.2)
    ax[1].set(xlabel="training step", ylabel="mean target of true positives",
              title=f"(b) Rare finding (prevalence {a.prev:.0%})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    f = os.path.join(a.out, "fig_sharpening_analysis.png")
    fig.savefig(f, dpi=200)
    print(f"\nFigure saved: {f}")


if __name__ == "__main__":
    main()
