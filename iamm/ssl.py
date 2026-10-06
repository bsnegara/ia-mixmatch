"""Core components of IA-MixMatch (multi-label, sigmoid outputs)."""
import torch
import torch.nn.functional as F

EPS = 1e-6


def logit(p):
    p = p.clamp(EPS, 1 - EPS)
    return torch.log(p) - torch.log1p(-p)


def sharpen_bernoulli(p, T):
    """Eq. (2): logit S(p,T) = logit(p) / T. T can be a scalar or a per-class tensor (C,)."""
    return torch.sigmoid(logit(p) / T)


def labelwise_temperature(prior_l, T, beta=1.0):
    """Eq. (3) variant (ablation): T_c = T * r_c^beta, r_c = prior_c / max prior."""
    r = (prior_l / prior_l.max()).clamp(min=1e-3)
    return T * r.pow(beta)


def mean_preserving_sharpen(p, T, iters=40):
    """Bernoulli sharpening followed by a per-class logit shift chosen (by bisection over the batch)
    so that the batch mean of the targets equals the batch mean of the inputs.
    Returns (targets, shift). No feedback loop and no gain to tune."""
    z = logit(p) / T
    mu = p.mean(0)
    lo = torch.full_like(mu, -40.0)
    hi = torch.full_like(mu, 40.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        above = torch.sigmoid(z + mid).mean(0) > mu
        hi = torch.where(above, mid, hi)
        lo = torch.where(above, lo, mid)
    d = (lo + hi) / 2
    return torch.sigmoid(z + d), d


class PCS:
    """Prior-corrected sharpening.
    mode="mps"      (default) optional distribution alignment of the raw guesses (strength eta) followed by
                    mean-preserving sharpening: sharpening no longer shrinks the targets of rare findings.
    mode="naive"    distribution alignment followed by standard sharpening (ablation).
    mode="integral" previous design: additive shift adapted by an integral controller (ablation; unstable
                    in real training because the model absorbs the shift)."""
    def __init__(self, prior_l, mode="mps", eta=0.0, kappa=0.002, m=0.999, target=None, max_shift=8.0):
        tgt = prior_l.clone() if target is None else target.clone()
        self.target = tgt.clamp(1e-4, 1 - 1e-4)
        self.m_raw = self.target.clone()          # EMA of raw guesses (distribution alignment)
        self.m_hat = self.target.clone()          # EMA of sharpened targets (integral mode)
        self.delta = torch.zeros_like(self.target)
        self.last_mp = torch.zeros_like(self.target)
        self.mode, self.eta, self.kappa, self.m, self.max_shift = mode, eta, kappa, m, max_shift

    def da_shift(self):
        return self.eta * (logit(self.target) - logit(self.m_raw))

    @torch.no_grad()
    def targets(self, qbar, T):
        """Returns (q_tilde, q): corrected guess before sharpening and final target."""
        qbar = qbar.float()
        if self.mode == "integral":
            qt = torch.sigmoid(logit(qbar) + self.delta)
            return qt, sharpen_bernoulli(qt, T)
        qt = torch.sigmoid(logit(qbar) + self.da_shift()) if self.eta > 0 else qbar
        if self.mode == "naive":
            return qt, sharpen_bernoulli(qt, T)
        q, d = mean_preserving_sharpen(qt, T)
        self.last_mp = d
        return qt, q

    @torch.no_grad()
    def update(self, qbar, q_sharp, integrate=True):
        self.m_raw.mul_(self.m).add_((1 - self.m) * qbar.float().mean(0)).clamp_(1e-6, 1 - 1e-6)
        if self.mode == "integral":
            self.m_hat.mul_(self.m).add_((1 - self.m) * q_sharp.float().mean(0)).clamp_(1e-6, 1 - 1e-6)
            if integrate:
                self.delta += self.kappa * (logit(self.target) - logit(self.m_hat))
                self.delta.clamp_(-self.max_shift, self.max_shift)

    def shift(self):
        """Logit shift applied before sharpening (integral: delta; otherwise: distribution alignment)."""
        return self.delta if self.mode == "integral" else self.da_shift()

    def state_dict(self):
        return {k: getattr(self, k) for k in
                ["target", "m_raw", "m_hat", "delta", "last_mp", "mode", "eta", "kappa", "m"]}

    def load_state_dict(self, d):
        for k in ["target", "m_raw", "m_hat", "delta", "last_mp"]:
            if k in d:
                getattr(self, k).copy_(d[k])
        for k in ["mode", "eta", "kappa", "m"]:
            if k in d:
                setattr(self, k, d[k])


class ACW:
    """Adaptive class-aware weighting, Eqs. (6)-(10), with the global status tracked per polarity.
    Row 0 = negative polarity, row 1 = positive polarity.
    hard=True gives binary masks (ablation, and the FreeMatch baseline)."""
    def __init__(self, n_classes, device, m=0.999, hard=False):
        self.m, self.hard = m, hard
        self.tau = torch.full((2,), 0.5, device=device)          # global status per polarity
        self.mu = torch.full((2, n_classes), 0.5, device=device)  # status per polarity & class
        self.var = torch.ones(2, device=device)                   # confidence variance per polarity

    @staticmethod
    def conf_polarity(q):
        return torch.maximum(q, 1 - q), (q >= 0.5).long()

    @torch.no_grad()
    def update(self, q):
        s, y = self.conf_polarity(q.float())
        m = self.m
        for pol in (0, 1):
            mask = y == pol
            if not mask.any():
                continue
            vals = s[mask]
            self.tau[pol] = m * self.tau[pol] + (1 - m) * vals.mean()
            if vals.numel() > 1:
                self.var[pol] = m * self.var[pol] + (1 - m) * vals.var(unbiased=False)
            cnt = mask.sum(0)
            has = cnt > 0
            mean_c = (s * mask).sum(0) / cnt.clamp(min=1)
            self.mu[pol, has] = m * self.mu[pol, has] + (1 - m) * mean_c[has]

    def thresholds(self):
        return self.mu / self.mu.max(dim=1, keepdim=True).values * self.tau[:, None]  # (2, C)

    @torch.no_grad()
    def weights(self, q):
        s, y = self.conf_polarity(q.float())
        thr = self.thresholds()
        t = torch.where(y == 1, thr[1][None, :], thr[0][None, :])
        if self.hard:
            return (s >= t).float()
        v = torch.where(y == 1, self.var[1], self.var[0]).clamp(min=1e-6)
        return torch.where(s >= t, torch.ones_like(s), torch.exp(-(s - t) ** 2 / (2 * v)))

    def state_dict(self):
        return {"tau": self.tau, "mu": self.mu, "var": self.var, "m": self.m, "hard": self.hard}

    def load_state_dict(self, d):
        self.tau.copy_(d["tau"]); self.mu.copy_(d["mu"]); self.var.copy_(d["var"])
        self.m, self.hard = d["m"], d["hard"]


def mixup(X, Y, W, alpha):
    """MixUp with lambda' = max(lambda, 1 - lambda); inputs, targets and weights share lambda."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)
    idx = torch.randperm(X.shape[0], device=X.device)
    return (lam * X + (1 - lam) * X[idx], lam * Y + (1 - lam) * Y[idx],
            lam * W + (1 - lam) * W[idx], lam)


def asymmetric_loss(logits, y, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    """Asymmetric loss (Ridnik et al., 2021) for the supervised baseline."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    lp = y * torch.log(p.clamp(min=EPS)) * (1 - p) ** gamma_pos
    ln = (1 - y) * torch.log((1 - p_neg).clamp(min=EPS)) * p_neg ** gamma_neg
    return -(lp + ln).mean()


def class_balanced_bce(logits, y, n_pos, beta=0.999):
    """Class-balanced loss (Cui et al., 2019) for multi-label BCE: each finding is weighted by the
    inverse effective number of its labeled positives; weights are normalised to mean 1."""
    eff = (1 - beta ** n_pos.clamp(min=1)) / (1 - beta)
    w = 1.0 / eff
    w = w / w.mean()
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
