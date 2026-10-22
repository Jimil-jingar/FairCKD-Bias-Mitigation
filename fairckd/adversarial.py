"""Adversarial Debiasing (Zhang, Lemoine & Mitchell, 2018) - in-processing bias mitigation.

A predictor (logistic regression) learns to detect CKD while an adversary tries to recover the patient's
demographic group from the predictor's output. The predictor is updated so that it

  * lowers its own loss  (detect CKD well),
  * does NOT move in a direction that helps the adversary (projection term), and
  * actively raises the adversary's loss  (alpha * adversary gradient).

With the Equalized Odds constraint the adversary also sees the true label, so the predictor is pushed to make
its errors - not its predictions - independent of the group.

Implemented in plain NumPy (no TensorFlow / PyTorch), so it trains in seconds and runs in the hosted web app.
Works for two or more groups (the adversary is a softmax classifier over the groups).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import SEED


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class _Adam:
    def __init__(self, shapes, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + eps))
        return out


class AdversarialDebiasing:
    """Logistic-regression predictor trained against a group adversary.

    Parameters
    ----------
    alpha : strength of the adversarial term (0 = plain logistic regression).
    constraint : "equalized_odds" (adversary sees the true label) or "demographic_parity".
    epochs, batch_size, lr : optimisation settings (Adam).
    l2 : L2 penalty on the predictor weights.
    """

    def __init__(self, alpha=1.0, constraint="equalized_odds", epochs=60, batch_size=128, lr=0.01,
                 l2=1e-3, seed=SEED):
        self.alpha, self.constraint = alpha, constraint
        self.epochs, self.batch_size, self.lr, self.l2, self.seed = epochs, batch_size, lr, l2, seed

    # ------------------------------------------------------------------ adversary forward / backward
    def _adv_inputs(self, s, y):
        k = 1.0 + abs(self.c_)
        a = _sigmoid(k * s)
        if self.constraint == "equalized_odds":
            F = np.column_stack([a, a * y, a * (1 - y)])
            dF_da = np.column_stack([np.ones_like(a), y, 1 - y])
        else:
            F, dF_da = a[:, None], np.ones((len(a), 1))
        return k, a, F, dF_da

    def _adv_loss_grads(self, s, y, G):
        """Cross-entropy of the adversary and its gradients w.r.t. adversary params and the predictor logit s."""
        n = len(s)
        k, a, F, dF_da = self._adv_inputs(s, y)
        H = F @ self.U_ + self.v_
        H -= H.max(axis=1, keepdims=True)
        P = np.exp(H); P /= P.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(P[G.astype(bool)] + 1e-12))
        dH = (P - G) / n
        dU, dv = F.T @ dH, dH.sum(axis=0)
        dF = dH @ self.U_.T
        da = (dF * dF_da).sum(axis=1)
        ds = da * k * a * (1 - a)
        dc = np.sum(da * s * a * (1 - a)) * np.sign(self.c_ if self.c_ != 0 else 1.0)
        return loss, dU, dv, dc, ds

    # ------------------------------------------------------------------ training
    def fit(self, Z, y, sensitive_features, sample_weight=None):
        Z = np.asarray(Z, float); y = np.asarray(y, float); A = np.asarray(sensitive_features).astype(str)
        rng = np.random.default_rng(self.seed)
        self.groups_ = np.unique(A)
        G = (A[:, None] == self.groups_[None, :]).astype(float)
        n, d = Z.shape
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)

        # warm start: ordinary logistic regression (standard practice; the adversary then removes group signal)
        lr0 = LogisticRegression(max_iter=2000, random_state=self.seed).fit(Z, y, sample_weight=sw)
        self.w_, self.b_ = lr0.coef_.ravel().astype(float), float(lr0.intercept_[0])
        m = 3 if self.constraint == "equalized_odds" else 1
        self.U_ = rng.normal(0, 0.1, (m, len(self.groups_)))
        self.v_ = np.log(G.mean(axis=0) + 1e-9)
        self.c_ = 1.0

        opt_p = _Adam([(d,), ()], self.lr)
        opt_a = _Adam([self.U_.shape, self.v_.shape, ()], self.lr * 5)
        self.history_ = []
        for ep in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Zb, yb, Gb, wb = Z[idx], y[idx], G[idx], sw[idx]
                s = Zb @ self.w_ + self.b_
                # adversary step (it gets to learn first)
                _, dU, dv, dc, _ = self._adv_loss_grads(s, yb, Gb)
                self.U_, self.v_, c = opt_a.step([self.U_, self.v_, np.array(self.c_)], [dU, dv, np.array(dc)])
                self.c_ = float(c)
                # predictor step
                p = _sigmoid(s)
                ds_pred = wb * (p - yb) / wb.sum()
                gp = np.append(Zb.T @ ds_pred + self.l2 * self.w_, ds_pred.sum())
                _, _, _, _, ds_adv = self._adv_loss_grads(s, yb, Gb)
                ga = np.append(Zb.T @ ds_adv, ds_adv.sum())
                unit = ga / (np.linalg.norm(ga) + 1e-12)
                alpha = self.alpha * np.sqrt((ep + 1) / self.epochs)       # ramp up the adversary (Zhang et al.)
                g = gp - np.dot(gp, unit) * unit - alpha * ga
                w, b = opt_p.step([self.w_, np.array(self.b_)], [g[:-1], np.array(g[-1])])
                self.w_, self.b_ = w, float(b)
            s_all = Z @ self.w_ + self.b_
            la, *_ = self._adv_loss_grads(s_all, y, G)
            p_all = _sigmoid(s_all)
            lp = -np.average(y * np.log(p_all + 1e-12) + (1 - y) * np.log(1 - p_all + 1e-12), weights=sw)
            self.history_.append({"epoch": ep + 1, "predictor loss": lp, "adversary loss": la})
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, Z):
        return np.asarray(Z, float) @ self.w_ + self.b_

    def predict_proba(self, Z):
        p = _sigmoid(self.decision_function(Z))
        return np.column_stack([1 - p, p])

    def predict(self, Z):
        return (self.predict_proba(Z)[:, 1] >= 0.5).astype(int)


ALPHA_GRID = (0.5, 1.0, 3.0, 10.0)


def _eodds_gap(y, pred, A):
    gaps = []
    for lab in (1, 0):
        r = [pred[(A == a) & (y == lab)].mean() for a in np.unique(A) if ((A == a) & (y == lab)).any()]
        gaps.append(max(r) - min(r) if r else 0.0)
    return max(gaps)


def _bal_acc(y, pred):
    return (pred[y == 1].mean() + 1 - pred[y == 0].mean()) / 2


def fit_tuned(Z, y, A, cut, grid=ALPHA_GRID, max_loss=0.03, seed=SEED, sample_weight=None):
    """Adversarial training is known to be sensitive to alpha. Alpha is therefore chosen on a validation split of
    the TRAINING data only: the smallest Equalized Odds gap among alphas whose balanced accuracy stays within
    `max_loss` of plain logistic regression (alpha = 0). `cut(proba, y_train)` turns scores into decisions.
    Returns the model refitted on all training data, plus the chosen alpha."""
    from sklearn.model_selection import train_test_split
    Z, y, A = np.asarray(Z, float), np.asarray(y), np.asarray(A).astype(str)
    strat = np.char.add(A, y.astype(str))
    _, cnt = np.unique(strat, return_counts=True)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed,
                              stratify=strat if cnt.min() >= 2 else y)
    ref = None
    scores = []
    for alpha in (0.0,) + tuple(grid):
        m = AdversarialDebiasing(alpha=alpha, seed=seed).fit(Z[tr], y[tr], A[tr])
        pred = cut(m.predict_proba(Z[va])[:, 1], y[tr])
        ba, gap = _bal_acc(y[va], pred), _eodds_gap(y[va], pred, A[va])
        if alpha == 0.0:
            ref = ba
            continue
        scores.append((alpha, ba, gap))
    ok = [t for t in scores if t[1] >= ref - max_loss]
    best = min(ok, key=lambda t: (t[2], -t[1])) if ok else max(scores, key=lambda t: t[1])
    model = AdversarialDebiasing(alpha=best[0], seed=seed).fit(Z, y, A, sample_weight=sample_weight)
    model.alpha_scores_ = scores
    return model, best[0]
