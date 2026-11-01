#!/usr/bin/env python3
"""
Hybrid Soft Computing for Predictive Maintenance and Maintenance Scheduling
(Advanced version)

Soft-computing components
  Neural   : single sigmoid, MLP (backprop/Adam), RBF network (K-means + linear output)
  Fuzzy    : zero-order Sugeno priority, Mamdani comparison, ANFIS-style first-order
             Sugeno predictor (hybrid learning: LSE consequents + L-BFGS premises)
  Evolution: genetic algorithm (permutation), PSO (random-key) tuning of the RBF network
             and PSO / SA / random-search baselines for scheduling
  Ensemble : soft-voting of the learned predictors, isotonic calibration

Run:  python case_study_advanced.py        (about 2-4 minutes, CPU only)
All outputs are written to ./outputs (CSV tables + figures/).
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
FEATURES = ["temperature", "vibration", "pressure", "load", "vibration_mean_5",
            "temperature_mean_5", "vibration_std_5", "temperature_slope_4"]


# ----------------------------------------------------------------------------
# 1. Synthetic data
# ----------------------------------------------------------------------------
def simulate_machine(mid, rng, life=None):
    """Daily history of one machine. Wear grows non-linearly; each machine has its own
    baseline offsets and wear speed, so prediction is not trivial."""
    L = int(life if life is not None else rng.integers(65, 121))
    t = np.arange(1, L + 1)
    wear = (t / L) ** rng.uniform(2.0, 3.0)
    load = np.clip(0.6 + 0.15 * np.sin(t / 7 + rng.uniform(0, 6)) + rng.normal(0, .08, L), .2, 1.1)
    temp = 62 + rng.normal(0, 2.5) + 15 * wear + 8 * load + rng.normal(0, 2.2, L)
    vib = 3.2 + rng.normal(0, .3) + 2.5 * wear + 1.2 * load + rng.normal(0, .35, L)
    pres = 5 + .8 * load - .9 * wear + rng.normal(0, .12, L)
    return pd.DataFrame(dict(machine_id=mid, day=t, temperature=temp, vibration=vib,
                             pressure=pres, load=load, rul_days=L - t))


def add_features(df):
    """Causal rolling features, computed inside each machine only (no leakage)."""
    g = df.groupby("machine_id")
    df = df.copy()
    df["vibration_mean_5"] = g.vibration.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["temperature_mean_5"] = g.temperature.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["vibration_std_5"] = g.vibration.transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0)
    df["temperature_slope_4"] = g.temperature.transform(lambda s: s.diff(4)).fillna(0)
    return df


def make_dataset(n_machines=180, seed=SEED):
    rng = np.random.default_rng(seed)
    df = add_features(pd.concat([simulate_machine(i, rng) for i in range(n_machines)], ignore_index=True))
    df = df[df.rul_days > 0].copy()
    df["failure_within_7_days"] = ((df.rul_days > 0) & (df.rul_days <= 7)).astype(int)
    return df.reset_index(drop=True)


def split_data(df):
    gs = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    dev, te = next(gs.split(df, groups=df.machine_id))
    dev_df = df.iloc[dev]
    gs = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1)
    tr, va = next(gs.split(dev_df, groups=dev_df.machine_id))
    train, val, test = dev_df.iloc[tr], dev_df.iloc[va], df.iloc[te]
    ids = [set(p.machine_id) for p in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "machine leakage"
    return train, val, test


# ----------------------------------------------------------------------------
# 2. Neural / neuro-fuzzy predictors
# ----------------------------------------------------------------------------
class RBFClassifier:
    """Two-stage RBF network: K-means centres, shared Gaussian width, logistic output."""
    def __init__(self, n_centres=24, width_scale=1.0):
        self.n_centres, self.width_scale = int(n_centres), float(width_scale)

    def fit(self, X, y):
        self.centres = KMeans(self.n_centres, n_init=3, random_state=SEED).fit(X).cluster_centers_
        d = np.linalg.norm(self.centres[:, None] - self.centres[None], axis=2)
        self.width = max(float(np.median(d[d > 0])) * self.width_scale, .1)
        self.out = LogisticRegression(max_iter=1000).fit(self.transform(X), y)
        return self

    def transform(self, X):
        d2 = ((X[:, None, :] - self.centres[None]) ** 2).sum(2)
        return np.exp(-d2 / (2 * self.width ** 2))

    def predict_proba(self, X):
        return self.out.predict_proba(self.transform(X))


class ANFIS:
    """First-order Sugeno ANFIS with Gaussian membership functions (grid partition).
    Hybrid learning (variable projection): for given premise parameters the consequent
    parameters are solved exactly by ridge least squares (forward pass of Jang's ANFIS);
    the premise parameters (centres, widths) are then tuned by L-BFGS-B (backward pass)."""
    def __init__(self, idx, n_mf=2, max_iter=30, sub=4000):
        self.idx, self.m, self.max_iter, self.sub = list(idx), n_mf, max_iter, sub
        self.d = len(self.idx)
        self.rules = list(itertools.product(range(n_mf), repeat=self.d))

    def _design(self, X, p):
        d, m = self.d, self.m
        c, s = p[:d * m].reshape(d, m), np.exp(p[d * m:]).reshape(d, m)
        mu = np.exp(-((X[:, :, None] - c[None]) ** 2) / (2 * s[None] ** 2))
        w = np.stack([np.prod([mu[:, k, j] for k, j in enumerate(r)], axis=0) for r in self.rules], 1)
        wbar = w / (w.sum(1, keepdims=True) + 1e-12)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        return (wbar[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)

    @staticmethod
    def _lse(A, y):
        return np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y)

    def fit(self, X, y):
        X = X[:, self.idx]
        q = np.quantile(X, np.linspace(.2, .8, self.m), axis=0).T          # d x m
        s = np.tile(X.std(0, keepdims=True).T / self.m + 1e-3, (1, self.m))
        p0 = np.concatenate([q.ravel(), np.log(s).ravel()])
        sub = np.random.default_rng(SEED).choice(len(X), min(self.sub, len(X)), replace=False)
        Xs, ys = X[sub], y[sub]

        def loss(p):
            A = self._design(Xs, p)
            return float(np.mean((A @ self._lse(A, ys) - ys) ** 2))

        res = minimize(loss, p0, method="L-BFGS-B", options=dict(maxiter=self.max_iter))
        self.p = res.x
        A = self._design(X, self.p)
        self.theta = self._lse(A, y)
        return self

    def predict_proba(self, X):
        p = np.clip(self._design(X[:, self.idx], self.p) @ self.theta, 0, 1)
        return np.column_stack([1 - p, p])


class SoftVote:
    def __init__(self, members): self.members = members
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def pso_tune_rbf(Xtr, ytr, Xva, yva, seed=SEED, swarm=8, iters=8):
    """Particle swarm optimisation of the RBF hyper-parameters (n_centres, width_scale)
    using validation average precision as fitness. The test set is never touched."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([8., .3]), np.array([48., 3.])
    pos = rng.uniform(lo, hi, (swarm, 2)); vel = np.zeros_like(pos)
    fit = lambda x: average_precision_score(yva, RBFClassifier(round(x[0]), x[1]).fit(Xtr, ytr).predict_proba(Xva)[:, 1])
    pbest, pfit = pos.copy(), np.array([fit(x) for x in pos])
    g = pbest[pfit.argmax()].copy(); gfit = pfit.max(); hist = [gfit]
    for _ in range(iters):
        r1, r2 = rng.random((swarm, 2)), rng.random((swarm, 2))
        vel = .72 * vel + 1.49 * r1 * (pbest - pos) + 1.49 * r2 * (g - pos)
        pos = np.clip(pos + vel, lo, hi)
        f = np.array([fit(x) for x in pos])
        better = f > pfit
        pbest[better], pfit[better] = pos[better], f[better]
        if pfit.max() > gfit: gfit, g = pfit.max(), pbest[pfit.argmax()].copy()
        hist.append(gfit)
    return dict(n_centres=int(round(g[0])), width_scale=float(g[1]), val_AP=float(gfit), history=hist)


# ----------------------------------------------------------------------------
# 3. Evaluation utilities
# ----------------------------------------------------------------------------
def best_threshold(y, p):
    grid = np.arange(.05, .951, .01)
    return float(grid[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid]))])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); tot = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any(): tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def metrics(y, p, thr):
    yh = p >= thr
    return dict(precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh),
                F1=f1_score(y, yh), AP=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p),
                Brier=brier_score_loss(y, p), ECE=ece(y, p))


def machine_bootstrap(test, p, thr, reps=300, seed=SEED):
    """Resample whole machines (not rows) because rows of one machine are correlated."""
    rng = np.random.default_rng(seed)
    y = test.failure_within_7_days.to_numpy()
    groups = [np.flatnonzero(test.machine_id.to_numpy() == m) for m in test.machine_id.unique()]
    out = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        out.append((f1_score(y[idx], p[idx] >= thr, zero_division=0),
                    recall_score(y[idx], p[idx] >= thr, zero_division=0),
                    precision_score(y[idx], p[idx] >= thr, zero_division=0)))
    out = np.array(out)
    return {k: (float(np.percentile(out[:, i], 2.5)), float(np.percentile(out[:, i], 97.5)))
            for i, k in enumerate(["F1", "recall", "precision"])}


def train_models(train, val, test):
    sc = StandardScaler().fit(train[FEATURES])
    Xtr, Xva, Xte = (sc.transform(d[FEATURES]) for d in (train, val, test))
    ytr, yva, yte = (d.failure_within_7_days.to_numpy() for d in (train, val, test))
    models = {"Single sigmoid": LogisticRegression(max_iter=1000).fit(Xtr, ytr)}
    models["MLP"] = MLPClassifier((32, 16), activation="relu", solver="adam", alpha=.01,
                                  max_iter=350, random_state=SEED).fit(Xtr, ytr)
    models["RBF"] = RBFClassifier().fit(Xtr, ytr)
    pso = pso_tune_rbf(Xtr, ytr, Xva, yva)
    models["RBF (PSO-tuned)"] = RBFClassifier(pso["n_centres"], pso["width_scale"]).fit(Xtr, ytr)
    top3 = np.argsort(-np.abs(models["Single sigmoid"].coef_[0]))[:3]
    models["ANFIS"] = ANFIS(top3).fit(Xtr, ytr.astype(float))
    models["Soft ensemble"] = SoftVote([models[k] for k in ("Single sigmoid", "MLP", "RBF (PSO-tuned)", "ANFIS")])

    rows, probs, thr = [], {}, {}
    for name, m in models.items():
        pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr[name], probs[name] = best_threshold(yva, pv), pt
        rows.append(dict(model=name, validation_AP=average_precision_score(yva, pv), threshold=thr[name],
                         **metrics(yte, pt, thr[name])))
    fixed = (test.temperature > 76) | (test.vibration > 5.8)
    rows.append(dict(model="Fixed thresholds", precision=precision_score(yte, fixed), recall=recall_score(yte, fixed),
                     F1=f1_score(yte, fixed)))
    table = pd.DataFrame(rows)
    chosen = table.dropna(subset=["validation_AP"]).sort_values("validation_AP", ascending=False).model.iloc[0]
    # isotonic calibration of the selected model on validation data only
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(models[chosen].predict_proba(Xva)[:, 1], yva)
    cal_test = iso.predict(probs[chosen])
    cal = dict(Brier_before=brier_score_loss(yte, probs[chosen]), Brier_after=brier_score_loss(yte, cal_test),
               ECE_before=ece(yte, probs[chosen]), ECE_after=ece(yte, cal_test))
    return sc, models, chosen, table, probs, thr, iso, cal, pso, (yte,)


def mlp_seed_study(train, test, seeds=(1, 2, 3, 4, 5)):
    sc = StandardScaler().fit(train[FEATURES])
    Xtr, Xte = sc.transform(train[FEATURES]), sc.transform(test[FEATURES])
    ytr, yte = train.failure_within_7_days.to_numpy(), test.failure_within_7_days.to_numpy()
    res = []
    for s in seeds:
        p = MLPClassifier((32, 16), alpha=.01, max_iter=350, random_state=s).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        res.append((average_precision_score(yte, p), f1_score(yte, p >= .3)))
    r = np.array(res)
    return dict(AP_mean=r[:, 0].mean(), AP_std=r[:, 0].std(), F1_mean=r[:, 1].mean(), F1_std=r[:, 1].std())


# ----------------------------------------------------------------------------
# 4. Fuzzy priority (Sugeno + Mamdani)
# ----------------------------------------------------------------------------
CONSEQ = np.array([[10, 20, 35], [35, 55, 75], [70, 85, 100]], float)


def memberships(x):
    x = float(np.clip(x, 0, 1))
    return np.array([max(0, 1 - 2 * x), max(0, 1 - abs(2 * x - 1)), max(0, 2 * x - 1)])


def fuzzy_priority(risk, crit, conseq=CONSEQ):
    """Zero-order Sugeno, product AND, weighted average."""
    f = np.outer(memberships(risk), memberships(crit))
    return float((f * conseq).sum() / f.sum())


_OUT_SETS = {"VL": ([0, 10, 30], [1, 1, 0]), "L": ([10, 30, 55], [0, 1, 0]), "M": ([30, 55, 80], [0, 1, 0]),
             "H": ([55, 80, 100], [0, 1, 0]), "VH": ([80, 100], [0, 1])}
_MAMDANI_RULES = [["VL", "L", "L"], ["L", "M", "H"], ["H", "H", "VH"]]


def mamdani_priority(risk, crit):
    """Mamdani: min-AND, min-implication, max-aggregation, centroid defuzzification."""
    u = np.linspace(0, 100, 501)
    agg = np.zeros_like(u)
    mr, mc = memberships(risk), memberships(crit)
    for i in range(3):
        for j in range(3):
            xs, ys = _OUT_SETS[_MAMDANI_RULES[i][j]]
            agg = np.maximum(agg, np.minimum(min(mr[i], mc[j]), np.interp(u, xs, ys)))
    return float((u * agg).sum() / agg.sum())


# ----------------------------------------------------------------------------
# 5. Fleets and cost model
# ----------------------------------------------------------------------------
def make_fleet(scaler, model, iso, n=12, seed=2026, cutoff=80):
    """Separate fleet observed only through `cutoff`; risk = calibrated model score."""
    rng = np.random.default_rng(seed)
    rows = [add_features(simulate_machine(i, rng, life=int(rng.integers(cutoff + 2, 121)))).query("day == @cutoff")
            for i in range(n)]
    obs = pd.concat(rows, ignore_index=True)
    risk = np.clip(iso.predict(model.predict_proba(scaler.transform(obs[FEATURES]))[:, 1]), .005, 1)  # floor: isotonic can output exact 0
    return _finish_fleet(pd.DataFrame(dict(machine=np.arange(n), risk=risk, true_rul=obs.rul_days.to_numpy())), rng)


def synthetic_fleet(n, rng):
    return _finish_fleet(pd.DataFrame(dict(machine=np.arange(n), risk=np.clip(rng.beta(.6, 3, n), 0, 1))), rng)


def _finish_fleet(f, rng):
    n = len(f)
    f["criticality"] = rng.beta(2, 2, n)
    f["priority"] = [fuzzy_priority(r, c) for r, c in zip(f.risk, f.criticality)]
    f["failure_loss"] = 8000 + 22000 * f.criticality
    f["prod_rate"] = 150 + 250 * f.criticality * rng.uniform(.6, 1.4, n)
    f["maint_cost"] = 1200 + 600 * rng.random(n)
    f["x"], f["y"] = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    return f


def build_costs(f, crews=2, days=6, priority=None, weight_fail=1., weight_other=1.):
    """cost[i, s] of servicing machine i in slot s (day = s // crews + 1). Returns total and parts."""
    pri = f.priority.to_numpy() if priority is None else priority
    d = np.repeat(np.arange(1, days + 1), crews)[None, :].astype(float)
    q = 1 - (1 - f.risk.to_numpy()[:, None]) ** (d / 7)       # constant daily hazard assumption
    fail = q * f.failure_loss.to_numpy()[:, None]
    prod = f.prod_rate.to_numpy()[:, None] * d
    urg = 3 * pri[:, None] * (d - 1)
    maint = np.repeat(f.maint_cost.to_numpy()[:, None], d.shape[1], 1)
    return weight_fail * fail + weight_other * (prod + urg + maint), dict(fail=fail, other=prod + urg + maint)


def objective(order, costs):
    return float(costs[np.asarray(order), np.arange(len(order))].sum())


def make_travel_objective(costs, f, crews, tc=10.):
    """Non-linear (sequence-dependent) objective: crews travel between consecutive jobs.
    Not an assignment problem any more, so the exact linear solver no longer applies."""
    xy = f[["x", "y"]].to_numpy()
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    def obj(order):
        order = np.asarray(order)
        return float(costs[order, np.arange(len(order))].sum() + tc * D[order[:-crews], order[crews:]].sum())
    return obj


# ----------------------------------------------------------------------------
# 6. Metaheuristic schedulers (all use the same evaluation budget: 160 x 60)
# ----------------------------------------------------------------------------
def ordered_crossover(a, b, rng):
    n = len(a)
    i, j = sorted(rng.choice(n + 1, 2, replace=False))
    child = -np.ones(n, int); child[i:j] = a[i:j]
    used = set(a[i:j].tolist())
    child[child < 0] = [g for g in b if g not in used]
    return child


def genetic(obj, n, seed, init=None, gens=160, pop=60, pm=.25):
    rng = np.random.default_rng(seed)
    P = ([np.asarray(init).copy()] if init is not None else []) + [rng.permutation(n) for _ in range(pop - (init is not None))]
    trace = []
    for _ in range(gens):
        sc = np.array([obj(o) for o in P]); rk = np.argsort(sc); trace.append(float(sc[rk[0]]))
        nxt = [P[i].copy() for i in rk[:2]]
        pick = lambda: P[(c := rng.choice(pop, 3, replace=False))[np.argmin(sc[c])]]
        while len(nxt) < pop:
            ch = ordered_crossover(pick(), pick(), rng)
            if rng.random() < pm:
                i, j = rng.choice(n, 2, replace=False); ch[i], ch[j] = ch[j], ch[i]
            nxt.append(ch)
        P = nxt
    best = min(P, key=obj); trace.append(obj(best))
    return best, trace


def pso_schedule(obj, n, seed, init=None, iters=160, swarm=60):
    """Random-key PSO: a particle is a real vector, the schedule is its argsort."""
    rng = np.random.default_rng(seed)
    pos = rng.random((swarm, n)); vel = np.zeros((swarm, n))
    if init is not None: pos[0, np.asarray(init)] = np.arange(n) / n
    dec = lambda x: np.argsort(x)
    pf = np.array([obj(dec(x)) for x in pos]); pb = pos.copy()
    g = pb[pf.argmin()].copy(); gf = pf.min(); trace = [gf]
    for _ in range(iters - 1):
        r1, r2 = rng.random((swarm, n)), rng.random((swarm, n))
        vel = .72 * vel + 1.49 * r1 * (pb - pos) + 1.49 * r2 * (g - pos)
        pos = pos + vel
        f = np.array([obj(dec(x)) for x in pos]); b = f < pf
        pb[b], pf[b] = pos[b], f[b]
        if pf.min() < gf: gf, g = pf.min(), pb[pf.argmin()].copy()
        trace.append(gf)
    return dec(g), trace


def simulated_annealing(obj, n, seed, init=None, iters=9600, T0=300., T1=1.):
    rng = np.random.default_rng(seed)
    cur = np.asarray(init).copy() if init is not None else rng.permutation(n)
    fc = obj(cur); best, fb = cur.copy(), fc; trace = []
    for k in range(iters):
        T = T0 * (T1 / T0) ** (k / iters)
        i, j = rng.choice(n, 2, replace=False)
        cand = cur.copy(); cand[i], cand[j] = cand[j], cand[i]
        fn = obj(cand)
        if fn < fc or rng.random() < np.exp((fc - fn) / T): cur, fc = cand, fn
        if fc < fb: best, fb = cur.copy(), fc
        if (k + 1) % 60 == 0: trace.append(fb)
    return best, trace


def random_search(obj, n, seed, init=None, evals=9600):
    rng = np.random.default_rng(seed)
    best = np.asarray(init).copy() if init is not None else rng.permutation(n)
    fb = obj(best); trace = []
    for k in range(evals):
        c = rng.permutation(n); fc = obj(c)
        if fc < fb: best, fb = c, fc
        if (k + 1) % 60 == 0: trace.append(fb)
    return best, trace


METHODS = {"GA": genetic, "PSO": pso_schedule, "SA": simulated_annealing, "Random search": random_search}


# ----------------------------------------------------------------------------
# 7. Experiments
# ----------------------------------------------------------------------------
def exact_assignment(costs):
    r, c = linear_sum_assignment(costs)
    order = np.empty(len(costs), int); order[c] = r
    return order


def schedule_ablation(f, crews=2, days=6):
    costs, _ = build_costs(f, crews, days)
    n = len(f)
    cand = {"FIFO (machine id)": np.arange(n),
            "Risk-only greedy": np.argsort(-f.risk.to_numpy()),
            "Criticality-only greedy": np.argsort(-f.criticality.to_numpy()),
            "Fuzzy greedy (Sugeno)": np.argsort(-f.priority.to_numpy()),
            "Mamdani greedy": np.argsort(-np.array([mamdani_priority(r, c) for r, c in zip(f.risk, f.criticality)]))}
    rows = [dict(method=k, objective=objective(v, costs)) for k, v in cand.items()]
    ga = [genetic(lambda o: objective(o, costs), n, s, cand["Fuzzy greedy (Sugeno)"])[0] for s in (11, 22, 33, 44, 55)]
    rows.append(dict(method="GA best of 5", objective=min(objective(o, costs) for o in ga)))
    ex = exact_assignment(costs); rows.append(dict(method="Exact assignment", objective=objective(ex, costs)))
    t = pd.DataFrame(rows); t["change_vs_FIFO_%"] = 100 * (1 - t.objective / t.objective.iloc[0])
    return t, ex, costs


def ga_vs_exact_many(n_inst=30, n=12, crews=2, days=6):
    gaps = []
    for k in range(n_inst):
        f = synthetic_fleet(n, np.random.default_rng(1000 + k))
        costs, _ = build_costs(f, crews, days)
        g = np.argsort(-f.priority.to_numpy())
        ga = genetic(lambda o: objective(o, costs), n, k, g)[0]
        gaps.append(max(0., 100 * (objective(ga, costs) / objective(exact_assignment(costs), costs) - 1)))
    return np.array(gaps)


def travel_benchmark(n=24, crews=3, days=8, instances=5, seeds=5):
    rows, traces = [], {m: [] for m in METHODS}
    for inst in range(instances):
        f = synthetic_fleet(n, np.random.default_rng(500 + inst))
        costs, _ = build_costs(f, crews, days)
        obj = make_travel_objective(costs, f, crews)
        greedy = np.argsort(-f.priority.to_numpy()); fg = obj(greedy)
        for s in range(seeds):
            for name, fn in METHODS.items():
                best, tr = fn(obj, n, 100 * inst + s, greedy)
                rows.append(dict(instance=inst, seed=s, method=name, objective=obj(best), greedy=fg))
                if inst == 0: traces[name].append(tr[:160])
    df = pd.DataFrame(rows)
    df["gain_vs_greedy_%"] = 100 * (1 - df.objective / df.greedy)
    return df, traces


def fuzzy_sensitivity(f, trials=200, noise=.2, seed=0):
    rng = np.random.default_rng(seed)
    base_pri = f.priority.to_numpy()
    base_ord = exact_assignment(build_costs(f)[0]); base_day = np.empty(len(f), int); base_day[base_ord] = np.arange(len(f)) // 2
    rho, same = [], []
    for _ in range(trials):
        C = np.clip(CONSEQ * rng.uniform(1 - noise, 1 + noise, CONSEQ.shape), 0, 100)
        pri = np.array([fuzzy_priority(r, c, C) for r, c in zip(f.risk, f.criticality)])
        rho.append(spearmanr(base_pri, pri)[0])
        o = exact_assignment(build_costs(f, priority=pri)[0]); day = np.empty(len(f), int); day[o] = np.arange(len(f)) // 2
        same.append(np.mean(day == base_day))
    return dict(spearman_mean=float(np.mean(rho)), spearman_min=float(np.min(rho)),
                same_day_mean=float(np.mean(same)), same_day_min=float(np.min(same)))


def pareto_sweep(f, crews=2, days=6):
    pts = []
    for lam in np.linspace(0, 1, 21):
        _, parts = build_costs(f, crews, days)
        c = lam * parts["fail"] + (1 - lam) * parts["other"]
        o = exact_assignment(c); idx = (o, np.arange(len(o)))
        pts.append((lam, parts["fail"][idx].sum(), parts["other"][idx].sum()))
    return pd.DataFrame(pts, columns=["lambda", "expected_failure_loss", "production_urgency_maint_cost"])


# ----------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------
def figures(table, probs, yte, chosen, cm, pso, tb_df, traces, f, order, pareto, gaps, thr):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    t = table.set_index("model")[["precision", "recall", "F1"]]
    ax = t.plot.bar(figsize=(9, 4), rot=25); ax.set_title("Held-out machines: classification metrics"); plt.tight_layout()
    plt.savefig(FIG / "fig_metrics.png"); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, p in probs.items():
        fp, tp, _ = roc_curve(yte, p); ax[0].plot(fp, tp, label=k)
        pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=k)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=7)
    pc = probs[chosen]; bins = np.linspace(0, 1, 11); mid, obs = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (pc >= a) & (pc <= b)
        if m.sum() > 5: mid.append(pc[m].mean()); obs.append(yte[m].mean())
    ax[2].plot(mid, obs, "o-"); ax[2].plot([0, 1], [0, 1], "k--", lw=.7); ax[2].set(title=f"Reliability: {chosen}", xlabel="Predicted", ylabel="Observed")
    plt.tight_layout(); plt.savefig(FIG / "fig_roc_pr_calibration.png"); plt.close()

    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center")
    ax.set(title=f"{chosen}: confusion matrix", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
    plt.tight_layout(); plt.savefig(FIG / "fig_confusion.png"); plt.close()

    g = np.linspace(0, 1, 41); S = np.array([[fuzzy_priority(r, c) for c in g] for r in g])
    M = np.array([[mamdani_priority(r, c) for c in g[::2]] for r in g[::2]])
    fig = plt.figure(figsize=(11, 4.2))
    a1 = fig.add_subplot(121, projection="3d"); X, Y = np.meshgrid(g, g)
    a1.plot_surface(X, Y, S, cmap="viridis"); a1.set(xlabel="Criticality", ylabel="Risk", zlabel="Priority", title="Sugeno surface")
    a2 = fig.add_subplot(122, projection="3d"); X2, Y2 = np.meshgrid(g[::2], g[::2])
    a2.plot_surface(X2, Y2, M, cmap="magma"); a2.set(xlabel="Criticality", ylabel="Risk", zlabel="Priority", title="Mamdani surface")
    plt.tight_layout(); plt.savefig(FIG / "fig_fuzzy_surfaces.png"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, tr in traces.items():
        a = np.array(tr); m, s = a.mean(0), a.std(0)
        ax[0].plot(m, label=name); ax[0].fill_between(range(len(m)), m - s, m + s, alpha=.15)
    ax[0].set(title="Convergence, 24 machines / 3 crews / travel cost", xlabel="Generation (x60 evaluations)", ylabel="Objective"); ax[0].legend()
    tb_df.boxplot(column="gain_vs_greedy_%", by="method", ax=ax[1], grid=False)
    ax[1].set(title="Gain over fuzzy-greedy (25 runs / method)", xlabel="", ylabel="% improvement"); fig.suptitle("")
    plt.tight_layout(); plt.savefig(FIG / "fig_scheduler_benchmark.png"); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.2)); cmap = plt.cm.viridis(f.priority / 100)
    for s, m in enumerate(order):
        d = s // 2; ax.barh(f"M{m}", 1, left=d, color=cmap[m], edgecolor="w")
    ax.set(title="Optimal 12-machine schedule (colour = fuzzy priority)", xlabel="Days elapsed"); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(FIG / "fig_gantt.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pareto.expected_failure_loss, pareto.production_urgency_maint_cost, "o-")
    ax.set(title="Pareto front (exact, weighted-sum sweep)", xlabel="Expected failure loss", ylabel="Production + urgency + maintenance")
    plt.tight_layout(); plt.savefig(FIG / "fig_pareto.png"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(pso["history"], "o-"); ax[0].set(title="PSO tuning of RBF (validation AP)", xlabel="Iteration")
    ax[1].hist(gaps, bins=10); ax[1].set(title="GA gap to exact optimum, 30 instances (%)", xlabel="% above optimum")
    plt.tight_layout(); plt.savefig(FIG / "fig_pso_gap.png"); plt.close()


# ----------------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    df = make_dataset(); train, val, test = split_data(df)
    print(f"rows={len(df)} train/val/test machines = {train.machine_id.nunique()}/{val.machine_id.nunique()}/{test.machine_id.nunique()}"
          f" | test rows={len(test)} positives={int(test.failure_within_7_days.sum())}")
    sc, models, chosen, table, probs, thr, iso, cal, pso, (yte,) = train_models(train, val, test)
    pd.set_option("display.width", 200); print("\n== Prediction (held-out machines) ==\n", table.round(4).to_string(index=False))
    print("\nSelected by validation AP:", chosen, "| PSO-tuned RBF:", {k: v for k, v in pso.items() if k != "history"})
    print("Calibration (isotonic on validation):", {k: round(v, 4) for k, v in cal.items()})
    ci = machine_bootstrap(test, probs[chosen], thr[chosen]); print("Machine-level 95% bootstrap CI:", ci)
    seeds = mlp_seed_study(train, test); print("MLP 5-seed variability:", {k: round(v, 4) for k, v in seeds.items()})
    cm = confusion_matrix(yte, probs[chosen] >= thr[chosen])
    table.to_csv(OUT / "prediction_metrics.csv", index=False)

    fleet = make_fleet(sc, models[chosen], iso)
    print("\n== Fleet (12 machines, observed to day 80) ==\n", fleet[["machine", "true_rul", "risk", "criticality", "priority"]].round(3).to_string(index=False))
    abl, ex, costs = schedule_ablation(fleet); print("\n== Scheduling ablation (linear model) ==\n", abl.round(2).to_string(index=False))
    abl.to_csv(OUT / "scheduling_ablation.csv", index=False)
    fleet.to_csv(OUT / "fleet.csv", index=False)
    rho = spearmanr(fleet.priority, [mamdani_priority(r, c) for r, c in zip(fleet.risk, fleet.criticality)])[0]
    print(f"\nSugeno vs Mamdani priority Spearman rho = {rho:.3f}")
    sens = fuzzy_sensitivity(fleet); print("Fuzzy consequent sensitivity (+-20%, 200 trials):", {k: round(v, 3) for k, v in sens.items()})
    gaps = ga_vs_exact_many(); print(f"GA vs exact over 30 random instances: mean gap {gaps.mean():.4f}%  max {gaps.max():.4f}%")
    tb, traces = travel_benchmark()
    summ = tb.groupby("method").agg(mean_obj=("objective", "mean"), std_obj=("objective", "std"),
                                    mean_gain_pct=("gain_vs_greedy_%", "mean")).sort_values("mean_obj")
    print("\n== Sequence-dependent (travel) scheduling: 24 machines, 3 crews, 8 days; 5 instances x 5 seeds ==\n", summ.round(2))
    piv = tb.pivot_table(index=["instance", "seed"], columns="method", values="objective")
    for other in ("PSO", "SA", "Random search"):
        try: print(f"Wilcoxon GA vs {other}: p = {wilcoxon(piv['GA'], piv[other]).pvalue:.4g}")
        except ValueError: print(f"Wilcoxon GA vs {other}: identical results")
    summ.to_csv(OUT / "travel_benchmark_summary.csv"); tb.to_csv(OUT / "travel_benchmark_runs.csv", index=False)
    pareto = pareto_sweep(fleet); pareto.to_csv(OUT / "pareto.csv", index=False)
    figures(table, probs, yte, chosen, cm, pso, tb, traces, fleet, ex, pareto, gaps, thr)
    json.dump(dict(chosen=chosen, calibration=cal, bootstrap_ci=ci, mlp_seeds=seeds, sensitivity=sens,
                   ga_gap_mean=float(gaps.mean()), pso=pso, runtime_s=time.time() - t0),
              open(OUT / "summary.json", "w"), indent=2, default=float)
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
