"""Retrospective comparison of candidate-acquisition rules (Supplementary Fig. S5).

Supplementary Fig. S5 asks a narrow question: given the compositions this
campaign actually measured, how quickly would each of four acquisition rules
have reached the best one? The pool is frozen to the measured compositions, a
starting subset is handed to each rule, and each rule then requests one batch
at a time. Every requested point is already measured, so no new experiment is
involved.

Four rules are compared, all on the same Gaussian-process surrogate and the
same 15-dimensional inputs:

    gp_pareto_unc   the rule used in this work: a non-dominated filter in
                    (|S_ANE|, 1/kappa, U) followed by ranking on |S_ANE|/kappa
                    for the exploitation half and on U for the exploration half
    gp_ratio_ucb    an upper-confidence bound on the scalar |S_ANE|/kappa,
                    the closest single-objective analogue of Algorithm S5
    gp_ehvi         expected hypervolume improvement over the observed front
    random          uniform sampling without replacement

What this cannot answer
-----------------------
A pool-based replay can only select compositions that were measured, and the
pool was itself assembled by the rule under test. The comparison is therefore
conditional on this candidate pool: it reports how the rules ordered a fixed
set, not what a different rule would have discovered in the full composition
space. Supplementary Fig. S5 states this bound explicitly and it should not be
read as a general ranking of acquisition functions.

Inputs and conventions
----------------------
The pool is `data/data.csv`, the 70-composition dataset, with `cycle_added`
marking the 45 initial entries as cycle 0. Features come from
`ane.features.featurize`, so this baseline and the campaign surrogate see
identical inputs: the same ILR construction, the same element order, and the
element properties of Supplementary Table S8(a).

Objectives are the two the manuscript maximizes, 1/kappa and |S_ANE|.
Thermal conductivity is modeled on a log scale so that the posterior cannot
place mass on non-positive kappa, which would make 1/kappa diverge.

    python run_benchmark.py                 # the three published configurations
    python run_benchmark.py --only batch1   # one of them
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ane.elements import ELEMENTS  # noqa: E402
from ane.features import featurize  # noqa: E402

DATA = REPO / "data" / "data.csv"
OUT = HERE / "results"

#: Kernel hyperparameters routinely hit their bounds on datasets this small.
#: That is expected here and does not invalidate the posterior, so the warnings
#: are silenced to keep the output readable.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

#: Root of the per-repetition seeds. Repetition s uses RNG_BASE + s for both
#: the starting subset and the acquisition draws, so every rule sees the same
#: starting data at a given repetition and the comparison stays paired.
RNG_BASE = 20250101

#: The configurations behind the published panels. Each names the pool, the
#: fraction of the cycle-0 rows the campaigns start from, the batch size and
#: the number of acquisition cycles.
CONFIGURATIONS = {
    "batch1": dict(
        pool="all",
        cycle0_subsample=1.0,
        batch=1,
        cycles=25,
        seeds=50,
        note="Supplementary Fig. S5(a): unit batch size from the full initial dataset",
    ),
    "robust": dict(
        pool="all",
        cycle0_subsample=0.8,
        batch=5,
        cycles=5,
        seeds=50,
        note="Supplementary Fig. S5(b, c): randomized starting data",
    ),
    "measured_only": dict(
        pool="measured",
        cycle0_subsample=1.0,
        batch=1,
        cycles=25,
        seeds=50,
        note="Supplementary Note S1: the same comparison with the 13 reconstructed "
        "literature entries removed from the training set",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pool(which: str) -> pd.DataFrame:
    """The measured pool, optionally without the reconstructed literature rows.

    `kappa_source` marks the 13 entries whose kappa was reconstructed rather
    than measured. Dropping them removes them from the starting data and from
    the candidate pool alike, which is what Supplementary Note S1 reports.
    """
    df = pd.read_csv(DATA)
    if which == "measured":
        df = df[df.kappa_source == "measured"]
    elif which != "all":
        raise ValueError("pool must be 'all' or 'measured'")
    return df.reset_index(drop=True)


def objectives(kappa: np.ndarray, s_ane: np.ndarray) -> np.ndarray:
    """The two maximization objectives of the manuscript, (1/kappa, |S_ANE|)."""
    return np.column_stack(
        [1.0 / np.asarray(kappa, float), np.abs(np.asarray(s_ane, float))]
    )


def hypervolume_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Dominated hypervolume for two maximization objectives."""
    pts = front[(front[:, 0] > ref[0]) & (front[:, 1] > ref[1])]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(hv)


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, maximization on every column."""
    n = Y.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def _make_gp(seed: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(1), length_scale_bounds=(1e-2, 1e3), nu=2.5
    ) + WhiteKernel(1e-2, (1e-6, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=4, random_state=seed
    )


@dataclass
class GPPosterior:
    kappa_mu: np.ndarray
    kappa_sd: np.ndarray
    s_mu: np.ndarray
    s_sd: np.ndarray


def fit_predict_gp(X_train, kappa_train, s_train, X_query, seed: int) -> GPPosterior:
    """Independent GPs for log-kappa and |S_ANE|.

    log-kappa is modeled rather than kappa so that the posterior cannot place
    mass on non-positive thermal conductivity, which would make 1/kappa
    diverge. Predictions are pushed back through exp() with the log-normal
    correction applied to the mean.
    """
    scaler = StandardScaler().fit(X_train)
    Xt, Xq = scaler.transform(X_train), scaler.transform(X_query)

    gp_k = _make_gp(seed).fit(Xt, np.log(kappa_train))
    gp_s = _make_gp(seed + 1).fit(Xt, s_train)

    log_mu, log_sd = gp_k.predict(Xq, return_std=True)
    kappa_mu = np.exp(log_mu + 0.5 * log_sd**2)
    kappa_sd = kappa_mu * np.sqrt(np.clip(np.exp(log_sd**2) - 1.0, 0, None))

    s_mu, s_sd = gp_s.predict(Xq, return_std=True)
    return GPPosterior(
        kappa_mu, np.maximum(kappa_sd, 1e-9), s_mu, np.maximum(s_sd, 1e-9)
    )


def acq_random(rng, n_pool, batch, **_):
    return rng.choice(n_pool, size=batch, replace=False)


def _hvi_2d_batch(a, b, front, ref=(0.0, 0.0)):
    """Exact hypervolume improvement of points (a, b) over a fixed 2-D front.

    For two maximization objectives the dominated region is a staircase, so the
    improvement has a closed form and the Monte-Carlo loop only has to average
    it. The dominated area is a sum of horizontal slabs; clipping each corner
    to (a, b) gives the part of the candidate rectangle that is already
    covered, and the improvement is what is left of that rectangle.
    """
    rx, ry = ref
    a = np.maximum(a - rx, 0.0)
    b = np.maximum(b - ry, 0.0)
    if front.size == 0:
        return a * b

    pts = front[(front[:, 0] > rx) & (front[:, 1] > ry)]
    if pts.size == 0:
        return a * b
    pts = pts[np.argsort(-pts[:, 0])]
    xs = pts[:, 0] - rx
    ys = pts[:, 1] - ry

    covered = np.zeros_like(a)
    prev = np.zeros_like(b)
    for xi, yi in zip(xs, ys):
        cur = np.minimum(yi, b)
        covered += np.minimum(xi, a) * np.maximum(cur - prev, 0.0)
        prev = np.maximum(prev, cur)
    return np.maximum(a * b - covered, 0.0)


def acq_ehvi(rng, post: GPPosterior, Y_obs, batch, n_mc=512, **_):
    """Monte-Carlo expected hypervolume improvement over the observed front."""
    front = Y_obs[pareto_mask(Y_obs)]
    ref = (0.0, 0.0)

    n = len(post.kappa_mu)
    kappa_s = np.exp(
        rng.normal(
            np.log(np.maximum(post.kappa_mu, 1e-6)),
            post.kappa_sd / np.maximum(post.kappa_mu, 1e-6),
            size=(n_mc, n),
        )
    )
    s_s = rng.normal(post.s_mu, post.s_sd, size=(n_mc, n))
    f1 = 1.0 / np.maximum(kappa_s, 1e-6)
    f2 = np.abs(s_s)

    ehvi = _hvi_2d_batch(f1, f2, front, ref).mean(axis=0)
    return np.argsort(-ehvi)[:batch]


def acq_ratio_ucb(rng, post: GPPosterior, batch, beta=2.0, **_):
    """Upper-confidence bound on the scalar figure of merit |S_ANE| / kappa.

    Algorithm S5 line 20 ranks the performance pool by exactly this ratio, so
    this is the closest single-objective analogue of what the campaign does and
    it isolates what the three-objective construction adds.
    """
    ratio_mu = post.s_mu / np.maximum(post.kappa_mu, 1e-6)
    rel = np.sqrt(
        (post.s_sd / np.maximum(np.abs(post.s_mu), 1e-6)) ** 2
        + (post.kappa_sd / np.maximum(post.kappa_mu, 1e-6)) ** 2
    )
    return np.argsort(-(ratio_mu + beta * np.abs(ratio_mu) * rel))[:batch]


def acq_pareto_uncertainty(rng, post: GPPosterior, batch, **_):
    """The selection rule of this work, applied to the GP posterior.

    Half the batch is taken from the predicted Pareto front ranked by
    S_mean / kappa_mean (exploitation) and half by the median-normalized
    disagreement score U (exploration), matching Algorithm S5.

    The exploitation share is floored at one. Without the floor, `batch // 2`
    is zero at unit batch size and the rule degenerates into pure uncertainty
    sampling, which is a different algorithm. With the floor, unit batch size
    reduces to taking the best-ranked Pareto point, the natural single-pick
    reduction of the batch rule, and batch sizes of two or more are unaffected
    (5 of 10, as in the campaign).
    """
    f1, f2 = 1.0 / np.maximum(post.kappa_mu, 1e-6), post.s_mu
    sd_f1 = post.kappa_sd / np.maximum(post.kappa_mu, 1e-6) ** 2
    U = np.sqrt(
        (sd_f1 / max(np.median(sd_f1), 1e-12)) ** 2
        + (post.s_sd / max(np.median(post.s_sd), 1e-12)) ** 2
    )
    front = np.where(pareto_mask(np.column_stack([f1, f2, U])))[0]
    if front.size == 0:
        front = np.arange(len(f1))

    n_exploit = max(1, batch // 2)
    ratio = post.s_mu[front] / np.maximum(post.kappa_mu[front], 1e-6)
    exploit = front[np.argsort(-ratio)][:n_exploit]
    rest = np.setdiff1d(front, exploit)
    explore = rest[np.argsort(-U[rest])][: batch - n_exploit]
    picked = np.concatenate([exploit, explore])

    if picked.size < batch:  # fall back to filling from outside the front
        extra = np.setdiff1d(np.argsort(-U), picked)[: batch - picked.size]
        picked = np.concatenate([picked, extra])
    return picked[:batch]


STRATEGIES = {
    "random": acq_random,
    "gp_ehvi": acq_ehvi,
    "gp_ratio_ucb": acq_ratio_ucb,
    "gp_pareto_unc": acq_pareto_uncertainty,
}


def run_campaign(X, kappa, s_ane, strategy, seed, n_cycles, batch, seed_mask):
    """Replay one campaign from a fixed starting set."""
    rng = np.random.default_rng(seed)
    n = len(kappa)
    observed = np.zeros(n, dtype=bool)
    observed[np.asarray(seed_mask, bool)] = True

    ratio = np.abs(s_ane) / kappa
    ref = np.array([0.0, 0.0])
    trace = [
        dict(
            n_exp=int(observed.sum()),
            best_ratio=float(ratio[observed].max()),
            hv=hypervolume_2d(objectives(kappa[observed], s_ane[observed]), ref),
        )
    ]

    for _ in range(n_cycles):
        pool = np.where(~observed)[0]
        if pool.size == 0:
            break
        k = min(batch, pool.size)

        if strategy == "random":
            pick_local = acq_random(rng, pool.size, k)
        else:
            post = fit_predict_gp(
                X[observed], kappa[observed], np.abs(s_ane[observed]), X[pool], seed
            )
            Y_obs = objectives(kappa[observed], s_ane[observed])
            pick_local = STRATEGIES[strategy](rng=rng, post=post, Y_obs=Y_obs, batch=k)

        observed[pool[np.asarray(pick_local, int)]] = True
        trace.append(
            dict(
                n_exp=int(observed.sum()),
                best_ratio=float(ratio[observed].max()),
                hv=hypervolume_2d(objectives(kappa[observed], s_ane[observed]), ref),
            )
        )
    return trace


def starting_mask(base_mask: np.ndarray, fraction: float, repetition: int) -> np.ndarray:
    """The cycle-0 rows, or a fresh random share of them for each repetition.

    A fraction below one makes the otherwise deterministic rules stochastic
    through the data, which is what produces the bands in Supplementary
    Fig. S5(c). The same subset is used by every rule at a given repetition, so
    the comparison stays paired. Compositions acquired during the campaign are
    never placed in the starting set, so the pool optimum cannot leak into it.
    """
    if fraction >= 1.0:
        return base_mask
    idx = np.where(base_mask)[0]
    keep = np.random.default_rng(RNG_BASE + repetition).choice(
        idx, size=int(round(fraction * len(idx))), replace=False
    )
    mask = np.zeros(len(base_mask), bool)
    mask[keep] = True
    return mask


def run_configuration(name: str, cfg: dict) -> pd.DataFrame:
    df = load_pool(cfg["pool"])
    X = featurize(df[list(ELEMENTS)].to_numpy(float))
    kappa = df["kxx"].to_numpy(float)
    s_ane = df["S_ANE"].to_numpy(float)
    base_mask = (df["cycle_added"] == 0).to_numpy()

    print(f"[{name}] {cfg['note']}")
    print(
        f"  pool = {len(df)} compositions, features = {X.shape[1]}, "
        f"cycle-0 rows = {int(base_mask.sum())}"
    )
    print(f"  best |S_ANE|/kappa in pool = {(np.abs(s_ane) / kappa).max():.6f} um/A")

    rows = []
    for strategy in STRATEGIES:
        for repetition in range(cfg["seeds"]):
            mask = starting_mask(base_mask, cfg["cycle0_subsample"], repetition)
            for rec in run_campaign(
                X,
                kappa,
                s_ane,
                strategy,
                RNG_BASE + repetition,
                cfg["cycles"],
                cfg["batch"],
                mask,
            ):
                rows.append(dict(strategy=strategy, seed=repetition, **rec))
        print(f"  finished {strategy}")
    return pd.DataFrame(rows)


def experiments_to_optimum(df: pd.DataFrame, optimum: float) -> dict:
    """Acquisitions needed to reach the pool optimum, per rule.

    Reported as the median and the mean over repetitions, and as the count of
    repetitions that reached it at all. The rules driven by the surrogate are
    deterministic once the starting set is fixed, so at `cycle0_subsample = 1`
    their spread is zero and only random sampling varies.
    """
    start = int(df.n_exp.min())
    out = {}
    for strategy in STRATEGIES:
        g = df[df.strategy == strategy]
        hits = []
        for _, run in g.groupby("seed"):
            run = run.sort_values("n_exp")
            reached = run[run.best_ratio >= optimum - 1e-9]
            hits.append(int(reached.n_exp.min()) - start if len(reached) else np.nan)
        arr = np.array(hits, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[strategy] = dict(
            median=float(np.median(finite)) if finite.size else None,
            mean=float(finite.mean()) if finite.size else None,
            reached=int(finite.size),
            repetitions=int(arr.size),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=sorted(CONFIGURATIONS),
        help="run a single configuration instead of all three",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else list(CONFIGURATIONS)

    summary = {
        "input": str(DATA.relative_to(REPO)).replace("\\", "/"),
        "input_sha256": sha256(DATA),
        "features": "ane.features.featurize (Supplementary Table S8(a))",
        "rng_base": RNG_BASE,
        "configurations": {},
    }

    for name in names:
        cfg = CONFIGURATIONS[name]
        res = run_configuration(name, cfg)
        path = OUT / f"benchmark_{name}.csv"
        res.to_csv(path, index=False)

        pool_optimum = float(res.best_ratio.max())
        summary["configurations"][name] = dict(
            cfg,
            output=path.name,
            output_sha256=sha256(path),
            pool_optimum=pool_optimum,
            experiments_to_optimum=experiments_to_optimum(res, pool_optimum),
        )
        print(f"  wrote {path.relative_to(REPO)}")
        for strategy, stats in summary["configurations"][name][
            "experiments_to_optimum"
        ].items():
            med = "  n/a" if stats["median"] is None else f"{stats['median']:5.1f}"
            avg = "   n/a" if stats["mean"] is None else f"{stats['mean']:6.2f}"
            print(
                f"    {strategy:<15} median {med}  mean {avg}  "
                f"reached {stats['reached']}/{stats['repetitions']}"
            )
        print()

    if not args.only:
        (OUT / "acquisition_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {(OUT / 'acquisition_summary.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
