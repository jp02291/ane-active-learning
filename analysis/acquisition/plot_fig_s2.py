"""Draw Supplementary Fig. S2 and the hypervolume companion from the deposited results.

    python plot_fig_s2.py

Panel (a) uses `benchmark_batch1.csv`: unit batch size, every campaign starting
from all 45 cycle-0 rows. Panel (b) uses `benchmark_robust.csv`, where each
repetition starts from a fresh random 80% of those rows and five compositions
are chosen at a time.

The dominated hypervolume of the same `robust` runs is not a panel of Fig. S2.
The legend of Fig. S2 summarizes it and points to this archive, so it is drawn
here as a separate file, `hypervolume_robust.png`.

One convention is worth stating because the panels do not show it. Every curve
in (a) is a mean over the 50 repetitions. At unit batch size from the full
initial dataset that mean is exact for `gp_ratio_ucb` and `gp_pareto_unc`,
which rank the pool from the posterior alone and so follow one path from a
fixed start. `gp_ehvi` draws Monte-Carlo samples and `random` samples the pool,
so for those two the curve averages 50 different trajectories. In (b) all four
rules vary through the starting data as well.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ORDER = ["gp_pareto_unc", "gp_ratio_ucb", "gp_ehvi", "random"]
STYLE = {
    "gp_pareto_unc": dict(color="#d62728", ls="-", label=r"Pareto + $U$", z=5),
    "gp_ratio_ucb": dict(color="#2ca02c", ls="-", label="GP-UCB", z=4),
    "gp_ehvi": dict(color="#1f77b4", ls="-", label="GP-EHVI", z=3),
    "random": dict(color="0.55", ls="--", label="Random", z=2),
}


def pool_optimum(df: pd.DataFrame) -> float:
    return float(df.best_ratio.max())


def panel_a(ax, df: pd.DataFrame) -> None:
    """Best |S_ANE|/kappa found so far, one composition per step."""
    opt = pool_optimum(df)
    start = int(df.n_exp.min())
    ax.axhline(opt, color="k", ls=":", lw=1.4, zorder=1)
    for key in ORDER:
        g = df[df.strategy == key].groupby("n_exp").best_ratio.mean()
        st = STYLE[key]
        ax.step(
            g.index - start, g.to_numpy(), where="post",
            color=st["color"], ls=st["ls"], lw=1.8, label=st["label"], zorder=st["z"],
        )
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel(r"Best $|S_{\mathrm{ANE}}|/\kappa$ ($\mu$m A$^{-1}$)")
    ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor="k", fancybox=False)


def reached_fraction(df: pd.DataFrame, key: str, opt: float) -> tuple[list[int], list[float]]:
    g = df[df.strategy == key]
    budgets = sorted(df.n_exp.unique())
    frac = []
    for budget in budgets:
        upto = g[g.n_exp <= budget]
        reached = sum(
            1 for _, run in upto.groupby("seed") if (run.best_ratio >= opt - 1e-9).any()
        )
        frac.append(100.0 * reached / g.seed.nunique())
    return budgets, frac


def panel_b(ax, df: pd.DataFrame) -> None:
    """Share of repetitions that had found the best composition, five per step."""
    opt = pool_optimum(df)
    start = int(df.n_exp.min())
    for key in ORDER:
        budgets, frac = reached_fraction(df, key, opt)
        st = STYLE[key]
        ax.step(
            np.array(budgets) - start, frac, where="post",
            color=st["color"], ls=st["ls"], lw=1.8, label=st["label"], zorder=st["z"],
        )
        ax.annotate(
            f"{frac[-1]:.0f}%",
            xy=(budgets[-1] - start, frac[-1]), xytext=(4, 0),
            textcoords="offset points", fontsize=8, color=st["color"], va="center",
        )
    ax.set_ylim(-5, 108)
    ax.set_xlim(-1.2, 29.5)
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel("Repetitions reaching pool optimum (%)")
    ax.legend(fontsize=8, loc="upper left", frameon=True, edgecolor="k", fancybox=False)


def hypervolume(ax, df: pd.DataFrame) -> None:
    """Dominated hypervolume of the characterized set, mean and s.d. over repetitions."""
    start = int(df.n_exp.min())
    for key in ORDER:
        g = df[df.strategy == key].groupby("n_exp").hv
        mean, sd = g.mean(), g.std()
        st = STYLE[key]
        x = mean.index - start
        ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.8,
                label=st["label"], zorder=st["z"])
        ax.fill_between(x, mean - sd, mean + sd, color=st["color"],
                        alpha=0.13, lw=0, zorder=st["z"] - 1)
    ax.set_xlabel("Number of experiments")
    ax.set_ylabel(r"Dominated hypervolume in the (1/$\kappa$, $|S_{\mathrm{ANE}}|$) plane")
    ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor="k", fancybox=False)


def finish(ax) -> None:
    ax.tick_params(labelsize=9, direction="in", top=True, right=True)
    for side in ax.spines.values():
        side.set_linewidth(1.4)


def save(fig, stem: str) -> None:
    # matplotlib stamps a creation date into PDF metadata, which would make the
    # deposited file differ on every run. Suppressing it keeps the manifest hash
    # stable, so a reader can check the figure the same way as the CSVs.
    for suffix, kwargs in (
        ("png", dict(dpi=400)),
        ("pdf", dict(metadata={"CreationDate": None})),
    ):
        out = RESULTS / f"{stem}.{suffix}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out.relative_to(HERE.parents[1])}")


def main() -> None:
    batch1 = pd.read_csv(RESULTS / "benchmark_batch1.csv")
    robust = pd.read_csv(RESULTS / "benchmark_robust.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    panel_a(axes[0], batch1)
    panel_b(axes[1], robust)
    for ax, tag in zip(axes, "ab"):
        finish(ax)
        ax.annotate(f"({tag})", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-46, 10), textcoords="offset points", fontsize=13, va="bottom")
    fig.tight_layout()
    save(fig, "fig_S2")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    hypervolume(ax, robust)
    finish(ax)
    fig.tight_layout()
    save(fig, "hypervolume_robust")
    plt.close(fig)


if __name__ == "__main__":
    main()
