"""Draw the three panels of Supplementary Fig. S5 from the deposited results.

    python plot_fig_s5.py

Panel (a) uses `benchmark_batch1.csv`: unit batch size, every campaign starting
from all 45 cycle-0 rows. Panels (b) and (c) use `benchmark_robust.csv`, where
each repetition starts from a fresh random 80% of those rows.

One convention is worth stating because the panels do not show it. Every curve
is a mean over the 50 repetitions. At unit batch size from the full initial
dataset that mean is exact for `gp_ratio_ucb` and `gp_pareto_unc`, which rank
the pool from the posterior alone and so follow one path from a fixed start.
`gp_ehvi` draws Monte-Carlo samples and `random` samples the pool, so for those
two the curve averages 50 different trajectories. The caption should say which
curves are averages. Panels (b) and (c) use the randomized starting data, where
all four rules vary.
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
    "gp_pareto_unc": dict(color="#c0392b", ls="-", label=r"Pareto + $U$ (this work)", z=5),
    "gp_ratio_ucb": dict(color="#2e7d32", ls="-", label=r"GP-UCB on $|S|/\kappa$", z=4),
    "gp_ehvi": dict(color="#1f5c8b", ls="-", label="GP-EHVI", z=3),
    "random": dict(color="0.55", ls="--", label="Random", z=2),
}


def pool_optimum(df: pd.DataFrame) -> float:
    return float(df.best_ratio.max())


def panel_a(ax, df: pd.DataFrame) -> None:
    opt = pool_optimum(df)
    start = int(df.n_exp.min())
    ax.axhline(opt, color="0.3", ls=":", lw=0.9, zorder=1)
    ax.annotate(
        "pool optimum", xy=(start + 1, opt), xytext=(0, 3),
        textcoords="offset points", fontsize=7.2, color="0.3",
    )
    for key in ORDER:
        g = df[df.strategy == key].groupby("n_exp").best_ratio.mean()
        st = STYLE[key]
        ax.step(
            g.index - start, g.to_numpy(), where="post",
            color=st["color"], ls=st["ls"], lw=1.6, label=st["label"], zorder=st["z"],
        )
    ax.set_xlabel("experiments performed")
    ax.set_ylabel(r"best $|S_{\mathrm{ANE}}|/\kappa$ found ($\mu$m A$^{-1}$)")
    ax.set_title("(a) unit batch size", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.2, loc="lower right", frameon=False)


def panel_b(ax, df: pd.DataFrame) -> None:
    opt = pool_optimum(df)
    start = int(df.n_exp.min())
    budgets = sorted(df.n_exp.unique())
    for key in ORDER:
        g = df[df.strategy == key]
        frac = []
        for budget in budgets:
            upto = g[g.n_exp <= budget]
            reached = sum(
                1 for _, run in upto.groupby("seed")
                if (run.best_ratio >= opt - 1e-9).any()
            )
            frac.append(100.0 * reached / g.seed.nunique())
        st = STYLE[key]
        ax.step(
            np.array(budgets) - start, frac, where="post",
            color=st["color"], ls=st["ls"], lw=1.6, label=st["label"], zorder=st["z"],
        )
        ax.annotate(
            f"{frac[-1]:.0f}%",
            xy=(budgets[-1] - start, frac[-1]), xytext=(3, 0),
            textcoords="offset points", fontsize=7.2, color=st["color"], va="center",
        )
    ax.set_ylim(-4, 108)
    ax.set_xlabel("experiments performed")
    ax.set_ylabel("repetitions reaching the pool optimum (%)")
    ax.set_title("(b) randomized starting data", fontsize=9.5, loc="left")


def panel_c(ax, df: pd.DataFrame) -> None:
    start = int(df.n_exp.min())
    for key in ORDER:
        g = df[df.strategy == key].groupby("n_exp").hv
        mean, sd = g.mean(), g.std()
        st = STYLE[key]
        x = mean.index - start
        ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.6,
                label=st["label"], zorder=st["z"])
        ax.fill_between(x, mean - sd, mean + sd, color=st["color"],
                        alpha=0.13, lw=0, zorder=st["z"] - 1)
    ax.set_xlabel("experiments performed")
    ax.set_ylabel("dominated hypervolume")
    ax.set_title("(c) hypervolume, same runs", fontsize=9.5, loc="left")


def main() -> None:
    batch1 = pd.read_csv(RESULTS / "benchmark_batch1.csv")
    robust = pd.read_csv(RESULTS / "benchmark_robust.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    panel_a(axes[0], batch1)
    panel_b(axes[1], robust)
    panel_c(axes[2], robust)
    for ax in axes:
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()

    # matplotlib stamps a creation date into PDF metadata, which would make the
    # deposited file differ on every run. Suppressing it keeps the manifest hash
    # stable, so a reader can check the figure the same way as the CSVs.
    for suffix, kwargs in (
        ("png", dict(dpi=400)),
        ("pdf", dict(metadata={"CreationDate": None})),
    ):
        out = RESULTS / f"fig_S5.{suffix}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
