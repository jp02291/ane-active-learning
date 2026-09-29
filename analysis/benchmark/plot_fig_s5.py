"""Draw the two panels of Supplementary Fig. S5 from the deposited results.

    python plot_fig_s5.py

Panel (a) is the MAE of kappa and panel (b) the MAE of |S_ANE|. For each model,
small circles are the nine cross-validation splits, the diamond and bar are
their mean and one standard deviation, and the open star is the MAE on the nine
held-out compositions.

The two evaluations are on different footings and the figure should not be read
as a ranking. Cross-validation averages nine partitions of 36 samples; the
held-out set is nine samples evaluated once. The DNN markers come from a single
seeded model rather than the pruned ensemble, so they move more between runs
than the other four.

The horizontal scatter of the circles is cosmetic, drawn from a fixed seed so
the figure is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

MODELS = ["DNN", "KRR", "SVR", "XGB", "GPR"]
COLORS = {
    "DNN": "#1f6fdc",
    "KRR": "#ff7f0e",
    "SVR": "#2ca062",
    "XGB": "#d62728",
    "GPR": "#a970dc",
}
JITTER_SEED = 42

YLABEL = {
    "kxx": r"MAE of $\kappa$ (W m$^{-1}$ K$^{-1}$)",
    "S_ANE": r"MAE of $|S_{\mathrm{ANE}}|$ ($\mu$V K$^{-1}$)",
}


def panel(ax, folds: pd.DataFrame, summary: pd.DataFrame, test: pd.DataFrame,
          target: str, rng: np.random.Generator) -> None:
    for x, model in enumerate(MODELS):
        c = COLORS[model]
        points = folds[(folds.model == model) & (folds.target == target)].MAE.to_numpy()
        ax.scatter(x + rng.uniform(-0.12, 0.12, points.size), points,
                   s=22, color=c, alpha=0.7, linewidths=0, zorder=2)
        row = summary[(summary.model == model) & (summary.target == target)].iloc[0]
        ax.errorbar(x, row.MAE_mean, yerr=row.MAE_std, fmt="D", ms=8,
                    color=c, ecolor=c, elinewidth=1.6, capsize=5, zorder=3)
        held = float(test[(test.model == model) & (test.target == target)].MAE.iloc[0])
        ax.scatter(x + 0.3, held, marker="*", s=190, facecolors="white",
                   edgecolors="k", linewidths=1.2, zorder=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_xlim(-0.6, len(MODELS) - 0.3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(YLABEL[target], fontsize=11)
    ax.tick_params(labelsize=9, direction="in", top=True, right=True)
    for side in ax.spines.values():
        side.set_linewidth(1.4)


def main() -> None:
    folds = pd.read_csv(RESULTS / "cv_fold_metrics.csv", encoding="utf-8-sig")
    summary = pd.read_csv(RESULTS / "cv_metrics_summary.csv", encoding="utf-8-sig")
    test = pd.read_csv(RESULTS / "held_out_test_metrics.csv", encoding="utf-8-sig")

    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    panel(axes[0], folds, summary, test, "kxx", rng)
    panel(axes[1], folds, summary, test, "S_ANE", rng)

    handles = [
        Line2D([], [], ls="", marker="o", ms=6, color="0.4", label="Cross-validation split"),
        Line2D([], [], ls="-", marker="D", ms=6, color="k", label="Mean ± SD of nine splits"),
        Line2D([], [], ls="", marker="*", ms=11, markerfacecolor="white",
               markeredgecolor="k", label="Held-out test set"),
    ]
    axes[1].legend(handles=handles, fontsize=8.5, loc="upper right", frameon=False)
    for ax, tag in zip(axes, "ab"):
        ax.annotate(f"({tag})", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-50, 10), textcoords="offset points", fontsize=13, va="bottom")
    fig.tight_layout()

    # matplotlib stamps a creation date into PDF metadata, which would make the
    # deposited file differ on every run. Suppressing it keeps the hash stable.
    for suffix, kwargs in (
        ("png", dict(dpi=400)),
        ("pdf", dict(metadata={"CreationDate": None})),
    ):
        out = RESULTS / f"fig_S5.{suffix}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
