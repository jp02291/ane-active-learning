"""Draw the four panels of Supplementary Fig. S6 from the deposited results.

    python plot_fig_s6.py

Panels (a) and (b) show the repeated cross-validation: one small marker per
split, a diamond at the mean and a bar at one standard deviation over the nine
splits. Panels (c) and (d) show the held-out set, where there is a single value
per model and no spread to draw.

The two evaluations are on different footings and the figure should not be read
as a ranking. Cross-validation averages nine partitions of 36 samples; the
held-out set is nine samples evaluated once. The DNN markers come from a single
seeded model rather than the pruned ensemble, so they move more between runs
than the other four.

The horizontal scatter in (a) and (b) is cosmetic, drawn from a fixed seed so
the figure is reproducible.
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

MODELS = ["DNN", "KRR", "SVR", "XGB", "GPR"]
COLORS = {m: f"C{i}" for i, m in enumerate(MODELS)}
JITTER_SEED = 42

TARGETS = {
    "kxx": dict(
        label=r"MAE (W m$^{-1}$ K$^{-1}$)",
        cv_title="Cross-validation: thermal conductivity",
        test_title="Held-out test: thermal conductivity",
    ),
    "S_ANE": dict(
        label=r"MAE ($\mu$V K$^{-1}$)",
        cv_title=r"Cross-validation: $|S_{\mathrm{ANE}}|$",
        test_title=r"Held-out test: $|S_{\mathrm{ANE}}|$",
    ),
}


def panel_cv(ax, folds: pd.DataFrame, summary: pd.DataFrame, target: str) -> None:
    rng = np.random.default_rng(JITTER_SEED)
    for x, model in enumerate(MODELS):
        points = folds[(folds.model == model) & (folds.target == target)].MAE.to_numpy()
        ax.scatter(
            x + rng.uniform(-0.16, 0.16, points.size), points,
            s=18, color=COLORS[model], alpha=0.75, linewidths=0, zorder=2,
        )
        row = summary[(summary.model == model) & (summary.target == target)].iloc[0]
        ax.errorbar(
            x, row.MAE_mean, yerr=row.MAE_std,
            fmt="D", ms=7, color=COLORS[model], ecolor=COLORS[model],
            elinewidth=1.4, capsize=4, zorder=3,
        )
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(TARGETS[target]["label"])
    ax.set_title(TARGETS[target]["cv_title"], fontsize=10)
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def panel_test(ax, test: pd.DataFrame, target: str) -> None:
    values = [
        float(test[(test.model == m) & (test.target == target)].MAE.iloc[0])
        for m in MODELS
    ]
    ax.scatter(range(len(MODELS)), values, marker="D", s=70, color="C0", zorder=3)
    for x, v in enumerate(values):
        ax.annotate(
            f"{v:.2f}", xy=(x, v), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=8.5,
        )
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel(TARGETS[target]["label"])
    ax.set_title(TARGETS[target]["test_title"], fontsize=10)
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    folds = pd.read_csv(RESULTS / "cv_fold_metrics.csv")
    summary = pd.read_csv(RESULTS / "cv_metrics_summary.csv")
    test = pd.read_csv(RESULTS / "held_out_test_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    panel_cv(axes[0, 0], folds, summary, "kxx")
    panel_cv(axes[0, 1], folds, summary, "S_ANE")
    panel_test(axes[1, 0], test, "kxx")
    panel_test(axes[1, 1], test, "S_ANE")

    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.tick_params(labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.annotate(
            f"({tag})", xy=(0, 1), xycoords="axes fraction",
            xytext=(-42, 14), textcoords="offset points",
            fontsize=12, fontweight="bold", va="bottom",
        )
    fig.tight_layout()

    # matplotlib stamps a creation date into PDF metadata, which would make the
    # deposited file differ on every run. Suppressing it keeps the hash stable.
    for suffix, kwargs in (
        ("png", dict(dpi=400)),
        ("pdf", dict(metadata={"CreationDate": None})),
    ):
        out = RESULTS / f"fig_S6.{suffix}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
