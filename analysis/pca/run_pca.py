"""Principal component analysis of the nominated compositions (Fig. 3d).

The biplot in Fig. 3(d) projects the 30 compositions nominated over the three
active-learning cycles onto their first two principal components. The analysis
runs on the eight nominal atomic fractions, not on the ILR coordinates used as
surrogate inputs, so that the loading directions read directly as element
contents.

Two conventions here are load-bearing.

The eight fractions are standardized to zero mean and unit variance before the
decomposition. Mean-centering alone gives a different projection -- Fe and Ga
carry the largest raw spread and would dominate -- so the reported 31.0% / 27.3%
variance split is specific to the standardized run. `test_pca_reproduction.py`
pins both figures.

The loading arrows in the published figure are drawn with a common display
scaling (`ARROW_SCALE`); their directions are meaningful, their absolute lengths
are not. The scaling is applied only when writing the plotting artifacts, never
to `pca.components_` as reported in `pca_loadings.csv`.

Input is `data/candidates.csv`, which carries the cycle index and the
exploitation/exploration label alongside the composition, so the figure can be
rebuilt without any separate per-cycle file.

    python run_pca.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]
CANDIDATES = REPO / "data" / "candidates.csv"
OUT = HERE / "results"

N_COMPONENTS = 2
ARROW_SCALE = 4.0  # display scaling for the loading arrows only
CYCLE_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#C44E52"}
TYPE_MARKERS = {"exploitation": "o", "exploration": "^"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidates() -> pd.DataFrame:
    """The 30 nominated compositions, ordered by cycle then selection type."""
    df = pd.read_csv(CANDIDATES)

    missing = [c for c in ELEMENTS + ["cycle", "selection_type"] if c not in df.columns]
    if missing:
        raise KeyError(f"{CANDIDATES.name} is missing columns {missing}")
    if len(df) != 30:
        raise ValueError(f"expected 30 nominated compositions, found {len(df)}")

    sums = df[ELEMENTS].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=5e-3):
        off = df.loc[~np.isclose(sums, 1.0, atol=5e-3), "label"].tolist()
        raise ValueError(f"compositions do not close to unit sum: {off}")

    return df


def run_pca(df: pd.DataFrame) -> tuple[PCA, np.ndarray]:
    """Standardize the eight fractions, then take the first two components."""
    scaled = StandardScaler().fit_transform(df[ELEMENTS].to_numpy(dtype=float))
    pca = PCA(n_components=N_COMPONENTS)
    scores = pca.fit_transform(scaled)
    return pca, scores


def plot(df: pd.DataFrame, pca: PCA, scores: np.ndarray, path: Path) -> None:
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    for cycle, sub in df.groupby("cycle"):
        for stype, s2 in sub.groupby("selection_type"):
            idx = s2.index.to_numpy()
            ax.scatter(
                scores[idx, 0],
                scores[idx, 1],
                c=CYCLE_COLORS.get(int(cycle), "0.4"),
                marker=TYPE_MARKERS.get(stype, "s"),
                s=90,
                edgecolor="white",
                linewidth=0.8,
                label=f"cycle {int(cycle)}, {stype}",
                zorder=3,
            )

    # loading arrows: direction from components_, length scaled for legibility
    for i, elem in enumerate(ELEMENTS):
        x, y = pca.components_[0, i] * ARROW_SCALE, pca.components_[1, i] * ARROW_SCALE
        ax.arrow(0, 0, x, y, color="gray", alpha=0.8, width=0.01,
                 head_width=0.12, length_includes_head=True, zorder=2)
        ax.text(x * 1.12, y * 1.12, elem, color="gray", ha="center", va="center")

    ax.axhline(0, color="0.85", lw=0.8, zorder=1)
    ax.axvline(0, color="0.85", lw=0.8, zorder=1)
    ax.set_xlabel(f"Principal component 1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"Principal component 2 ({ev[1]:.1f}%)")
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_candidates()
    pca, scores = run_pca(df)
    ev = pca.explained_variance_ratio_ * 100

    scores_df = df[["label", "cycle", "selection_type"]].copy()
    scores_df["PC1"] = scores[:, 0]
    scores_df["PC2"] = scores[:, 1]
    scores_df.to_csv(OUT / "pca_scores.csv", index=False)

    # unscaled loadings; the figure applies ARROW_SCALE for display only
    pd.DataFrame(
        {"element": ELEMENTS, "PC1": pca.components_[0], "PC2": pca.components_[1]}
    ).to_csv(OUT / "pca_loadings.csv", index=False)

    plot(df, pca, scores, OUT / "pca_biplot.png")

    summary = {
        "n_compositions": int(len(df)),
        "features": ELEMENTS,
        "preprocessing": "StandardScaler (zero mean, unit variance)",
        "n_components": N_COMPONENTS,
        "explained_variance_ratio_percent": [round(float(v), 1) for v in ev],
        "explained_variance_cumulative_percent": round(float(ev.sum()), 1),
        "arrow_scale_display_only": ARROW_SCALE,
        "input_sha256": sha256(CANDIDATES),
    }
    (OUT / "pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"PC1 {ev[0]:.1f}%   PC2 {ev[1]:.1f}%   cumulative {ev.sum():.1f}%")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
