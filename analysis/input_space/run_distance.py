"""Distances of the cycle-1 nominations from the training data in the model input space.

    python run_distance.py

The Discussion states that the three Co-free Fe-Mn alloys nominated in cycle 1
lay farther from the training data than any other cycle-1 candidate, and that
their two closest training compositions contained 12.5 at.% Mn with |S_ANE| of
1.4 uV/K or less. This script computes those distances as the Methods define
them: Euclidean distances between compositions in the 15-dimensional input
space of the surrogate (seven ILR coordinates and eight descriptors, from
`ane.features.featurize`), with each feature min-max scaled to the cycle-1
real training data.

Inputs are `data/split/cycle1/train.csv` (the 36 cycle-1 training
compositions) and the cycle-1 rows of `data/candidates.csv`. Outputs go to
`results/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ane.features import featurize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]


def scaled_features(train: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Featurize both sets and min-max scale them with the training ranges."""
    x_train = featurize(train[ELEMENTS].to_numpy(dtype=np.float64))
    x_other = featurize(other[ELEMENTS].to_numpy(dtype=np.float64))
    lo, hi = x_train.min(axis=0), x_train.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (x_train - lo) / span, (x_other - lo) / span


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "split" / "cycle1" / "train.csv")
    cand = pd.read_csv(ROOT / "data" / "candidates.csv")
    cand = cand[cand.cycle == 1].reset_index(drop=True)

    s_train, s_cand = scaled_features(train, cand)

    # nearest-neighbour distance of every training composition to the others
    d_tt = np.linalg.norm(s_train[:, None, :] - s_train[None, :, :], axis=2)
    np.fill_diagonal(d_tt, np.inf)
    nn_train = d_tt.min(axis=1)

    d_ct = np.linalg.norm(s_cand[:, None, :] - s_train[None, :, :], axis=2)
    rows = []
    for i, cand_row in cand.iterrows():
        order = np.argsort(d_ct[i])
        first, second = order[0], order[1]
        rows.append(
            dict(
                label=cand_row.label,
                selection_type=cand_row.selection_type,
                contains_Co=bool(cand_row.Co > 0),
                contains_Mn=bool(cand_row.Mn > 0),
                predicted_S_ANE=cand_row.pred_S_ANE,
                measured_S_ANE=cand_row.measured_S_ANE,
                nn_distance=round(float(d_ct[i, first]), 4),
                nearest_train=train.label[first],
                nearest_train_Mn=train.Mn[first],
                nearest_train_S_ANE=train.S_ANE[first],
                second_train=train.label[second],
                second_train_distance=round(float(d_ct[i, second]), 4),
                second_train_Mn=train.Mn[second],
                second_train_S_ANE=train.S_ANE[second],
            )
        )
    table = pd.DataFrame(rows).sort_values("nn_distance", ascending=False)

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "cycle1_candidate_distances.csv", index=False)

    fe_mn = table[table.contains_Mn & ~table.contains_Co]
    summary = {
        "definition": (
            "Euclidean distance in the 15-dimensional input space of ane.features.featurize, "
            "each feature min-max scaled to the 36 cycle-1 training compositions"
        ),
        "training_nn_distance": {
            "median": round(float(np.median(nn_train)), 3),
            "max": round(float(nn_train.max()), 3),
        },
        "fe_mn_cycle1_candidates": {
            "labels": fe_mn.label.tolist(),
            "nn_distance_min": round(float(fe_mn.nn_distance.min()), 3),
            "nn_distance_max": round(float(fe_mn.nn_distance.max()), 3),
            "max_S_ANE_of_two_nearest_training_entries": round(
                float(max(fe_mn.nearest_train_S_ANE.max(), fe_mn.second_train_S_ANE.max())), 3
            ),
        },
        "largest_nn_distance_among_other_cycle1_candidates": round(
            float(table[~table.label.isin(fe_mn.label)].nn_distance.max()), 3
        ),
    }
    (RESULTS / "distance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table[["label", "selection_type", "nn_distance", "nearest_train"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
