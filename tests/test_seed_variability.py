"""The seed-variability table in Supplementary Note S4 must follow from the data.

Note S4 reports three quantities per cycle and the deposited per-run file is the
only evidence for them. These tests recompute each column from
`data/seed_variability_runs.csv` and pin the published values, so that the table
and the file cannot drift apart.

The normalizations differ between columns and are not obvious from the table
alone, which is the reason to fix them here:

  seed s.d.          standard deviation of MAE_kappa over all 60 runs of the
                     cycle (3 partitions x 20 seeds), as a percentage of their
                     mean
  median null |dMAE| median absolute difference between every pair of seeds
                     *within* a partition -- pairs across partitions are not
                     comparable, because the partitions have different held-out
                     sets -- pooled over partitions, as a percentage of the
                     cycle mean
  observed |dMAE|    the branch difference reported in Fig. 2(a), as a
                     percentage of the mean of the two campaign MAEs of that
                     cycle. It is normalized by the campaign's own scale, not by
                     the reimplementation's, because the two differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "seed_variability_runs.csv"

#: cycle -> (seed s.d. %, median null |dMAE| %, observed |dMAE| %) in Note S4
NOTE_S4 = {1: (29, 18, 22), 2: (25, 16, 28), 3: (37, 13, 15)}

#: cycle -> (MAE_kappa without augmentation, with augmentation) from Note S3
CAMPAIGN_MAE = {1: (3.740, 3.011), 2: (2.141, 2.850), 3: (3.194, 2.755)}


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return pd.read_csv(RUNS)


def test_file_has_the_runs_note_s4_describes(runs: pd.DataFrame) -> None:
    """20 seeds on each of three partitions, for each of three cycles."""
    assert len(runs) == 180
    assert sorted(runs.cycle.unique()) == [1, 2, 3]
    for cycle, g in runs.groupby("cycle"):
        assert g.partition.nunique() == 3
        assert g.seed.nunique() == 20
        assert len(g) == 60


@pytest.mark.parametrize("cycle", sorted(NOTE_S4))
def test_seed_standard_deviation(cycle: int, runs: pd.DataFrame) -> None:
    v = runs[runs.cycle == cycle].mae_kappa.to_numpy()
    assert round(100 * v.std(ddof=1) / v.mean()) == NOTE_S4[cycle][0]


@pytest.mark.parametrize("cycle", sorted(NOTE_S4))
def test_median_null_difference(cycle: int, runs: pd.DataFrame) -> None:
    g = runs[runs.cycle == cycle]
    diffs = []
    for _, part in g.groupby("partition"):
        v = part.mae_kappa.to_numpy()
        diffs.extend(np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)])
    assert round(100 * np.median(diffs) / g.mae_kappa.mean()) == NOTE_S4[cycle][1]


@pytest.mark.parametrize("cycle", sorted(NOTE_S4))
def test_observed_difference(cycle: int) -> None:
    without, with_aug = CAMPAIGN_MAE[cycle]
    observed = abs(with_aug - without)
    assert round(100 * observed / ((without + with_aug) / 2)) == NOTE_S4[cycle][2]


def test_final_epoch_range_and_weak_correlation(runs: pd.DataFrame) -> None:
    """Note S4: E* ranged from 6 to 194 and correlated only weakly, |r| <= 0.36."""
    assert (runs.E_star.min(), runs.E_star.max()) == (6, 194)
    worst = max(
        abs(np.corrcoef(g.E_star, g[metric])[0, 1])
        for _, g in runs.groupby("cycle")
        for metric in ("mae_kappa", "mae_sane")
    )
    assert worst <= 0.36


def test_observed_difference_is_inside_the_null_range(runs: pd.DataFrame) -> None:
    """The claim the note rests on: no cycle's branch difference is unusual."""
    for cycle in sorted(NOTE_S4):
        g = runs[runs.cycle == cycle]
        diffs = []
        for _, part in g.groupby("partition"):
            v = part.mae_kappa.to_numpy()
            diffs.extend(np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)])
        without, with_aug = CAMPAIGN_MAE[cycle]
        observed_pct = 100 * abs(with_aug - without) / ((without + with_aug) / 2)
        null_pct = 100 * np.asarray(diffs) / g.mae_kappa.mean()
        assert observed_pct <= np.quantile(null_pct, 0.95), (
            f"cycle {cycle}: the branch difference would sit outside the null range"
        )
