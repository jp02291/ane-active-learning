"""The seed-variability table in Supplementary Note S2 must follow from the data.

Note S2 reports, for each cycle and target, the median null difference, the
observed branch difference and its percentile within the null distribution, and
the deposited per-run file is the only evidence for them. The seed standard
deviation is pinned as well because it fixes the scale of the runs. These tests recompute each column from
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
  observed |dMAE|    the branch difference reported in Fig. 3(a), as a
                     percentage of the mean of the two campaign MAEs of that
                     cycle (Supplementary Note S1). It is normalized by the
                     campaign's own scale, not by the reimplementation's,
                     because the two differ.
  percentile         share of the pooled null differences, in percent, that
                     are smaller than the observed difference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "seed_variability_runs.csv"

#: cycle -> (seed s.d. %, median null |dMAE| %, observed |dMAE| %) for kappa in Note S2
NOTE_S2 = {1: (29, 18, 22), 2: (25, 16, 29), 3: (37, 13, 15)}

#: cycle -> (MAE_kappa without augmentation, with augmentation) from Note S1
CAMPAIGN_MAE = {1: (3.740, 3.011), 2: (2.137, 2.850), 3: (3.194, 2.755)}

#: cycle -> (MAE_S_ANE without augmentation, with augmentation) from Note S1
CAMPAIGN_MAE_SANE = {1: (0.549, 0.463), 2: (0.791, 0.867), 3: (0.723, 0.358)}

#: target -> cycle -> (median null |dMAE| %, observed |dMAE| %, percentile) in Note S2
NOTE_S2_TABLE = {
    "mae_kappa": {1: (18, 22, 56), 2: (16, 29, 78), 3: (13, 15, 55)},
    "mae_sane": {1: (17, 17, 51), 2: (22, 9, 22), 3: (15, 68, 96)},
}


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return pd.read_csv(RUNS)


def test_file_has_the_runs_note_s2_describes(runs: pd.DataFrame) -> None:
    """20 seeds on each of three partitions, for each of three cycles."""
    assert len(runs) == 180
    assert sorted(runs.cycle.unique()) == [1, 2, 3]
    for cycle, g in runs.groupby("cycle"):
        assert g.partition.nunique() == 3
        assert g.seed.nunique() == 20
        assert len(g) == 60


@pytest.mark.parametrize("cycle", sorted(NOTE_S2))
def test_seed_standard_deviation(cycle: int, runs: pd.DataFrame) -> None:
    v = runs[runs.cycle == cycle].mae_kappa.to_numpy()
    assert round(100 * v.std(ddof=1) / v.mean()) == NOTE_S2[cycle][0]


@pytest.mark.parametrize("cycle", sorted(NOTE_S2))
def test_median_null_difference(cycle: int, runs: pd.DataFrame) -> None:
    g = runs[runs.cycle == cycle]
    diffs = []
    for _, part in g.groupby("partition"):
        v = part.mae_kappa.to_numpy()
        diffs.extend(np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)])
    assert round(100 * np.median(diffs) / g.mae_kappa.mean()) == NOTE_S2[cycle][1]


@pytest.mark.parametrize("cycle", sorted(NOTE_S2))
def test_observed_difference(cycle: int) -> None:
    without, with_aug = CAMPAIGN_MAE[cycle]
    observed = abs(with_aug - without)
    assert round(100 * observed / ((without + with_aug) / 2)) == NOTE_S2[cycle][2]


def test_final_epoch_range_and_weak_correlation(runs: pd.DataFrame) -> None:
    """Note S2: E* ranged from 6 to 194 and correlated only weakly, |r| <= 0.36."""
    assert (runs.E_star.min(), runs.E_star.max()) == (6, 194)
    worst = max(
        abs(np.corrcoef(g.E_star, g[metric])[0, 1])
        for _, g in runs.groupby("cycle")
        for metric in ("mae_kappa", "mae_sane")
    )
    assert worst <= 0.36


def test_observed_difference_is_inside_the_null_range(runs: pd.DataFrame) -> None:
    """For kappa, no cycle's branch difference lies in the upper tail of the null."""
    for cycle in sorted(NOTE_S2):
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


def _null_and_observed(runs: pd.DataFrame, cycle: int, column: str) -> tuple[np.ndarray, float]:
    g = runs[runs.cycle == cycle]
    diffs = []
    for _, part in g.groupby("partition"):
        v = part[column].to_numpy()
        diffs.extend(np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)])
    campaign = CAMPAIGN_MAE if column == "mae_kappa" else CAMPAIGN_MAE_SANE
    without, with_aug = campaign[cycle]
    observed = 100 * abs(with_aug - without) / ((without + with_aug) / 2)
    return 100 * np.asarray(diffs) / g[column].mean(), observed


@pytest.mark.parametrize("column", sorted(NOTE_S2_TABLE))
@pytest.mark.parametrize("cycle", [1, 2, 3])
def test_note_s2_table(cycle: int, column: str, runs: pd.DataFrame) -> None:
    """Every entry of the Note S2 table, for both targets."""
    null, observed = _null_and_observed(runs, cycle, column)
    median, obs, percentile = NOTE_S2_TABLE[column][cycle]
    assert round(float(np.median(null))) == median
    assert round(observed) == obs
    assert round(100 * float(np.mean(null < observed))) == percentile


def test_only_cycle3_s_ane_reaches_the_upper_tail(runs: pd.DataFrame) -> None:
    """The one difference Note S1 and Note S2 single out."""
    for column in NOTE_S2_TABLE:
        for cycle in (1, 2, 3):
            null, observed = _null_and_observed(runs, cycle, column)
            upper = observed > np.quantile(null, 0.95)
            assert upper == (column == "mae_sane" and cycle == 3)
