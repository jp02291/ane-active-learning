"""The Supplementary Fig. S5 comparison must stay reproducible and stay honest.

`analysis/acquisition/run_benchmark.py` replays the campaign on the measured
pool for four acquisition rules. Three things about it can break without
changing anything a reader can see.

The descriptor source can drift. Running the rules on descriptors the campaign
surrogate never used would change every number in the panels while changing
nothing a reader can see, so these tests pin the featurization to
`ane.features`.

The deposited results can drift from the summary that quotes them, or from the
manuscript. The published acquisition counts are pinned here against the CSVs.

The pool can drift. The comparison is only meaningful if the starting set is
the 45 cycle-0 rows and the target is the campaign optimum, so both are pinned.

A full re-run is far too slow for a test suite, so the harness itself is
exercised on one short campaign and checked for determinism instead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "acquisition" / "run_benchmark.py"
RESULTS = ROOT / "analysis" / "acquisition" / "results"
SUMMARY = RESULTS / "acquisition_summary.json"

STRATEGIES = ("random", "gp_ehvi", "gp_ratio_ucb", "gp_pareto_unc")

#: median acquisitions to the pool optimum, as reported for each panel
PUBLISHED_MEDIANS = {
    "batch1": {"gp_pareto_unc": 2, "gp_ratio_ucb": 15, "gp_ehvi": 15, "random": 8.5},
    "robust": {"gp_pareto_unc": 10, "gp_ratio_ucb": 20, "gp_ehvi": 25, "random": 15},
    "measured_only": {"gp_pareto_unc": 4, "gp_ratio_ucb": 9, "gp_ehvi": 10, "random": 8.5},
}

#: repetitions that reached the pool optimum under randomized starting data
PUBLISHED_ROBUST_REACHED = {
    "gp_pareto_unc": 50,
    "gp_ratio_ucb": 43,
    "gp_ehvi": 30,
    "random": 35,
}

#: the best |S_ANE| / kappa present in the measured pool, Fe0.75Ga0.13Al0.12
POOL_OPTIMUM = 0.3591703782490195


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(RESULTS / f"benchmark_{name}.csv")
        for name in PUBLISHED_MEDIANS
    }


def _medians(df: pd.DataFrame) -> dict[str, float]:
    start = int(df.n_exp.min())
    out = {}
    for strategy in STRATEGIES:
        hits = []
        for _, run in df[df.strategy == strategy].groupby("seed"):
            run = run.sort_values("n_exp")
            reached = run[run.best_ratio >= POOL_OPTIMUM - 1e-9]
            hits.append(int(reached.n_exp.min()) - start if len(reached) else np.nan)
        finite = np.array(hits, float)
        finite = finite[np.isfinite(finite)]
        out[strategy] = float(np.median(finite)) if finite.size else float("nan")
    return out


def test_features_come_from_the_campaign_descriptor_set(mod) -> None:
    """Not a local element table. Ga is the entry that separates the two sets."""
    from ane.elements import ELEMENT_PROPS

    assert mod.featurize.__module__ == "ane.features"
    assert tuple(mod.ELEMENTS) == ("Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt")
    assert ELEMENT_PROPS["Ga"]["radius"] == 1.408


def test_pool_is_the_deposited_dataset(mod) -> None:
    full = mod.load_pool("all")
    measured = mod.load_pool("measured")
    assert len(full) == 70
    assert int((full.cycle_added == 0).sum()) == 45
    assert len(measured) == 57
    assert int((measured.cycle_added == 0).sum()) == 32
    ratio = (full.S_ANE.abs() / full.kxx).max()
    assert ratio == pytest.approx(POOL_OPTIMUM, rel=1e-12)


def test_optimum_survives_dropping_the_reconstructed_entries(mod) -> None:
    """Note S1 only means something if both pools share the same target."""
    measured = mod.load_pool("measured")
    assert (measured.S_ANE.abs() / measured.kxx).max() == pytest.approx(
        POOL_OPTIMUM, rel=1e-12
    )


@pytest.mark.parametrize("name", sorted(PUBLISHED_MEDIANS))
def test_deposited_results_have_the_expected_shape(name, results) -> None:
    df = results[name]
    assert set(df.strategy) == set(STRATEGIES)
    assert df.seed.nunique() == 50
    assert set(df.columns) == {"strategy", "seed", "n_exp", "best_ratio", "hv"}
    assert df.best_ratio.max() == pytest.approx(POOL_OPTIMUM, rel=1e-12)


@pytest.mark.parametrize("name", sorted(PUBLISHED_MEDIANS))
def test_published_acquisition_counts(name, results) -> None:
    assert _medians(results[name]) == pytest.approx(PUBLISHED_MEDIANS[name])


def test_published_reach_counts_under_randomized_starting_data(results) -> None:
    df = results["robust"]
    reached = {}
    for strategy in STRATEGIES:
        g = df[df.strategy == strategy]
        reached[strategy] = sum(
            1 for _, run in g.groupby("seed") if (run.best_ratio >= POOL_OPTIMUM - 1e-9).any()
        )
    assert reached == PUBLISHED_ROBUST_REACHED


def test_random_is_unaffected_by_the_reconstructed_entries(results) -> None:
    """Random does not use the surrogate, so the two pools must agree exactly."""
    a = _medians(results["batch1"])["random"]
    b = _medians(results["measured_only"])["random"]
    assert a == b


def test_summary_matches_the_deposited_results(summary, results) -> None:
    """The quoted table cannot drift away from the CSVs it summarizes."""
    for name, df in results.items():
        entry = summary["configurations"][name]
        assert entry["pool_optimum"] == pytest.approx(float(df.best_ratio.max()))
        for strategy, stats in entry["experiments_to_optimum"].items():
            assert stats["median"] == pytest.approx(_medians(df)[strategy])


def test_harness_runs_and_is_deterministic(mod) -> None:
    """One short campaign, twice, to catch a change in the replay itself."""
    df = mod.load_pool("all")
    X = mod.featurize(df[list(mod.ELEMENTS)].to_numpy(float))
    kappa = df["kxx"].to_numpy(float)
    s_ane = df["S_ANE"].to_numpy(float)
    mask = (df["cycle_added"] == 0).to_numpy()

    runs = [
        mod.run_campaign(X, kappa, s_ane, "gp_pareto_unc", mod.RNG_BASE, 3, 1, mask)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert [r["n_exp"] for r in runs[0]] == [45, 46, 47, 48]
    assert runs[0][-1]["best_ratio"] == pytest.approx(POOL_OPTIMUM, rel=1e-12)
