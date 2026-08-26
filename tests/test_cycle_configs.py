"""The cycle configurations must reproduce the campaign's data cutoffs.

`data/data.csv` accumulates every measurement the campaign made. Splitting it
whole trains each cycle on compositions that cycle had not yet measured, which
is the one reproducibility failure a reader cannot detect from the outputs.
These tests pin the row counts and held-out sizes the manuscript reports, so
that an edit to the configuration or to the dataset cannot quietly undo it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ane.config import PipelineConfig
from ane.data import load_dataset, make_split

ROOT = Path(__file__).resolve().parents[1]

#: cycle -> (compositions available, training rows, held-out rows)
CAMPAIGN = {1: (45, 36, 9), 2: (55, 45, 10), 3: (65, 55, 10)}


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_cycle_config_declares_its_cutoff(cycle: int) -> None:
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    assert cfg.cycle == cycle
    assert cfg.split.up_to_cycle == cycle, (
        f"configs/cycle{cycle}.yaml must set split.up_to_cycle, or stage 0 "
        f"splits all seventy compositions"
    )


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_cycle_cutoff_selects_the_data_that_existed(cycle: int) -> None:
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    df = load_dataset(ROOT / cfg.split.input_csv, up_to_cycle=cfg.split.up_to_cycle)
    assert len(df) == CAMPAIGN[cycle][0]


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_split_sizes_match_the_manuscript(cycle: int, tmp_path: Path) -> None:
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    cfg.split.input_csv = str(ROOT / cfg.split.input_csv)
    cfg.split.output_dir = str(tmp_path)
    train, test = make_split(cfg.split, verbose=False)

    _, n_train, n_test = CAMPAIGN[cycle]
    assert (len(train), len(test)) == (n_train, n_test)

    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["up_to_cycle"] == cycle
    # All three memberships are pinned. Cycle 1's was retained; cycles 2 and 3
    # were recovered from the Fig. 2 prediction records and pinned so that
    # stage 0 copies them rather than drawing a partition that would not be the
    # one behind the reported numbers. See tests/test_recovered_splits.py.
    assert manifest["split_source"] == "fixed_reported_membership"
    assert len(manifest["train"]) == n_train
    assert len(manifest["test"]) == n_test
    assert not set(manifest["train"]) & set(manifest["test"])


def test_splitting_the_whole_file_is_refused(tmp_path: Path) -> None:
    """Without a cutoff, stage 0 must fail rather than split all seventy rows."""
    cfg = PipelineConfig.from_yaml(ROOT / "configs/default.yaml")
    cfg.split.input_csv = str(ROOT / cfg.split.input_csv)
    cfg.split.output_dir = str(tmp_path)
    assert cfg.split.up_to_cycle is None
    with pytest.raises(ValueError, match="up_to_cycle"):
        make_split(cfg.split, verbose=False)


# ---------------------------------------------------------------------------
# Artifact scoping and the selected scenario
# ---------------------------------------------------------------------------

#: cycle -> the scenario directory the campaign carried into stages 3 and 4
SELECTED = {1: "dnn_gan_n200", 2: "dnn_base", 3: "dnn_gan_n200"}


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_every_path_is_scoped_to_its_cycle(cycle: int) -> None:
    """No cycle may read or write inside another cycle's directories.

    Sharing `artifacts/` meant stage 1 of a later cycle overwrote the generator
    and the base surrogate an earlier one was built on, and the provenance of a
    directory could not be read from its path.
    """
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    root = f"artifacts/cycle{cycle}"
    assert cfg.artifact_root == root
    assert cfg.split.output_dir == f"data/split/cycle{cycle}"

    scoped = [
        cfg.gan.output_dir, cfg.gan.output_prefix, cfg.gan.surrogate_model_dir,
        cfg.surrogate.artifact_dir, cfg.selection.ensemble_dir, cfg.selection.output_dir,
    ]
    if cfg.surrogate.generated_csv is not None:
        scoped.append(cfg.surrogate.generated_csv)
    for path in scoped:
        assert path.startswith(root), f"{path} is not under {root}"

    for path in (cfg.gan.train_csv, cfg.gan.test_csv,
                 cfg.surrogate.train_csv, cfg.surrogate.test_csv):
        assert path.startswith(cfg.split.output_dir), f"{path} is not this cycle's split"


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_selected_scenario_is_the_one_the_campaign_used(cycle: int) -> None:
    """Stages 3 and 4 must build the ensemble the campaign selected.

    Cycle 2 retained the real-only branch (Supplementary Table S5), so a
    configuration that still pointed at an augmented scenario would train the
    wrong model while the comment claimed otherwise.
    """
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    selected = SELECTED[cycle]
    assert cfg.surrogate.artifact_dir.endswith(selected)
    assert cfg.selection.ensemble_dir == f"{cfg.surrogate.artifact_dir}/ensemble_trained"

    if selected == "dnn_base":
        assert cfg.surrogate.generated_csv is None, (
            "the real-only branch must not be given a synthetic dataset"
        )
    else:
        assert cfg.surrogate.generated_csv is not None
        n = selected.rsplit("n", 1)[-1]
        assert cfg.surrogate.generated_csv.endswith(f"augmented_data_n{n}.csv")
        assert int(n) in cfg.gan.generated_sizes


@pytest.mark.parametrize("cycle", sorted(CAMPAIGN))
def test_gan_filters_against_this_cycles_reference_surrogate(cycle: int) -> None:
    """f_ref is the current cycle's real-only surrogate, not a previous one."""
    cfg = PipelineConfig.from_yaml(ROOT / f"configs/cycle{cycle}.yaml")
    assert cfg.gan.surrogate_model_dir == f"artifacts/cycle{cycle}/dnn_base"


def test_package_and_citation_versions_agree() -> None:
    """A repository whose two version strings disagree cannot be cited cleanly."""
    import re

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    pkg = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    cff = re.search(r"^version:\s*(\S+)", citation, re.M)
    assert pkg and cff, "both files must declare a version"
    assert pkg.group(1) == cff.group(1), (
        f"pyproject.toml says {pkg.group(1)}, CITATION.cff says {cff.group(1)}"
    )


def test_stage1_docs_say_same_cycle_not_previous() -> None:
    """The GAN filters against this cycle's reference surrogate.

    `gan.surrogate_model_dir` is scoped to the running cycle, so any prose
    saying stage 1 needs the previous cycle's model contradicts the code.
    """
    for rel in ("scripts/1_train_gan.py", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "previous cycle" not in text, (
            f"{rel} still describes the reference surrogate as coming from the "
            f"previous cycle; it comes from the same cycle"
        )


def test_lock_file_pins_exactly_or_says_nothing() -> None:
    """A lock file may be incomplete, but every line it does declare must be exact.

    `>=` in a lock file defeats its purpose: it looks like a record and behaves
    like a range. Anything not known is left commented out instead.
    """
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    pins = [l.strip() for l in lock.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert pins, "the lock file should pin at least what is known"
    for line in pins:
        assert "==" in line, f"{line!r} is not an exact pin"
        for loose in (">=", "<=", "~=", ">", "<"):
            assert loose not in line, f"{line!r} uses a range in a lock file"
    assert any(l.startswith("keras==") for l in pins), (
        "the Keras version is recorded in the surviving model file and should be pinned"
    )
