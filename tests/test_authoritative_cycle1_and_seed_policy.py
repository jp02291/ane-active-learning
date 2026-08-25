"""Regression checks for the remediated cycle-1 data and seed-42 reanalysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ane.config import PipelineConfig
from ane.data import load_dataset, make_split
from ane.elements import ELEMENTS


ROOT = Path(__file__).resolve().parents[1]


def _keys(df: pd.DataFrame) -> set[tuple[float, ...]]:
    comp = df[list(ELEMENTS)].to_numpy(float)
    comp = comp / comp.sum(axis=1, keepdims=True)
    return {tuple(np.round(row, 10)) for row in comp}


def test_authoritative_cycle1_table_is_exactly_the_initial_dataset() -> None:
    full = pd.read_csv(ROOT / "data" / "data.csv")
    initial = full.loc[full["cycle_added"] == 0].reset_index(drop=True)
    authority = pd.read_csv(ROOT / "data" / "authoritative_cycle1_45.csv")

    assert len(initial) == len(authority) == 45
    assert initial["sample_id"].is_unique
    assert initial["sample_id"].tolist() == authority["sample_id"].tolist()
    assert np.allclose(initial[list(ELEMENTS)].sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(
        initial[list(ELEMENTS) + ["kxx", "S_ANE"]].to_numpy(float),
        authority[list(ELEMENTS) + ["kxx", "S_ANE"]].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )


def test_cycle1_uses_the_recovered_36_9_membership(tmp_path: Path) -> None:
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "cycle1.yaml")
    cfg.split.input_csv = str(ROOT / cfg.split.input_csv)
    cfg.split.fixed_train_csv = str(ROOT / cfg.split.fixed_train_csv)
    cfg.split.fixed_test_csv = str(ROOT / cfg.split.fixed_test_csv)
    cfg.split.output_dir = str(tmp_path)

    full = load_dataset(cfg.split.input_csv, up_to_cycle=1)
    train, test = make_split(cfg.split, verbose=False)

    assert (len(train), len(test)) == (36, 9)
    assert not (_keys(train) & _keys(test))
    assert _keys(train) | _keys(test) == _keys(full)


def test_master_seed_42_overrides_every_stochastic_root() -> None:
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "cycle1.yaml")
    assert cfg.gan.pool_seed == 2027
    assert cfg.surrogate.ensemble_seed_base == 2025

    cfg.apply_master_seed(42)
    manifest = cfg.seed_manifest()

    assert manifest["master_seed"] == 42
    assert manifest["split_seed"] == 42
    assert manifest["gan_seed"] == 42
    assert manifest["gan_pool_seed"] == 42
    assert manifest["surrogate_seed"] == 42
    assert manifest["ensemble_seed_base"] == 42
    assert manifest["selection_generation_seed"] == 42
    assert manifest["ensemble_member_rule"] == "ensemble_seed_base + member_id"


def test_master_seed_does_not_collapse_ensemble_members() -> None:
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "cycle1.yaml")
    cfg.apply_master_seed(42)
    member_seeds = [cfg.surrogate.ensemble_seed_base + i for i in range(60)]
    assert member_seeds[0] == 42
    assert member_seeds[-1] == 101
    assert len(set(member_seeds)) == 60
