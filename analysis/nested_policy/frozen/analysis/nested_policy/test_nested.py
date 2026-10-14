import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('nested', Path(__file__).with_name('run_nested.py'))
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)

def test_adaptive_only_uses_inner_scores_and_availability():
    assert n.choose_branch(.2, .1) == 'wgan'
    assert n.choose_branch(.1, .2) == 'real_only'
    assert n.choose_branch(.1, .1) == 'real_only'
    assert n.choose_branch(.2, .1, False) == 'real_only'
    assert n.choose_branch(.2, float('nan')) == 'real_only'

def test_all_outer_samples_held_once_per_seed():
    data = pd.read_csv(n.ROOT / 'data/data.csv').query('cycle_added == 0')
    for seed in [42, 2025, 314159]:
        seen = []
        for train, held in n.KFold(5, shuffle=True, random_state=seed).split(data):
            assert not set(train).intersection(held)
            seen.extend(held)
        assert sorted(seen) == list(range(45))

def test_invalid_fom_not_silently_dropped():
    y = np.array([[10., 2.], [20., 4.]])
    p = np.array([[-1., 2.], [20., 4.]])
    result = n.metrics(y, p, y, np.array([True, True]))
    assert result['mae_fom_all'] is None
    assert result['invalid_prediction_count'] == 1

def test_measured_metric_separates_reconstructed_labels():
    y = np.array([[10., 2.], [20., 4.]])
    p = np.array([[10., 2.], [20., 8.]])
    result = n.metrics(y, p, y, np.array([True, False]))
    assert result['mae_fom_measured'] == 0
    assert result['mae_fom_all'] == pytest.approx(.1)

def test_inner_fold_generated_data_dispatch_and_scaling():
    df = pd.read_csv(n.ROOT / 'data/data.csv').head(8)
    comp, y = n.data_arrays(df)
    tr, va = np.arange(6), np.arange(6, 8)
    fake = df.iloc[:2].copy()
    folds = [{'train': tr, 'val': va, 'generated': fake}]
    with n.local_generated_folds(folds):
        x, yy, xv, yv, w = n.s._fold_arrays(comp, y, tr, va, None, None, .2)
        assert len(x) == 8
        np.testing.assert_allclose(w, [1, 1, 1, 1, 1, 1, .2, .2])
        expected = n.ORIGINAL_FOLD_ARRAYS(comp, y, tr, va, fake[n.C].to_numpy(), fake[n.Y].to_numpy(), .2)
        for got, want in zip((x, yy, xv, yv, w), expected):
            np.testing.assert_allclose(got, want)
    assert n.s._fold_arrays is n.ORIGINAL_FOLD_ARRAYS
    assert n.s.cv_objective_loss is n.ORIGINAL_OBJECTIVE

def test_heldout_targets_cannot_change_fitted_models_or_policy(monkeypatch, tmp_path):
    data = pd.read_csv(n.ROOT / 'data/data.csv').query('cycle_added == 0').reset_index(drop=True)
    tr, te = next(n.KFold(5, shuffle=True, random_state=42).split(data))
    seen = []
    def real(df, d, seed, smoke):
        seen.append(df[n.C + n.Y].to_numpy().copy())
        return {'directory': 'real', 'cv_score': .2, 'training_ids': df.sample_id.tolist()}
    def inner(df, d, seed, smoke):
        seen.append(df[n.C + n.Y].to_numpy().copy())
        return []
    def generate(df, reference, d, seed, smoke):
        seen.append(df[n.C + n.Y].to_numpy().copy())
        return df, {'success': True}
    def aug(df, real, folds, gen, status, d, seed, smoke):
        return {'directory': 'aug', 'cv_score': .1, 'inner_generation_successes': 5}
    monkeypatch.setattr(n, 'fit_real', real)
    monkeypatch.setattr(n, 'prepare_inner', inner)
    monkeypatch.setattr(n, 'generate', generate)
    monkeypatch.setattr(n, 'fit_wgan', aug)
    monkeypatch.setattr(n, 'predict', lambda fit, comp: (np.tile([15., 3.], (len(comp), 1)), 0))
    n.run_fold(data, tmp_path / 'first', 42, 1, tr, te, True)
    changed = data.copy()
    changed.loc[te, n.Y] *= 17
    n.run_fold(changed, tmp_path / 'second', 42, 1, tr, te, True)
    for first, second in zip(seen[:3], seen[3:]):
        np.testing.assert_array_equal(first, second)
    paths = [tmp_path / name / 'seed_42/fold_1/decision_before_outer_scoring.json' for name in ['first', 'second']]
    assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())

def test_smoke_budget_is_explicit_and_full_budget_unchanged(tmp_path):
    full, smoke = n.config(tmp_path, 42, False), n.config(tmp_path, 42, True)
    assert (full.optuna_trials, full.k_folds, full.epochs_tune, full.epochs_final) == (100, 5, 100, 200)
    assert smoke.optuna_trials == 1
