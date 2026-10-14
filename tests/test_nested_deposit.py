"""Validate the deposited evaluation and reject incomplete or leaked inputs."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'analysis/nested_policy/summarize.py'
spec = importlib.util.spec_from_file_location('nested_deposit_summary', PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_deposited_results_recompute_from_preserved_predictions():
    summary.verify_hashes()
    tables, verification = summary.compute()
    summary.verify_report(tables, verification)
    assert verification['final_generation_successes'] == 12
    assert verification['prediction_rows'] == 270
    assert verification['FoM_measured_ties'] == 3


def test_generation_cannot_include_outer_test_rows(monkeypatch):
    original = summary.read

    def contaminated(path):
        result = original(path)
        path = Path(path)
        if path.name == 'generation_status.json' and path.parent.name == 'full_gan':
            membership = original(path.parent.parent / 'membership.json')
            result['training_ids'].append(membership['outer_test'][0])
        return result

    monkeypatch.setattr(summary, 'read', contaminated)
    with pytest.raises(AssertionError, match='Generation saw outer held-out rows'):
        summary.compute()


def test_missing_fold_is_not_silently_excluded(tmp_path):
    (tmp_path / 'protocol.json').write_bytes((summary.HERE / 'raw/protocol.json').read_bytes())
    with pytest.raises(AssertionError, match='Expected 15 complete folds'):
        summary.compute(tmp_path)


def test_r2_retains_negative_values():
    result = summary.metrics([{'true_sane': 1., 'pred_sane': 5.},
                              {'true_sane': 2., 'pred_sane': 5.}], 'sane')
    assert result['mae'] == 3.5
    assert result['r2'] == -49.
