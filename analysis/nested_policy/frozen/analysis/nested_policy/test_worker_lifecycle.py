from types import SimpleNamespace
import pytest
from worker_lifecycle import install, CheckpointRestart


def fake_audit():
    events = []
    def fit(df, directory, seed, smoke):
        (directory / 'fit_result.json').write_text('{}')
        return {'directory': str(directory)}
    return SimpleNamespace(event=lambda message, **kw: events.append(message), fit_real=fit), events


def test_restart_only_after_quota_saved_events():
    audit, events = fake_audit()
    install(audit, max_trials=2)
    audit.event('Training fold-local WGAN')
    audit.event('HPO trial finished')
    with pytest.raises(CheckpointRestart):
        audit.event('HPO trial finished')
    assert events[-1] == 'HPO trial finished'


def test_new_fit_checkpoint_survives_and_cached_fit_is_not_repeated(tmp_path):
    audit, _ = fake_audit()
    install(audit)
    with pytest.raises(CheckpointRestart):
        audit.fit_real(None, tmp_path, 42, False)
    assert (tmp_path / 'fit_result.json').exists()
    assert audit.fit_real(None, tmp_path, 42, False)['directory'] == str(tmp_path)


def test_completed_gan_requests_restart_but_start_does_not():
    audit, _ = fake_audit()
    install(audit)
    audit.event('Training fold-local WGAN')
    with pytest.raises(CheckpointRestart):
        audit.event('WGAN finished', success=False)
