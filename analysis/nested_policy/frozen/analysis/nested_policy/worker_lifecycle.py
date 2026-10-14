"""Bound process lifetime at saved checkpoints without reducing learning budgets."""
from pathlib import Path
import functools

RESTART_EXIT_CODE = 75


class CheckpointRestart(BaseException):
    pass


def install(audit, max_trials=10):
    if max_trials < 1:
        raise ValueError('max_trials must be positive')
    original_event = audit.event
    original_fit_real = audit.fit_real
    completed_here = 0

    def event(message, **kwargs):
        nonlocal completed_here
        original_event(message, **kwargs)
        # run_search has already committed the SQLite result AND sampler.joblib
        # before this event. A planned restart never leaves a RUNNING trial.
        if message == 'HPO trial finished':
            completed_here += 1
            if completed_here >= max_trials:
                raise CheckpointRestart(f'{completed_here} completed trials saved with sampler state')
        # generate writes generation_status.json before this event, including
        # accepted rejection outcomes, so no completed GAN is retrained.
        if message == 'WGAN finished':
            raise CheckpointRestart('Completed GAN result saved')

    @functools.wraps(original_fit_real)
    def fit_real(df, directory, seed, smoke):
        marker = Path(directory) / 'fit_result.json'
        was_complete = marker.exists()
        result = original_fit_real(df, directory, seed, smoke)
        if not was_complete:
            if not marker.exists():
                raise RuntimeError('Refit returned without its completion checkpoint')
            raise CheckpointRestart('New reference model and fit_result.json saved')
        return result

    audit.event = event
    audit.fit_real = fit_real
