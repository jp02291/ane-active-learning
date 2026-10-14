"""Bounded retries for transient Windows locks on atomic JSON publication only."""
import time
from pathlib import Path


def install(roots, timeout=5.0):
    roots = tuple(Path(root).resolve() for root in roots)
    original = Path.replace

    def replace(source, target):
        destination = Path(target).absolute()
        scoped = (destination.suffix == '.json' and source.suffix == '.tmp'
                  and any(destination.is_relative_to(root) for root in roots))
        if not scoped:
            return original(source, target)
        deadline = time.monotonic() + timeout
        while True:
            try:
                return original(source, target)
            except PermissionError as exc:
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    Path.replace = replace
    return original
