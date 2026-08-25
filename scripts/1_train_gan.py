#!/usr/bin/env python
"""Train the WGAN-GP and emit the augmented datasets (stage 1).

    python scripts/1_train_gan.py --config configs/cycle3.yaml

Requires the real-only surrogate from stage 2 of the *same* cycle -- run
`2_tune_hyperparams.py --scenario base` first. The consistency filter uses it
to reject implausible samples, so it must be trained on the data this cycle
had, not on an earlier cycle's. Pass `--no-filter` to
run without it -- the generated data is then unfiltered and should not be used
for a reported cycle.

The run fails rather than emitting data if no checkpoint improved the proxy
comparison. That is intentional: a generator that never helped has not earned
a place in the training set.
"""

from __future__ import annotations

import argparse

from ane.augment import train_gan
from ane.config import PipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="standardized reanalysis seed applied to every stochastic stage",
    )
    ap.add_argument("--surrogate-dir", default=None, help="override gan.surrogate_model_dir")
    ap.add_argument(
        "--no-filter",
        action="store_true",
        help="skip the surrogate-consistency filter (not for reported runs)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.master_seed is not None:
        cfg.apply_master_seed(args.master_seed)
    if args.surrogate_dir:
        cfg.gan.surrogate_model_dir = args.surrogate_dir
    if args.no_filter:
        cfg.gan.use_surrogate_filter = False

    train_gan(cfg.gan, verbose=not args.quiet)


if __name__ == "__main__":
    main()
