#!/usr/bin/env python
"""Train the deep ensemble and prune it (stage 3).

    python scripts/3_train_ensemble.py --config configs/cycle3.yaml

Reads `best_params.json` from the scenario artifact directory, trains
`surrogate.ensemble_size_raw` members, keeps the ones that are not outliers,
and writes them where stage 4 expects to find them. See `ane.surrogate` for
what the pruning criterion is and why the members are pruned at all.
"""

from __future__ import annotations

import argparse

from ane.config import PipelineConfig
from ane.surrogate import train_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="standardized reanalysis seed; member m uses seed + m",
    )
    ap.add_argument("--artifact-dir", default=None, help="override surrogate.artifact_dir")
    ap.add_argument(
        "--real-only",
        action="store_true",
        help="ignore surrogate.generated_csv and train on measured data alone",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.master_seed is not None:
        cfg.apply_master_seed(args.master_seed)
    if args.artifact_dir:
        cfg.surrogate.artifact_dir = args.artifact_dir
    if args.real_only:
        cfg.surrogate.generated_csv = None

    train_ensemble(cfg.surrogate, verbose=not args.quiet)


if __name__ == "__main__":
    main()
