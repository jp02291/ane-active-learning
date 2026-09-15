#!/usr/bin/env python
"""Partition the measured dataset into training and held-out sets (stage 0).

    python scripts/0_split_data.py --config configs/cycle3.yaml

Writes train.csv, test.csv and split_manifest.json under the configured output
directory. The partition is not random; see `ane.data` for what it does and why.

`data/data.csv` accumulates every measurement the campaign produced, so the
configuration must say which cycle is being reproduced. `split.up_to_cycle`
keeps only the rows that existed before that cycle began; the cycle
configurations set it, and stage 0 refuses to run without it on a file that
contains post-campaign rows.
"""

from __future__ import annotations

import argparse

from ane.config import PipelineConfig
from ane.data import make_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", default=None, help="override split.input_csv")
    ap.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="standardized reanalysis seed applied to every stochastic stage",
    )
    ap.add_argument(
        "--up-to-cycle",
        type=int,
        default=None,
        help="override split.up_to_cycle: keep only rows added before this cycle",
    )
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.master_seed is not None:
        cfg.apply_master_seed(args.master_seed)
    if args.input:
        cfg.split.input_csv = args.input
    if args.up_to_cycle is not None:
        cfg.split.up_to_cycle = args.up_to_cycle

    make_split(cfg.split)


if __name__ == "__main__":
    main()
