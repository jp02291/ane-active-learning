#!/usr/bin/env python
"""Enumerate, rank and export the next batch of candidates (stage 4).

    python scripts/4_select_candidates.py --config configs/cycle3.yaml

Writes the full candidate table, the three-objective Pareto front, and the two
sets of five compositions handed to synthesis. See `ane.select` for what the
objectives are and why uncertainty is one of them.
"""

from __future__ import annotations

import argparse

from ane.config import PipelineConfig
from ane.select import run_selection


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="standardized reanalysis seed applied to every stochastic stage",
    )
    ap.add_argument("--ensemble-dir", default=None, help="override selection.ensemble_dir")
    ap.add_argument("--output-dir", default=None, help="override selection.output_dir")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.master_seed is not None:
        cfg.apply_master_seed(args.master_seed)
    if args.ensemble_dir:
        cfg.selection.ensemble_dir = args.ensemble_dir
    if args.output_dir:
        cfg.selection.output_dir = args.output_dir

    run_selection(cfg.selection, verbose=not args.quiet)


if __name__ == "__main__":
    main()
