#!/usr/bin/env python
"""Optuna search and final refit, per augmentation scenario (stage 2).

    python scripts/2_tune_hyperparams.py --config configs/cycle3.yaml

Runs one search per scenario -- each generated set plus a real-only control --
and writes `best_params.json`, `final_best_model.h5` and `metrics_test.json`
into that scenario's artifact directory, then ranks the scenarios in
`scenario_summary.csv`, ranked on `optuna_cv_score` -- the five-fold
cross-validation score. The scenario at the top of that ranking is the one
whose `best_params.json` stage 3 should read.

Long: a hundred trials over five folds, for six scenarios. Use `--scenario` to
run one at a time.
"""

from __future__ import annotations

import argparse

from ane.config import PipelineConfig
from ane.surrogate import run_scenarios, scenario_configs, tune_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="standardized reanalysis seed applied to every stochastic stage",
    )
    ap.add_argument(
        "--scenario",
        default=None,
        help="run one scenario only: a generated-set size, or 'base' for real data alone",
    )
    ap.add_argument("--trials", type=int, default=None, help="override surrogate.optuna_trials")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.master_seed is not None:
        cfg.apply_master_seed(args.master_seed)
    if args.trials is not None:
        cfg.surrogate.optuna_trials = args.trials

    verbose = not args.quiet

    if args.scenario is None:
        run_scenarios(
            scenario_configs(cfg.surrogate, cfg.gan.generated_sizes, cfg.artifact_root),
            summary_path=f"{cfg.artifact_root}/scenario_summary.csv",
            verbose=verbose,
        )
        return

    if args.scenario == "base":
        cfg.surrogate.generated_csv = None
        cfg.surrogate.artifact_dir = f"{cfg.artifact_root}/dnn_base"
    else:
        n = int(args.scenario)
        cfg.surrogate.generated_csv = f"{cfg.artifact_root}/augmented_data_n{n}.csv"
        cfg.surrogate.artifact_dir = f"{cfg.artifact_root}/dnn_gan_n{n}"

    tune_scenario(cfg.surrogate, verbose=verbose)


if __name__ == "__main__":
    main()
