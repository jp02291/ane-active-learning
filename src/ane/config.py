"""Pipeline configuration.

Every value that was previously hard-coded inside a notebook `Config` class
lives here, in one dataclass tree that can be written to and read from YAML.
Each closed-loop cycle therefore leaves behind a file recording exactly what it
was run with, which is what makes a cycle reproducible after the fact.

Load a cycle configuration with

    cfg = PipelineConfig.from_yaml("configs/cycle3.yaml")

and write the defaults out with

    PipelineConfig().to_yaml("configs/default.yaml")

Values here are the ones reported in the supplementary material. Where the
supplementary text and the surviving notebook disagreed, the supplementary
values were adopted, and each such field carries a comment saying so. The run
logs that would settle those cases independently were not archived, so the test
suite pins both readings instead and neither can drift without a test failing:
`test_search_bounds_superseding_the_surviving_notebook` covers the DNN search
bounds, where the two sources differ, and
`test_gan_config_matches_the_notebook_everywhere` covers the GAN block, where
they agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitConfig:
    """Train-test partition.

    Compositions in the top `top_fraction` by |S_ANE| / kappa are placed in the
    training set, and the test set is drawn at random from the remainder.

    The retention is a deliberate choice, not an oversight: at each cycle only
    about ten high-performance compositions existed, and assigning any of them
    to the test set would have removed the target region from the training
    data. It is also what makes the partition target informed, so reported test
    metrics describe accuracy over the bulk of the composition space rather
    than within the high-performance region, and the manuscript states this.
    """

    input_csv: str = "data/data.csv"
    output_dir: str = "data/split"
    top_fraction: float = 0.15
    n_test: int = 10
    seed: int = 42

    #: The campaign membership.  When both paths are set, stage 0 validates and
    #: copies these rows instead of drawing a new partition.  All three cycles
    #: use it: cycle 1's membership was retained, and the cycle 2 and cycle 3
    #: memberships were recovered from the deposited Fig. 2 prediction records.
    #: A fresh draw would not reproduce any of them.
    fixed_train_csv: str | None = None
    fixed_test_csv: str | None = None

    #: Restrict the input to rows that existed before this cycle began, using
    #: the `cycle_added` column. `data/data.csv` accumulates every measurement
    #: the campaign produced, so splitting it whole reproduces no cycle: it
    #: trains cycle 1 on compositions cycle 1 was supposed to discover. Each
    #: `configs/cycleN.yaml` sets this. Leaving it unset is an error whenever
    #: the file contains rows added during the campaign; set it past the last
    #: cycle to use every row deliberately.
    up_to_cycle: int | None = None


@dataclass
class GANConfig:
    """WGAN-GP generation and filtering (Algorithm S2).

    The generator works in a nine-dimensional space -- seven isometric
    log-ratio coordinates and the two properties -- so that a sample is a
    composition and its properties together, and the properties are generated
    subject to whatever correlation with composition the critic has learned.

    Generated samples pass three filters before they are kept: the physical
    constraints, a range check against the measured properties, and a
    consistency check against a surrogate trained on measured data alone.
    Checkpoints are accepted only when a cheap proxy model improves on a
    cross-validation when the synthetic samples are added, which is what stops
    the run from keeping a generator that produces plausible-looking but
    useless data.

    These are the values of the campaign, corroborated three ways: the
    surviving GAN_v2.ipynb, Algorithm S2 of the supplementary material, and the
    saved generator checkpoint, whose first layer has shape (8, 64) and so
    settles `latent_dim` directly. An earlier transcription of this block
    carried `latent_dim` 16, `critic_iterations` 5, `epochs` 3000,
    `batch_size` 32, `proxy_hidden` (64, 64), `proxy_max_iter` 600,
    `fake_pool_min` 2000, `fake_select_k` 50, `surrogate_quantile` 0.95 and
    `surrogate_safety_factor` 1.5; the checkpoint ruled that out.

    Note that this is the opposite conclusion from the DNN search ranges in
    `SurrogateConfig`, where the notebook was the edited copy and the recorded
    optima proved it. Each block was settled on its own evidence rather than by
    trusting one source throughout.
    """

    train_csv: str = "data/split/train.csv"
    test_csv: str = "data/split/test.csv"
    output_dir: str = "artifacts"
    output_prefix: str = "artifacts/augmented_data_n"
    generated_sizes: tuple[int, ...] = (100, 200, 300, 400, 500)

    latent_dim: int = 8
    critic_iterations: int = 4
    gradient_penalty_weight: float = 10.0
    epochs: int = 10000
    batch_size: int = 16
    seed: int = 42

    learning_rate: float = 1e-4
    # beta_1 = 0 is the WGAN-GP recommendation: momentum on the critic makes
    # the gradient penalty fight the optimizer's history rather than the
    # current discriminator.
    adam_beta_1: float = 0.0
    adam_beta_2: float = 0.9

    # Noise added to both real and fake samples before the critic sees them,
    # annealed linearly to zero over the run. With a few dozen real rows the
    # critic otherwise memorizes them within a few hundred steps.
    instance_noise_std: float = 0.01

    # Checkpoints are evaluated, not saved blindly: every `eval_interval`
    # epochs the proxy comparison runs, and the generator is kept only if it
    # improves on the previous best by more than `improvement_tolerance`.
    eval_interval: int = 100
    early_stopping_patience: int = 50
    improvement_tolerance: float = 1e-4

    # -- physical constraints on generated compositions ---------------------
    #
    # Repeated from SelectionConfig on purpose: these bound what may be
    # generated, those bound what may be proposed for synthesis, and a cycle
    # could legitimately want them different.
    stoichiometry_min: float = 2.2
    stoichiometry_max: float = 3.8
    h_mix_max: float = 0.0
    mn_max: float = 0.25
    co_max: float = 0.60

    # Generated compositions are snapped to a grid and then rejected if the
    # snap moved them further than `snap_distance_max`, so that a sample is
    # kept only when the generator already produced something close to a
    # realizable composition rather than being rounded into one.
    #
    # 0.01 is the same grid the selection stage enumerates on. This field was
    # briefly recorded as 0.005 while the selection grid was recorded as
    # 0.01 -- the two had been transposed.
    snap_step: float = 0.01
    snap_distance_max: float = 0.015

    # Generated properties must lie within the measured range, widened by
    # `property_margin` of that range at each end. Extrapolating a little is
    # the point; extrapolating far is the generator inventing physics.
    property_quantile_low: float = 0.01
    property_quantile_high: float = 0.99
    property_margin: float = 0.15

    # -- surrogate-consistency filter ---------------------------------------
    #
    # The surrogate is the real-only model from stage 2, deliberately not one
    # trained on augmented data: filtering generated samples with a model that
    # learned from generated samples would be circular.
    use_surrogate_filter: bool = True
    surrogate_model_dir: str = "artifacts/dnn_base"
    surrogate_validation_csv: str | None = None
    surrogate_quantile: float = 0.99
    surrogate_safety_factor: float = 2.0
    surrogate_calibration_fraction: float = 0.20
    # The filter is skipped for the first `surrogate_warmup_epoch` epochs: an
    # untrained generator produces nothing that would pass, and applying it
    # early only wastes draws.
    surrogate_warmup_epoch: int = 1000

    # -- proxy evaluation used for checkpoint acceptance --------------------
    proxy_kfold_splits: int = 5
    proxy_hidden: tuple[int, ...] = (16, 16)
    proxy_max_iter: int = 1000
    fake_pool_min: int = 100
    fake_select_k: int = 20
    fake_draw_batch: int = 2000
    fake_max_rounds: int = 60
    # Relative weight of the property coordinates when selecting a diverse
    # subset; 1.0 means composition and properties count equally.
    kcenter_property_weight: float = 1.0

    # -- final pool ---------------------------------------------------------
    target_unique_samples: int = 800
    final_draw_per_round: int = 10000
    final_max_rounds: int = 100
    pool_seed: int = 2027
    # Written into the emitted CSV for reference. It is not what the surrogate
    # stage uses -- that weight is tuned per scenario as `w_gen` -- and is kept
    # only because the released augmented files carry the column.
    generated_sample_weight: float = 0.2

    # -- rare-element oversampling ------------------------------------------
    #
    # Real rows containing an element that appears in only a few of them are
    # drawn more often, so the critic sees the sparse corners of the space at
    # all. Without it the generator collapses onto the Fe-Ga majority.
    rare_min_count: int = 2
    rare_max_ratio: float = 0.25
    weight_boost_max: float = 5.0
    weight_cap: float = 9.0


@dataclass
class SurrogateConfig:
    """DNN hyperparameter search and ensemble construction (Algorithms S3, S4).

    The search ranges below are those used throughout the campaign. The range
    for `neurons_max` and `w_gen_max` was narrowed late in the third cycle after
    the objective had visibly saturated; the optima found before and after the
    change were close enough that a single range is reported. If a cycle needs
    different ranges, give it its own configuration file rather than editing
    these defaults, so that each cycle keeps a record of what it was run with.

    Two of those ranges differ from the copy of DNN_GAN.ipynb that survives,
    which searches `num_neurons` over 16-128 and `w_gen` over 0.01-0.30. Those
    narrower bounds are a later edit made while testing after the campaign had
    finished; the values here are the ones the reported results came from. The
    surviving notebook therefore does not reproduce them, and this package is
    what should be released.

    `surrogate.check_best_params` cross-checks a recorded `best_params.json`
    against these bounds, so the claim can be confirmed rather than trusted.
    """

    train_csv: str = "data/split/train.csv"
    test_csv: str = "data/split/test.csv"
    generated_csv: str | None = "artifacts/augmented_data_n200.csv"
    artifact_dir: str = "artifacts/dnn_gan_n200"

    optuna_trials: int = 100
    k_folds: int = 5
    seed: int = 42

    # `num_neurons`, `learning_rate`, `l2` and `w_gen` are sampled
    # log-uniformly; `num_layers` and `dropout` uniformly. That is a property of
    # the search, not a tunable, so it is not exposed as a field.
    layers_min: int = 1
    layers_max: int = 8
    neurons_min: int = 16
    neurons_max: int = 256
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 3e-3
    l2_min: float = 1e-6
    l2_max: float = 1e-3
    dropout_min: float = 0.0
    dropout_max: float = 0.2
    batch_sizes: tuple[int, ...] = (4, 8, 16, 32)
    w_gen_min: float = 0.01
    w_gen_max: float = 0.70

    # `w_gen` is only sampled when a generated set is configured; with real
    # data alone the weight is fixed at 1.0 and does not enter the search.
    w_gen_without_generated: float = 1.0

    tune_l2: bool = True
    l2_fixed: float = 1e-4

    # Epoch budgets. The search runs short (`epochs_tune`) because it is
    # comparing architectures, not fitting the final model; the K-fold pass
    # afterwards runs longer and is what the final epoch count comes from.
    epochs_tune: int = 100
    epochs_final: int = 200

    # Patience during the search. Short on purpose: a trial that has not
    # improved in ten epochs is unlikely to win, and the search runs a hundred
    # of them times five folds. The ensemble stage uses its own, much longer
    # patience (`ensemble_early_stopping_patience`); these were a single field
    # until the two notebooks turned out to disagree, and both values are
    # confirmed against their respective notebooks.
    early_stopping_patience: int = 10
    tune_lr_patience: int = 5

    # Patience during the final K-fold pass, which is measuring how long to
    # train rather than screening architectures, so it is allowed to run out.
    final_early_stopping_patience: int = 80
    final_lr_patience: int = 15

    lr_factor: float = 0.8
    lr_min: float = 1e-6

    # -- deep ensemble construction (Algorithm S4) --------------------------
    #
    # Member seeds are `ensemble_seed_base + member_id`, not `seed`. `seed`
    # governs the Optuna study and the data split; the ensemble deliberately
    # uses a different base so that member 0 is not a repeat of the tuning run.
    ensemble_seed_base: int = 2025
    ensemble_size_raw: int = 60
    ensemble_size_max_kept: int = 30
    ensemble_size_min_kept: int = 20

    ensemble_epochs: int = 300
    ensemble_val_fraction: float = 0.20
    ensemble_early_stopping_patience: int = 50
    ensemble_lr_patience: int = 15
    ensemble_lr_factor: float = 0.8

    # Member pruning. A member is dropped when its validation MAE is a robust
    # outlier among the 60, or when it predicts a physically implausible
    # kappa anywhere on its own validation split. The threshold is
    # median + mad_factor * MAD, falling back to Q3 + iqr_factor * IQR when the
    # MAD is degenerate -- a plain mean and standard deviation would be dragged
    # up by the very members being screened out.
    prune_mad_factor: float = 2.5
    prune_iqr_factor: float = 1.5
    prune_targetwise_mae: bool = True
    prune_kappa_floor: bool = True

    # The floor is data-derived unless pinned: max(quantile(train kappa, q),
    # kappa_floor_min). Set `kappa_member_floor` to override.
    kappa_member_floor: float | None = None
    kappa_floor_quantile: float = 0.05
    kappa_floor_min: float = 0.5

    # If the robust filters leave fewer than `ensemble_size_min_kept` members,
    # fall back to that many best by validation MAE and record the fallback in
    # the manifest. The ensemble spread is an objective downstream, so a thin
    # ensemble changes the selection rather than merely adding noise.
    allow_fallback_to_min_models: bool = True


@dataclass
class SelectionConfig:
    """Candidate enumeration and Pareto-based selection (Algorithm S5)."""

    ensemble_dir: str = "artifacts/dnn_gan_n200/ensemble_trained"
    output_dir: str = "artifacts/selection"
    output_prefix: str = "pareto_3objective"

    grid_step: float = 0.01
    generation_seed: int = 42
    limit_per_case: int = 1000
    n_group_a: tuple[int, ...] = (1, 2, 3)
    n_group_b: tuple[int, ...] = (1, 2, 3, 4, 5)

    stoichiometry_min: float = 2.2
    stoichiometry_max: float = 3.8
    h_mix_max: float = 0.0
    element_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"Co": (0.00, 0.60), "Mn": (0.00, 0.25)}
    )

    top_k: int = 5                      # per group: exploitation and exploration
    diversity_distance: float = 0.10
    require_positive_s_ane: bool = True
    require_positive_kappa: bool = True

    # How 1/kappa is formed at the ensemble-member level, before averaging.
    # "add_eps"       -- 1 / (kappa + eps), the convention used for every
    #                    reported cycle; a member predicting a small negative
    #                    kappa yields a large negative inverse, which pushes the
    #                    member spread up and marks the candidate as uncertain.
    # "clip_positive" -- 1 / clip(kappa, eps, None), which instead saturates
    #                    those members at 1/eps.
    # The two disagree only where members disagree about the sign of kappa, but
    # they disagree there by a lot, and the spread is an objective. Changing
    # this changes which candidates are selected: keep "add_eps" unless the
    # change is deliberate and recorded in the cycle configuration.
    kappa_inverse_mode: str = "add_eps"
    kappa_inverse_epsilon: float = 1e-8
    predict_batch_size: int = 8192
    min_ensemble_models: int = 20


@dataclass
class PipelineConfig:
    """Full configuration for one active-learning cycle."""

    cycle: int = 1

    #: Optional standardized reanalysis seed.  Historical cycle files omit
    #: this field because they retain the component seeds recorded for the
    #: campaign.  Setting it (or using ``--master-seed`` in a stage script)
    #: applies the same stochastic root to the split, GAN, final GAN pool,
    #: Optuna/CV, ensemble-member seed base, and candidate enumeration.
    #:
    #: Ensemble members still use ``master_seed + member_id``.  Giving every
    #: member the identical literal seed would collapse the intended ensemble
    #: diversity and is therefore deliberately not supported.
    master_seed: int | None = None

    #: Root under which every artifact of this cycle is written. Each cycle
    #: gets its own, so that stage 1 of cycle 3 cannot overwrite the generator
    #: or the base surrogate that cycle 2 was built on, and so that the
    #: provenance of a directory is readable from its path alone.
    artifact_root: str = "artifacts"

    split: SplitConfig = field(default_factory=SplitConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls(
            cycle=raw.get("cycle", 1),
            master_seed=raw.get("master_seed"),
            artifact_root=raw.get("artifact_root", "artifacts"),
            split=SplitConfig(**raw.get("split", {})),
            gan=GANConfig(**raw.get("gan", {})),
            surrogate=SurrogateConfig(**raw.get("surrogate", {})),
            selection=SelectionConfig(**raw.get("selection", {})),
        )
        if cfg.master_seed is not None:
            cfg.apply_master_seed(cfg.master_seed)
        return cfg

    def apply_master_seed(self, seed: int = 42) -> None:
        """Apply one reproducible stochastic root to every pipeline stage.

        This method is for a standardized reanalysis.  It does not imply that
        every random draw or every ensemble member receives the identical
        integer.  Component roots are fixed to ``seed`` and ensemble member
        ``m`` is deterministically assigned ``seed + m``.
        """
        seed = int(seed)
        if seed < 0:
            raise ValueError("master seed must be a non-negative integer")

        self.master_seed = seed
        self.split.seed = seed
        self.gan.seed = seed
        self.gan.pool_seed = seed
        self.surrogate.seed = seed
        self.surrogate.ensemble_seed_base = seed
        self.selection.generation_seed = seed

    def seed_manifest(self) -> dict[str, Any]:
        """Return the effective seed policy for provenance records."""
        return {
            "master_seed": self.master_seed,
            "split_seed": self.split.seed,
            "gan_seed": self.gan.seed,
            "gan_pool_seed": self.gan.pool_seed,
            "surrogate_seed": self.surrogate.seed,
            "ensemble_seed_base": self.surrogate.ensemble_seed_base,
            "ensemble_member_rule": "ensemble_seed_base + member_id",
            "selection_generation_seed": self.selection.generation_seed,
        }

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.as_dict(), fh, sort_keys=False, allow_unicode=True)

    def as_dict(self) -> dict[str, Any]:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            return obj

        return clean(asdict(self))
