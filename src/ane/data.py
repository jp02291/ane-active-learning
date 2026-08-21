"""Dataset loading, validation, and the train-test partition.

The partition deserves a note, because it is not a random hold-out and the
distinction matters when reading the reported test metrics.

Compositions in the top 15% by |S_ANE| / kappa are assigned to the training
set. The test set is then drawn from the remaining 85% by stratified sampling
over k-means clusters fitted to the standardized composition and property
vectors, so that it spans the same regions of the space rather than clumping.

The reason is arithmetic. At each cycle roughly ten high-performance
compositions existed in total; placing even two of them in the test set would
have left the surrogate with almost nothing to learn from in precisely the
region the search was aimed at. The cost is that the test set does not measure
accuracy where the campaign cares most, and the manuscript says so.

Because the clustering uses property values, the partition is informed by the
targets. No target value from a test sample reaches training -- the split only
decides membership -- but the test set is constructed to be representative
rather than drawn blind, and that should be stated when the metrics are quoted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SplitConfig
from .elements import ELEMENTS, GROUP_A, GROUP_B

__all__ = [
    "PROPERTY_COLUMNS",
    "load_dataset",
    "make_split",
    "figure_of_merit",
    "stoichiometry_ratio",
]

#: Target columns, in the order produced by the surrogate.
PROPERTY_COLUMNS: tuple[str, str] = ("kxx", "S_ANE")


def _normalize_header(s: object) -> object:
    """Fold the dash variants and collapse whitespace in a column name.

    Spreadsheets exported from different tools render minus signs as U+2212,
    U+2013 or U+2014, which makes an otherwise identical header fail to match.
    """
    if pd.isna(s):
        return s
    s = str(s).replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s.strip())


def load_dataset(
    path: str | Path,
    require_properties: bool = True,
    up_to_cycle: int | None = None,
) -> pd.DataFrame:
    """Read a composition-property table and check that it is usable.

    Validates presence of the element and property columns, that compositions
    close to one, and that no property is missing. Failing here is far cheaper
    than discovering a malformed row after a generation run.

    `up_to_cycle` restricts the table to rows measured before that cycle began,
    using the `cycle_added` column. Reproducing cycle 2 means training on what
    was known at the time, not on everything measured since; passing
    `up_to_cycle=2` gives the dataset as it stood when cycle 2 started. Without
    it the whole file is returned, which is what a fresh cycle wants.
    """
    df = pd.read_csv(path)

    if up_to_cycle is not None:
        if "cycle_added" not in df.columns:
            raise ValueError(
                f"{path}: up_to_cycle needs a 'cycle_added' column recording "
                f"which cycle each row entered the dataset in"
            )
        df = df[df["cycle_added"] < up_to_cycle].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"{path}: no rows predate cycle {up_to_cycle}")

    df.columns = [_normalize_header(c) for c in df.columns]

    missing = [c for c in ELEMENTS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing element columns {missing}")

    if require_properties:
        missing_props = [c for c in PROPERTY_COLUMNS if c not in df.columns]
        if missing_props:
            raise ValueError(f"{path}: missing property columns {missing_props}")
        if df[list(PROPERTY_COLUMNS)].isna().any().any():
            bad = df.index[df[list(PROPERTY_COLUMNS)].isna().any(axis=1)].tolist()
            raise ValueError(f"{path}: missing property values in rows {bad}")

    comp = df[list(ELEMENTS)].fillna(0.0).to_numpy(float)
    if (comp < 0).any():
        raise ValueError(f"{path}: negative atomic fractions present")
    off = np.abs(comp.sum(axis=1) - 1.0)
    if (off > 5e-3).any():
        rows = np.where(off > 5e-3)[0].tolist()
        raise ValueError(f"{path}: compositions do not sum to one in rows {rows}")

    return df


def figure_of_merit(df: pd.DataFrame) -> np.ndarray:
    """|S_ANE| / kappa, the scalar the campaign maximizes."""
    return np.abs(df["S_ANE"].to_numpy(float)) / (df["kxx"].to_numpy(float) + 1e-12)


def stoichiometry_ratio(df: pd.DataFrame) -> np.ndarray:
    """x_A / x_B, the substitution ratio constrained during enumeration."""
    a = df[list(GROUP_A)].fillna(0.0).sum(axis=1).to_numpy(float)
    b = df[list(GROUP_B)].fillna(0.0).sum(axis=1).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.inf)


def make_split(
    cfg: SplitConfig, df: pd.DataFrame | None = None, verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the dataset and write ``train.csv`` and ``test.csv``.

    Returns the two frames. The top `top_fraction` by figure of merit is held
    in the training set by construction; see the module docstring.
    """
    if df is None:
        df = load_dataset(cfg.input_csv, up_to_cycle=cfg.up_to_cycle)

    if (cfg.fixed_train_csv is None) != (cfg.fixed_test_csv is None):
        raise ValueError(
            "split.fixed_train_csv and split.fixed_test_csv must either both "
            "be set or both be omitted"
        )

    if cfg.fixed_train_csv is not None and cfg.fixed_test_csv is not None:
        train = load_dataset(cfg.fixed_train_csv)
        test = load_dataset(cfg.fixed_test_csv)
        _validate_fixed_split(df, train, test, cfg)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / "train.csv", index=False)
        test.to_csv(out / "test.csv", index=False)
        _write_split_manifest(
            out, cfg, train, test, split_source="fixed_reported_membership"
        )

        if verbose:
            print(f"loaded {len(df)} compositions from {cfg.input_csv}")
            print(
                "  using recovered campaign membership: "
                f"train {len(train)}, test {len(test)}"
            )
            print(f"  written to {out}/")
        return train, test

    if cfg.up_to_cycle is None and "cycle_added" in df.columns:
        later = df["cycle_added"].astype(float) > 0
        if later.any():
            raise ValueError(
                f"{cfg.input_csv} contains {int(later.sum())} rows added during the "
                f"campaign (cycle_added > 0), but split.up_to_cycle is unset. Splitting "
                f"the file whole reproduces no cycle. Use configs/cycle1.yaml, "
                f"configs/cycle2.yaml or configs/cycle3.yaml, or set split.up_to_cycle "
                f"past the last cycle to use every row deliberately."
            )

    work = df.copy()
    work["_fom"] = figure_of_merit(work)
    work = work.sort_values("_fom", ascending=False)

    n_top = int(len(work) * cfg.top_fraction)
    top_train = work.iloc[:n_top]
    remaining = work.iloc[n_top:]

    if cfg.n_test >= len(remaining):
        raise ValueError(
            f"n_test={cfg.n_test} leaves no training data: only "
            f"{len(remaining)} rows remain after reserving the top {n_top}"
        )

    comps = remaining[list(ELEMENTS)].fillna(0.0).to_numpy(float)
    props = remaining[list(PROPERTY_COLUMNS)].to_numpy(float)
    features = StandardScaler().fit_transform(np.hstack([comps, props]))

    n_clusters = min(cfg.n_clusters, len(remaining))
    labels = KMeans(
        n_clusters=n_clusters, random_state=cfg.seed, n_init=10
    ).fit_predict(features)

    rem_train, test = train_test_split(
        remaining,
        test_size=cfg.n_test,
        stratify=labels,
        random_state=cfg.seed,
    )

    train = pd.concat([top_train, rem_train], axis=0).drop(columns="_fom")
    test = test.drop(columns="_fom")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    _write_split_manifest(out, cfg, train, test, split_source="reconstructed")

    if verbose:
        print(f"loaded {len(df)} compositions from {cfg.input_csv}")
        print(f"  top {cfg.top_fraction:.0%} by |S_ANE|/kappa ({n_top} rows) held in train")
        print(f"  train {len(train)} rows  ({n_top} reserved + {len(rem_train)} stratified)")
        print(f"  test  {len(test)} rows   (stratified over {n_clusters} clusters)")
        print(f"  written to {out}/")

    return train, test


def _composition_keys(df: pd.DataFrame) -> list[tuple[float, ...]]:
    """Stable composition keys used to validate a recovered partition."""
    comp = df[list(ELEMENTS)].fillna(0.0).to_numpy(dtype=np.float64)
    comp = comp / comp.sum(axis=1, keepdims=True)
    return [tuple(np.round(row, 10)) for row in comp]


def _validate_fixed_split(
    full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: SplitConfig,
) -> None:
    """Confirm that fixed train/test files are a true partition of ``full``."""
    if len(test) != cfg.n_test:
        raise ValueError(
            f"fixed test split has {len(test)} rows; configuration declares "
            f"n_test={cfg.n_test}"
        )

    full_keys = _composition_keys(full)
    train_keys = _composition_keys(train)
    test_keys = _composition_keys(test)
    if len(set(train_keys)) != len(train_keys):
        raise ValueError("fixed training split contains duplicate compositions")
    if len(set(test_keys)) != len(test_keys):
        raise ValueError("fixed test split contains duplicate compositions")
    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise ValueError(
            f"fixed train and test splits overlap in {len(overlap)} compositions"
        )
    if set(train_keys) | set(test_keys) != set(full_keys):
        missing = set(full_keys) - (set(train_keys) | set(test_keys))
        extra = (set(train_keys) | set(test_keys)) - set(full_keys)
        raise ValueError(
            "fixed train/test membership does not partition the cycle dataset: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if len(train) + len(test) != len(full):
        raise ValueError(
            "fixed train/test row counts do not add up to the cycle dataset"
        )


def _split_label(df: pd.DataFrame) -> list[str]:
    """A stable per-row identifier: the `label` column if present, else the
    composition rounded to the grid the search enumerates on."""
    if "label" in df.columns:
        return [str(v) for v in df["label"]]
    comp = df[list(ELEMENTS)].fillna(0.0)
    return [
        "".join(f"{e}{comp.iloc[i][e]:.3f}" for e in ELEMENTS if comp.iloc[i][e] > 0)
        for i in range(len(comp))
    ]


def _write_split_manifest(
    out: Path,
    cfg: SplitConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    split_source: str,
) -> None:
    """Record exactly which compositions landed on which side.

    train.csv and test.csv already carry the rows, but a manifest that names
    them makes the partition checkable without re-running the clustering, and
    survives being pasted into a reviewer response.
    """
    manifest = {
        "split_source": split_source,
        "up_to_cycle": cfg.up_to_cycle,
        "seed": cfg.seed,
        "top_fraction": cfg.top_fraction,
        "n_test": cfg.n_test,
        "n_clusters": cfg.n_clusters,
        "n_train": int(len(train)),
        "n_test_actual": int(len(test)),
        "train": _split_label(train),
        "test": _split_label(test),
    }
    with open(out / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
