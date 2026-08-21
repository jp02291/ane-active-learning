from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from ane.features import featurize  # noqa: E402


ELEMENTS = ["Fe", "Co", "Mn", "Ga", "Al", "Si", "Ge", "Pt"]
INITIAL = REPO / "data" / "authoritative_cycle1_45.csv"
FULL = REPO / "data" / "data.csv"
OUT = HERE / "results"
RANDOM_SEED = 42
N_PERMUTATIONS = 200_000
BUDGETS = [10, 20, 25]


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def fom(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df.S_ANE.to_numpy(float)) / df.kxx.to_numpy(float)


def hypervolume_2d(kappa: np.ndarray, sane: np.ndarray) -> float:
    x = 1.0 / np.asarray(kappa, dtype=float)
    y = np.abs(np.asarray(sane, dtype=float))
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    points = sorted(zip(x[finite], y[finite]), reverse=True)
    if not points:
        return float("nan")
    area = 0.0
    max_y = 0.0
    for idx, (x_i, y_i) in enumerate(points):
        max_y = max(max_y, y_i)
        next_x = points[idx + 1][0] if idx + 1 < len(points) else 0.0
        area += (x_i - next_x) * max_y
    return float(area)


def maximin_order(initial_x: np.ndarray, candidate_x: np.ndarray) -> list[int]:
    selected: list[int] = []
    remaining = list(range(len(candidate_x)))
    reference = initial_x.copy()
    while remaining:
        block = candidate_x[remaining]
        d2 = ((block[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        min_distance = np.sqrt(d2.min(axis=1))
        local = int(np.argmax(min_distance))
        chosen = remaining.pop(local)
        selected.append(chosen)
        reference = np.vstack([reference, candidate_x[chosen]])
    return selected


def evaluate_order(
    name: str,
    order: np.ndarray,
    initial: pd.DataFrame,
    candidates: pd.DataFrame,
) -> list[dict]:
    initial_best = float(fom(initial).max())
    initial_hv = hypervolume_2d(initial.kxx.to_numpy(float), initial.S_ANE.to_numpy(float))
    top5 = set(np.argsort(fom(candidates))[-5:].tolist())
    rows = []
    for budget in BUDGETS:
        chosen = np.asarray(order[:budget], dtype=int)
        acquired = candidates.iloc[chosen]
        combined = pd.concat([initial, acquired], ignore_index=True)
        acquired_fom = fom(acquired)
        best_new = float(acquired_fom.max())
        hv = hypervolume_2d(combined.kxx.to_numpy(float), combined.S_ANE.to_numpy(float))
        rows.append({
            "strategy": name,
            "budget": budget,
            "best_new_fom": best_new,
            "best_overall_fom": max(initial_best, best_new),
            "improves_initial_best": best_new > initial_best,
            "best_new_kappa": float(acquired.iloc[int(acquired_fom.argmax())]["kxx"]),
            "best_new_S_ANE": float(acquired.iloc[int(acquired_fom.argmax())]["S_ANE"]),
            "hypervolume": hv,
            "hypervolume_gain_over_initial": hv - initial_hv,
            "final_top5_recovered": len(top5.intersection(chosen.tolist())),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    initial = pd.read_csv(INITIAL)
    full = pd.read_csv(FULL)
    candidates = full.loc[full.cycle_added.astype(int) > 0].copy().reset_index(drop=True)
    candidates.insert(0, "candidate_id", [f"EXP_{i+1:02d}" for i in range(len(candidates))])
    candidates["FOM_absS_over_kappa"] = fom(candidates)
    if len(initial) != 45 or len(candidates) != 25:
        raise RuntimeError("expected 45 initial and 25 prospectively measured candidates")

    active_order = np.argsort(candidates.cycle_added.to_numpy(int), kind="stable")
    raw_order = np.array(
        maximin_order(initial[ELEMENTS].to_numpy(float), candidates[ELEMENTS].to_numpy(float)),
        dtype=int,
    )
    all_comp = np.vstack([initial[ELEMENTS].to_numpy(float), candidates[ELEMENTS].to_numpy(float)])
    all_feat = StandardScaler().fit_transform(featurize(all_comp))
    feature_order = np.array(maximin_order(all_feat[:45], all_feat[45:]), dtype=int)

    strategy_rows = []
    strategy_rows.extend(evaluate_order("active_learning_observed", active_order, initial, candidates))
    strategy_rows.extend(evaluate_order("diversity_only_raw_composition", raw_order, initial, candidates))
    strategy_rows.extend(evaluate_order("diversity_only_standardized_features", feature_order, initial, candidates))
    strategies = pd.DataFrame(strategy_rows)

    rng = np.random.default_rng(RANDOM_SEED)
    random_orders = np.argsort(rng.random((N_PERMUTATIONS, len(candidates))), axis=1)
    candidate_fom = fom(candidates)
    initial_fom_best = float(fom(initial).max())
    initial_hv = hypervolume_2d(initial.kxx.to_numpy(float), initial.S_ANE.to_numpy(float))
    top5 = set(np.argsort(candidate_fom)[-5:].tolist())
    random_rows = []
    random_distributions: dict[int, dict[str, np.ndarray]] = {}
    for budget in BUDGETS:
        idx = random_orders[:, :budget]
        best = candidate_fom[idx].max(axis=1)
        recovered = np.array([len(top5.intersection(row.tolist())) for row in idx], dtype=int)
        # Hypervolume is inexpensive at 25 points but not vectorized; evaluate
        # a deterministic 20,000-permutation subset for this metric.
        hv_n = min(20_000, N_PERMUTATIONS)
        hv = np.empty(hv_n, dtype=float)
        for n in range(hv_n):
            acquired = candidates.iloc[idx[n]]
            combined = pd.concat([initial, acquired], ignore_index=True)
            hv[n] = hypervolume_2d(combined.kxx.to_numpy(float), combined.S_ANE.to_numpy(float))
        random_distributions[budget] = {"best": best, "recovered": recovered, "hv": hv}
        random_rows.append({
            "strategy": "random_without_replacement",
            "budget": budget,
            "best_new_fom_mean": float(best.mean()),
            "best_new_fom_median": float(np.median(best)),
            "best_new_fom_q025": float(np.quantile(best, 0.025)),
            "best_new_fom_q975": float(np.quantile(best, 0.975)),
            "probability_improves_initial_best": float(np.mean(best > initial_fom_best)),
            "top5_recovered_mean": float(recovered.mean()),
            "hypervolume_gain_mean": float(hv.mean() - initial_hv),
            "hypervolume_gain_q025": float(np.quantile(hv - initial_hv, 0.025)),
            "hypervolume_gain_q975": float(np.quantile(hv - initial_hv, 0.975)),
            "permutations_best_and_recovery": N_PERMUTATIONS,
            "permutations_hypervolume": hv_n,
        })
    random_summary = pd.DataFrame(random_rows)

    comparisons = []
    for row in strategies.itertuples(index=False):
        dist = random_distributions[int(row.budget)]
        comparisons.append({
            **row._asdict(),
            "percentile_vs_random_best_new_fom": float(np.mean(dist["best"] <= row.best_new_fom)),
            "random_probability_best_at_least_as_high": float(np.mean(dist["best"] >= row.best_new_fom)),
            "percentile_vs_random_hypervolume": float(np.mean(dist["hv"] <= row.hypervolume)),
            "random_probability_hypervolume_at_least_as_high": float(np.mean(dist["hv"] >= row.hypervolume)),
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    random_summary.to_csv(OUT / "random_baseline_summary.csv", index=False, encoding="utf-8-sig")

    candidates["active_order"] = np.argsort(active_order) + 1
    candidates["diversity_raw_order"] = np.argsort(raw_order) + 1
    candidates["diversity_feature_order"] = np.argsort(feature_order) + 1
    candidates.to_csv(OUT / "candidate_universe_and_orders.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    colors = {
        "active_learning_observed": "#1f77b4",
        "diversity_only_raw_composition": "#2ca02c",
        "diversity_only_standardized_features": "#9467bd",
    }
    for strategy, group in comparison.groupby("strategy"):
        axes[0].plot(group.budget, group.best_overall_fom, marker="o", label=strategy, color=colors[strategy])
        axes[1].plot(group.budget, group.hypervolume_gain_over_initial, marker="o", label=strategy, color=colors[strategy])
    axes[0].plot(random_summary.budget, np.maximum(initial_fom_best, random_summary.best_new_fom_mean), marker="o", linestyle="--", color="#777777", label="random mean")
    axes[0].fill_between(random_summary.budget, np.maximum(initial_fom_best, random_summary.best_new_fom_q025), np.maximum(initial_fom_best, random_summary.best_new_fom_q975), color="#aaaaaa", alpha=0.25)
    axes[1].plot(random_summary.budget, random_summary.hypervolume_gain_mean, marker="o", linestyle="--", color="#777777", label="random mean")
    axes[1].fill_between(random_summary.budget, random_summary.hypervolume_gain_q025, random_summary.hypervolume_gain_q975, color="#aaaaaa", alpha=0.25)
    axes[0].set_xlabel("Measured-candidate budget")
    axes[0].set_ylabel("Best |S_ANE| / kappa (including initial 45)")
    axes[1].set_xlabel("Measured-candidate budget")
    axes[1].set_ylabel("2D objective hypervolume gain over initial 45")
    axes[0].set_title("Best observed FOM")
    axes[1].set_title("Pareto hypervolume")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "baseline_comparison.png", dpi=220)
    fig.savefig(OUT / "baseline_comparison.pdf")
    plt.close(fig)

    active = comparison.loc[comparison.strategy == "active_learning_observed"].copy()
    summary = {
        "benchmark_type": "retrospective conditional benchmark on the 25 experimentally measured campaign candidates",
        "initial_rows": 45,
        "candidate_universe_rows": 25,
        "budgets": BUDGETS,
        "random_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "random_hypervolume_permutations": int(random_summary.permutations_hypervolume.iloc[0]),
        "initial_best_FOM": initial_fom_best,
        "active_learning": active.to_dict("records"),
        "major_limitation": (
            "The 25-candidate evaluation universe was itself chosen by active learning. Random and diversity-only "
            "baselines therefore test prioritization within the experimentally observed set, not counterfactual "
            "full-space discovery efficiency. Unmeasured random candidates have no experimental labels."
        ),
        "defensible_use": (
            "Report this as a conditional retrospective sensitivity analysis. Do not claim definitive superiority "
            "over random search unless an independently labeled candidate benchmark or prospective control is added."
        ),
        "inputs": {"authoritative_45_sha256": sha256(INITIAL), "data_csv_sha256": sha256(FULL)},
    }
    (OUT / "baseline_interpretation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
