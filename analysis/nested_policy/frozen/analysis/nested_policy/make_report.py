"""Summarize an audit from saved outer predictions, with honest incomplete status."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8-sig'))
    retry_mode = protocol.get('mode') in ('retry3', 'retry5')
    max_attempts = protocol.get('max_attempts', 1)
    files = sorted(p.parent / 'outer_predictions.csv' for p in root.glob('seed_*/fold_*/complete.json'))
    completed = len(files)
    status = '완료' if completed == 15 else '진행 중 또는 중단 — 전체 결론 금지'
    lines = ['# Real-only / Always WGAN / Adaptive nested 평가', '',
             f'상태: **{status}**. 완료된 외부 fold: {completed}/15.', '',
             '이 평가는 초기 45개 조성을 이용한 후향적 단일 DNN 분석이다. 실제 campaign의 재현, ensemble 평가 또는 능동학습 탐색 효율의 검증으로 해석하지 않는다.', '',
             'Always WGAN은 생성 실패 시 Real-only로 복귀하는 운용 정책이다. 생성에 성공한 경우만 비교한 결과와 전체 정책 결과를 구분해야 한다.', '']
    if retry_mode:
        lines[0] = f'# 최대 {max_attempts}회 WGAN 재학습 정책의 별도 nested 평가'
        lines += [f'이 폴더의 WGAN 및 Adaptive는 최초 시도를 포함해 최대 {max_attempts}회까지 생성 실패를 재시도하는 정책이다. 첫 성공을 사용하고 모두 실패하면 Real-only로 복귀한다. 기존 단일 시도 결과와 혼합하지 않는다.',
                  '기존 Real-only 및 필터 모델을 재사용하며, 증강 HPO와 Adaptive 선택 및 외부 예측은 이 정책의 생성 결과로 다시 계산한다.', '']
    if not files:
        lines += ['아직 외부 평가가 완료된 fold가 없어 성능 비교 수치를 제시하지 않는다.', '',
                  '상세 진행 상황은 status.json과 run.log, 실행 조건은 protocol.json을 확인한다.']
    else:
        p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        valid = np.isfinite(p.pred_kappa) & np.isfinite(p.pred_sane) & (p.pred_kappa > 0) & (p.pred_sane >= 0)
        p['true_fom'] = p.true_sane / p.true_kappa
        p['pred_fom'] = np.where(valid, p.pred_sane / p.pred_kappa, np.nan)
        p['invalid'] = ~valid
        rows = []
        for (seed, policy), g in p.groupby(['root_seed', 'policy']):
            measured = g[g.kappa_source == 'measured']
            rows.append({'root_seed': seed, 'policy': policy, 'n_outer_rows': len(g),
                         'n_measured_outer_rows': len(measured), 'invalid_predictions': int(g.invalid.sum()),
                         'mae_fom_measured': float((measured.true_fom - measured.pred_fom).abs().mean()) if len(measured) and not measured.invalid.any() else np.nan,
                         'mae_fom_all': float((g.true_fom - g.pred_fom).abs().mean()) if not g.invalid.any() else np.nan,
                         'mae_kappa': float((g.true_kappa - g.pred_kappa).abs().mean()),
                         'mae_sane': float((g.true_sane - g.pred_sane).abs().mean())})
        summary = pd.DataFrame(rows)
        summary.to_csv(root / 'pooled_seed_metrics.csv', index=False)
        p.to_csv(root / 'outer_predictions_with_fom.csv', index=False)
        cols = ['mae_fom_measured', 'mae_fom_all', 'mae_kappa', 'mae_sane']
        means = summary.groupby('policy')[cols].mean()
        lines += ['각 seed에서는 조성별 out-of-fold 오차를 모아 계산하고, 아래 표는 seed별 결과의 평균을 표시한다. 재구성 κ를 포함한 지표는 보조 지표이다.', '',
                  '| 정책 | 실측 행 FoM MAE (μm/A) | 전체 행 FoM MAE | κ MAE (W/mK) | SANE MAE (μV/K) |',
                  '|---|---:|---:|---:|---:|']
        for policy, row in means.iterrows():
            lines.append('| ' + policy + ' | ' + ' | '.join(f'{row[c]:.5f}' for c in cols) + ' |')
        lines += ['', '무효 예측이 있는 seed의 FoM MAE는 계산하지 않는다. 누락값을 제외한 평균은 우월성 결론으로 사용하지 않는다.', '']
        decisions = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(root.glob('seed_*/fold_*/decision_before_outer_scoring.json'))]
        selected = sum(d['selected'] == 'wgan' for d in decisions)
        successes = sum(d['full_generation_success'] for d in decisions)
        # Retry attempt markers are subordinate to each aggregate outcome;
        # counting both would double-count failures and successful retries.
        gens = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(root.glob('seed_*/fold_*/**/generation_status.json'))
                if 'attempts' not in f.relative_to(root).parts]
        lines += [f'- 최종 학습용 생성 성공: {successes}/{len(decisions)} fold.',
                  f'- Adaptive가 실제 WGAN 모델을 선택: {selected}/{len(decisions)} fold.',
                  f'- 내부 및 최종 생성의 최종 실패: {sum(not g["success"] for g in gens)}/{len(gens)}건 (시도별 중복 제외).', '',
                  '같은 45개 조성이 seed마다 반복되므로 15개 fold 또는 135개 예측을 독립 표본으로 간주하지 않는다. 본 보고서는 유의성 검정이나 일반적 우월성을 자동 선언하지 않는다.', '']
        if completed == 15:
            lines += ['전체 본평가가 끝났다. 우월성 판단 시 seed별 방향, 효과 크기, 생성 실패율, Adaptive와 Real-only의 차이를 함께 검토해야 한다.']
        else:
            lines += ['미완료 결과이다. 현재까지의 평균이나 우세한 seed만으로 결론을 내리면 안 된다.']
        policies = ['real_only', 'always_wgan_with_fallback', 'adaptive']
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for seed, g in summary.groupby('root_seed'):
            vals = g.set_index('policy').reindex(policies).mae_fom_measured.to_numpy()
            ax.plot(range(3), vals, marker='o', label=f'Seed {seed}')
        ax.set_xticks(range(3), ['Real-only', 'Retry WGAN\nwith fallback' if retry_mode else 'Always WGAN\nwith fallback', 'Retry Adaptive' if retry_mode else 'Adaptive'])
        ax.set_ylabel('FoM MAE on measured rows (μm/A)')
        ax.set_title(f'Nested audit: {completed}/15 outer folds complete')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(root / 'policy_comparison.png', dpi=180)
        plt.close(fig)
    (root / 'RESULTS_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Report updated: {completed}/15 outer folds')

if __name__ == '__main__':
    main()
