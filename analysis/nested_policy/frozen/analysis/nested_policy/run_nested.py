"""Strict fold-local, retrospective three-policy audit; no campaign artifacts fitted.

Step 1 deliberately evaluates single refitted DNNs, as in manuscript Fig. 3.
The ensemble/ranking audit is a separate subsequent experiment.
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from ane import surrogate as s, augment as a
from ane.config import GANConfig, SurrogateConfig
from ane.features import featurize

C = list(a.COMP_COLS)
Y = list(a.TARGET_COLS)
ORIGINAL_FOLD_ARRAYS = s._fold_arrays
ORIGINAL_BUILD_MODEL = s.build_model
ORIGINAL_OBJECTIVE = s.cv_objective_loss
ACTIVE_STATUS = None

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=s._json_default), encoding='utf-8')
    tmp.replace(path)

def event(message, **kwargs):
    record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'message': message, **kwargs}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if ACTIVE_STATUS:
        dump(ACTIVE_STATUS, record)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def data_arrays(df):
    return df[C].to_numpy(float), df[Y].to_numpy(float)

def dataset_fit_model(*args, **kwargs):
    """Same Keras model/loss/callbacks, explicit one-thread tf.data input.

    Avoids creating a large private pool at every tiny-data epoch. Shuffle
    order differs from Keras's implicit NumPy adapter and is disclosed.
    """
    model, bs = ORIGINAL_BUILD_MODEL(*args, **kwargs)
    original_fit = model.fit
    def fit(x, y=None, **kw):
        if y is None or not isinstance(x, np.ndarray):
            return original_fit(x, y, **kw)
        batch = kw.pop('batch_size', bs)
        weights = kw.pop('sample_weight', None)
        shuffle = kw.pop('shuffle', True)
        validation = kw.pop('validation_data', None)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        options.experimental_deterministic = True
        vals = (np.asarray(x, np.float32), np.asarray(y, np.float32))
        if weights is not None:
            vals += (np.asarray(weights, np.float32),)
        ds = tf.data.Dataset.from_tensor_slices(vals)
        if shuffle:
            ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
        ds = ds.batch(batch).with_options(options)
        if validation is not None:
            vd = tf.data.Dataset.from_tensor_slices(tuple(np.asarray(v, np.float32) for v in validation))
            kw['validation_data'] = vd.batch(batch).with_options(options)
        return original_fit(ds, **kw)
    model.fit = fit
    return model, bs

s.build_model = dataset_fit_model

def choose_branch(real_score, aug_score, available=True):
    return 'wgan' if available and np.isfinite(aug_score) and aug_score < real_score else 'real_only'

def config(directory, seed, smoke):
    cfg = SurrogateConfig(artifact_dir=str(directory), seed=int(seed))
    if smoke:
        cfg.optuna_trials = 1
        cfg.epochs_tune = 2
        cfg.epochs_final = 3
        cfg.k_folds = 2
    return cfg

def run_search(cfg, comp, targets, has_generated=False):
    """Resumable TPE with sampler state saved after every trial."""
    directory = Path(cfg.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sampler_path = directory / 'sampler.joblib'
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(
        direction='minimize', sampler=sampler, study_name='hpo',
        storage='sqlite:///' + (directory / 'study.sqlite3').as_posix(), load_if_exists=True,
    )
    # An interrupted RUNNING trial is failed, not silently reused or counted as evidence.
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    def objective(trial):
        p = s.suggest_params(trial, cfg, has_generated)
        value = s.cv_objective_loss(p, comp, targets, None, None, cfg)
        return float(value)
    def callback(study, trial):
        joblib.dump(study.sampler, sampler_path)
        event('HPO trial finished', directory=str(directory), trial=trial.number + 1,
              budget=cfg.optuna_trials, value=trial.value, best=study.best_value)
        dump(directory / 'best_params_live.json', study.best_params)
    if completed < cfg.optuna_trials:
        study.optimize(objective, n_trials=cfg.optuna_trials - completed,
                       callbacks=[callback], gc_after_trial=True)
    study.trials_dataframe().to_csv(directory / 'trials.csv', index=False)
    if not np.isfinite(study.best_value):
        raise RuntimeError('No finite HPO result; this is a computational/model failure, not a GAN rejection.')
    params = dict(study.best_params)
    params.setdefault('w_gen', cfg.w_gen_without_generated)
    dump(directory / 'best_params.json', params)
    return params, float(study.best_value)

def refit(df, generated, params, epochs, cfg):
    """Reference APIs receive training rows as diagnostic rows, NEVER held-out rows."""
    comp, targets = data_arrays(df)
    gc_, gy = data_arrays(generated) if generated is not None else (None, None)
    s.train_final_model(params, epochs, comp, targets, gc_, gy, comp, targets, cfg, verbose=False)
    directory = Path(cfg.artifact_dir)
    # These are explicitly training diagnostics, not test metrics.
    for old, new in [('metrics_test.json', 'training_diagnostics.json'),
                     ('test_predictions.csv', 'training_predictions.csv')]:
        (directory / old).replace(directory / new)

def fit_real(df, directory, seed, smoke):
    directory = Path(directory)
    result_file = directory / 'fit_result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))
    directory.mkdir(parents=True, exist_ok=True)
    dump(directory / 'training_ids.json', df.sample_id.tolist())
    cfg = config(directory, seed, smoke)
    comp, targets = data_arrays(df)
    event('Training real-only model', directory=str(directory), n=len(df))
    params, score = run_search(cfg, comp, targets)
    epoch_record = s.select_final_epoch(params, comp, targets, None, None, cfg, verbose=False)
    refit(df, None, params, epoch_record['best_epoch'], cfg)
    result = {'directory': str(directory), 'cv_score': score, 'params': params,
              'epoch_selection': epoch_record, 'training_ids': df.sample_id.tolist()}
    dump(result_file, result)
    return result

def predict(fit, comp):
    d = Path(fit['directory'])
    model = tf.keras.models.load_model(d / 'final_best_model.h5', compile=False)
    sx, sy = joblib.load(d / 'sx_final.joblib'), joblib.load(d / 'sy_final.joblib')
    out = sy.inverse_transform(model(sx.transform(featurize(comp)), training=False).numpy())
    reg = float(sum(float(v.numpy()) for v in model.losses))
    del model
    tf.keras.backend.clear_session()
    return out, reg

def generate(df, reference, directory, seed, smoke):
    directory = Path(directory)
    marker = directory / 'generation_status.json'
    if marker.exists():
        status = json.loads(marker.read_text(encoding='utf-8'))
        return pd.read_csv(status['csv']) if status['success'] else None, status
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / 'train.csv', index=False)
    # No held-out compositions are passed, even for duplicate exclusion.
    pd.DataFrame(columns=C + Y).to_csv(directory / 'empty_test.csv', index=False)
    cfg = GANConfig(
        train_csv=str(directory / 'train.csv'), test_csv=str(directory / 'empty_test.csv'),
        output_dir=str(directory), output_prefix=str(directory / 'augmented_n'),
        generated_sizes=(200,), seed=int(seed), pool_seed=int(seed + 100000),
        surrogate_model_dir=reference['directory'],
    )
    if smoke:
        cfg.epochs = 2
        cfg.eval_interval = 1
        cfg.fake_max_rounds = 1
        cfg.final_max_rounds = 1
        cfg.proxy_max_iter = 3
    dump(directory / 'config.json', dataclasses.asdict(cfg))
    dump(directory / 'training_ids.json', df.sample_id.tolist())
    tf.keras.utils.set_random_seed(seed)
    event('Training fold-local WGAN', directory=str(directory), n=len(df), epochs=cfg.epochs)
    status = {'training_ids': df.sample_id.tolist(), 'seed': seed, 'requested_n': 200}
    with (directory / 'training.log').open('w', encoding='utf-8', buffering=1) as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                manifest = a.train_gan(cfg, verbose=True)
            path = str(directory / 'augmented_n200.csv')
            generated = pd.read_csv(path)
            if len(generated) != 200:
                raise AssertionError('Wrong augmentation count')
            status.update(success=True, csv=path, manifest=manifest)
        except RuntimeError as exc:
            # Only documented statistical generation rejection is a policy fallback.
            expected = ('no checkpoint met', 'no generated composition survived', 'the filtered pool holds')
            if not any(str(exc).startswith(prefix) for prefix in expected):
                raise
            generated = None
            status.update(success=False, failure_type='generation_rejection', reason=str(exc))
    dump(marker, status)
    event('WGAN finished', directory=str(directory), success=status['success'])
    return generated, status

def prepare_inner(df, directory, seed, smoke):
    cfg = config(directory, seed, smoke)
    folds = []
    for i, (tr, va) in enumerate(KFold(cfg.k_folds, shuffle=True, random_state=seed).split(df), 1):
        train = df.iloc[tr].reset_index(drop=True)
        forbidden_ids = set(df.iloc[va].sample_id)
        d = Path(directory) / f'inner_{i}'
        local_seed = int((seed + 1009 * i) % (2**31 - 1))
        # HPO for the filter model occurs ONLY within this inner-training subset.
        reference = fit_real(train, d / 'reference', local_seed, smoke)
        assert not forbidden_ids.intersection(reference['training_ids'])
        generated, status = generate(train, reference, d / 'gan', local_seed, smoke)
        assert not forbidden_ids.intersection(status['training_ids'])
        fallback_pred, reg = predict(reference, df.iloc[va][C].to_numpy(float))
        sy = MinMaxScaler().fit(train[Y].to_numpy(float))
        fallback_loss = float(np.mean((sy.transform(df.iloc[va][Y]) - sy.transform(fallback_pred))**2) + reg)
        folds.append({'train': tr, 'val': va, 'generated': generated, 'status': status,
                      'fallback_loss': fallback_loss, 'reference': reference})
        dump(d / 'membership.json', {'train': train.sample_id.tolist(), 'validation': df.iloc[va].sample_id.tolist()})
    return folds

@contextlib.contextmanager
def local_generated_folds(folds):
    """Patch only the data provider for the existing CV/epoch selection code."""
    lookup = {tuple(f['train']): f for f in folds}
    def provider(comp, y, tr, va, unused_comp, unused_y, weight):
        f = lookup[tuple(tr)]
        assert np.array_equal(va, f['val'])
        gen = f['generated']
        gc_, gy = data_arrays(gen) if gen is not None else (None, None)
        return ORIGINAL_FOLD_ARRAYS(comp, y, tr, va, gc_, gy, weight)
    def objective(params, comp, y, unused_comp, unused_y, cfg):
        # Successful folds use the original CV loss. Rejected folds use the fully
        # local real-only fallback model, including its own local HPO.
        losses = []
        for f in folds:
            if f['generated'] is None:
                losses.append(f['fallback_loss'])
                continue
            tr, va = f['train'], f['val']
            Xtr, ytr, Xva, yva, weight = provider(comp, y, tr, va, None, None, params['w_gen'])
            tf.keras.utils.set_random_seed(cfg.seed)
            model, batch = s.build_model(Xtr.shape[1], params, n_targets=y.shape[1])
            h = model.fit(Xtr, ytr, validation_data=(Xva, yva), sample_weight=weight,
                          epochs=cfg.epochs_tune, batch_size=batch, verbose=0,
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=cfg.early_stopping_patience, restore_best_weights=True),
                                     tf.keras.callbacks.ReduceLROnPlateau(factor=cfg.lr_factor, patience=cfg.tune_lr_patience, min_lr=cfg.lr_min)])
            value = float(np.min(h.history['val_loss']))
            losses.append(value if np.isfinite(value) else float('inf'))
            del model, h
            tf.keras.backend.clear_session()
        return float(np.mean(losses))
    s._fold_arrays, s.cv_objective_loss = provider, objective
    try:
        yield
    finally:
        s._fold_arrays, s.cv_objective_loss = ORIGINAL_FOLD_ARRAYS, ORIGINAL_OBJECTIVE

def fit_wgan(df, real_fit, folds, full_generated, full_status, directory, seed, smoke):
    directory = Path(directory)
    result_file = directory / 'fit_result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))
    cfg = config(directory, seed, smoke)
    comp, targets = data_arrays(df)
    with local_generated_folds(folds):
        params, score = run_search(cfg, comp, targets, has_generated=True)
        # Preserve the deposited epoch-choice rule. On rejected folds this
        # diagnostic uses the trial architecture without synthetic samples.
        epoch_record = s.select_final_epoch(params, comp, targets, None, None, cfg, verbose=False)
    if full_generated is None:
        deployed = real_fit
    else:
        refit(df, full_generated, params, epoch_record['best_epoch'], cfg)
        deployed = {'directory': str(directory)}
    result = {'directory': deployed['directory'], 'cv_score': score, 'params': params,
              'epoch_selection': epoch_record, 'full_generation_success': full_status['success'],
              'inner_generation_successes': sum(f['status']['success'] for f in folds),
              'inner_generation_attempts': len(folds), 'training_ids': df.sample_id.tolist()}
    dump(result_file, result)
    return result

def metrics(truth, pred, training_y, measured_mask):
    sy = MinMaxScaler().fit(training_y)
    valid = np.isfinite(pred).all(axis=1) & (pred[:, 0] > 0) & (pred[:, 1] >= 0)
    true_fom = truth[:, 1] / truth[:, 0]
    pred_fom = np.full(len(pred), np.nan)
    pred_fom[valid] = pred[valid, 1] / pred[valid, 0]
    # Never hide invalid predictions by dropping rows from the headline metric.
    fom_mae = float(np.mean(np.abs(pred_fom - true_fom))) if valid.all() else None
    measured_fom = float(np.mean(np.abs(pred_fom[measured_mask] - true_fom[measured_mask]))) if valid[measured_mask].all() and measured_mask.any() else None
    return {'mae_fom_all': fom_mae, 'mae_fom_measured': measured_fom,
            'scaled_mse': float(np.mean((sy.transform(truth) - sy.transform(pred))**2)),
            'mae_kappa': float(np.mean(np.abs(pred[:, 0] - truth[:, 0]))),
            'mae_sane': float(np.mean(np.abs(pred[:, 1] - truth[:, 1]))),
            'invalid_prediction_count': int((~valid).sum()), 'n': len(truth),
            'n_measured': int(measured_mask.sum())}

def run_fold(data, output, root_seed, fold_no, train_idx, test_idx, smoke):
    d = output / f'seed_{root_seed}' / f'fold_{fold_no}'
    if (d / 'complete.json').exists():
        return
    train, held = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)
    seed = int((root_seed + fold_no * 100003) % (2**31 - 1))
    dump(d / 'membership.json', {'outer_train': train.sample_id.tolist(), 'outer_test': held.sample_id.tolist(), 'fit_seed': seed})
    real = fit_real(train, d / 'real_only', seed, smoke)
    folds = prepare_inner(train, d / 'inner', seed, smoke)
    generated, status = generate(train, real, d / 'full_gan', seed, smoke)
    augmented = fit_wgan(train, real, folds, generated, status, d / 'always_wgan', seed, smoke)
    selected = choose_branch(real['cv_score'], augmented['cv_score'], status['success'])
    decision = {'selected': selected, 'real_inner_score': real['cv_score'],
                'wgan_inner_score': augmented['cv_score'], 'full_generation_success': status['success'],
                'criterion': 'mean minimum fold-validation scaled MSE including model regularization; ties choose real-only',
                'outer_targets_used': False}
    # Written BEFORE any outer targets are used for scoring.
    dump(d / 'decision_before_outer_scoring.json', decision)
    assert not set(held.sample_id).intersection(real['training_ids'])
    pr, _ = predict(real, held[C].to_numpy(float))
    pa, _ = predict(augmented, held[C].to_numpy(float))
    predictions = {'real_only': pr, 'always_wgan_with_fallback': pa,
                   'adaptive': pa if selected == 'wgan' else pr}
    rows, summaries = [], []
    truth = held[Y].to_numpy(float)
    mask = held.kappa_source.eq('measured').to_numpy()
    for policy, pred in predictions.items():
        metric = metrics(truth, pred, train[Y].to_numpy(float), mask)
        summaries.append({'root_seed': root_seed, 'outer_fold': fold_no, 'policy': policy, **metric,
                          'adaptive_selected': selected, 'full_generation_success': status['success'],
                          'inner_generation_successes': augmented['inner_generation_successes']})
        for j, record in held.iterrows():
            rows.append({'root_seed': root_seed, 'outer_fold': fold_no, 'policy': policy,
                         'sample_id': record.sample_id, 'kappa_source': record.kappa_source,
                         'true_kappa': truth[j, 0], 'true_sane': truth[j, 1],
                         'pred_kappa': pred[j, 0], 'pred_sane': pred[j, 1],
                         'adaptive_selected': selected})
    pd.DataFrame(rows).to_csv(d / 'outer_predictions.csv', index=False)
    pd.DataFrame(summaries).to_csv(d / 'outer_metrics.csv', index=False)
    dump(d / 'complete.json', {'decision': decision, 'summaries': summaries})
    event('Outer fold complete', root_seed=root_seed, fold=fold_no, decision=selected)

def summarize(output):
    metric_files = sorted(output.glob('seed_*/fold_*/outer_metrics.csv'))
    prediction_files = sorted(output.glob('seed_*/fold_*/outer_predictions.csv'))
    if not metric_files:
        return
    m = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
    p = pd.concat([pd.read_csv(v) for v in prediction_files], ignore_index=True)
    m.to_csv(output / 'fold_metrics.csv', index=False)
    p.to_csv(output / 'predictions.csv', index=False)
    cols = ['mae_fom_all', 'mae_fom_measured', 'scaled_mse', 'mae_kappa', 'mae_sane', 'invalid_prediction_count']
    m.groupby(['root_seed', 'policy'])[cols].mean().to_csv(output / 'seed_summary.csv')
    m.groupby('policy')[cols].mean().to_csv(output / 'policy_summary.csv')
    # Descriptive paired deltas only; repeated folds are not independent samples.
    for metric in ['mae_fom_all', 'mae_fom_measured', 'scaled_mse']:
        wide = m.pivot(index=['root_seed', 'outer_fold'], columns='policy', values=metric)
        wide['adaptive_minus_real'] = wide.adaptive - wide.real_only
        wide['wgan_minus_real'] = wide.always_wgan_with_fallback - wide.real_only
        wide.to_csv(output / f'paired_{metric}.csv')
    dump(output / 'completion_summary.json', {'completed_outer_folds': len(metric_files),
          'expected_outer_folds': 15, 'complete': len(metric_files) == 15,
          'interpretation': 'Retrospective single-DNN audit; not independent datasets, not campaign reproduction. Always-WGAN includes explicitly reported real-only fallback on generation rejection.'})

def main():
    global ACTIVE_STATUS
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, action='append')
    ap.add_argument('--fold', type=int, action='append')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ACTIVE_STATUS = output / 'status.json'
    if args.summarize_only:
        summarize(output)
        return
    data = pd.read_csv(ROOT / 'data/data.csv')
    data = data.loc[data.cycle_added == 0].reset_index(drop=True)
    authoritative = pd.read_csv(ROOT / 'data/authoritative_cycle1_45.csv')
    assert len(data) == 45 and data.sample_id.is_unique
    check = data.set_index('sample_id')[C + Y].sort_index()
    expected = authoritative.set_index('sample_id')[C + Y].sort_index()
    np.testing.assert_allclose(check, expected, rtol=0, atol=1e-10)
    versions = {k: importlib.metadata.version(k) for k in ['tensorflow', 'keras', 'optuna', 'numpy', 'scikit-learn', 'pandas', 'scipy']}
    protocol = {'version': 1, 'smoke': args.smoke, 'data_sha256': sha(ROOT / 'data/data.csv'),
        'code_sha256': sha(__file__), 'versions': versions, 'python': sys.version,
        'dataset': '45 initial compositions, including 13 reconstructed thermal conductivities',
        'outer_seeds': [42, 2025, 314159], 'outer_folds': 5, 'retain_top_in_training': False,
        'inner_folds': 2 if args.smoke else 5, 'hpo_trials_per_branch': 1 if args.smoke else 100,
        'gan_epochs_max': 2 if args.smoke else 10000, 'synthetic_samples': 200,
        'selection_metric': 'deposited minimum fold validation scaled MSE (Keras loss includes L2)',
        'evaluation_primary': 'FoM MAE on directly measured outer rows',
        'evaluation_secondary': ['FoM MAE all rows', 'kappa MAE', 'S_ANE MAE', 'scaled MSE', 'invalid prediction rate'],
        'augmentation': 'fold-local generator, scalers, filter model, threshold, checkpoint and generation; filter HPO inside its allowed training subset',
        'fallback': 'inner generation rejection uses independently inner-trained tuned real-only reference; final generation rejection deploys final real-only; failures always retained',
        'adaptive': 'select WGAN only if its inner score is lower and final generation succeeds',
        'epoch_rule': 'preserve deposited best-fold epoch selection; do not use it to rank policies',
        'differences_from_campaign': ['strict inner-fold-local generation', 'no top-15% training retention', 'fixed NG=200', 'explicit tf.data adapter', 'installed versions recorded; not historical locked environment', 'no held-out compositions read by generator', 'no historical checkpoints reused'],
        'scope': 'step 1 single-model policy comparison; no ensemble superiority or acquisition efficiency inference'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding='utf-8'))
        if old != protocol:
            raise RuntimeError('Frozen protocol differs. Use a new output directory; do not mix runs.')
    else:
        dump(protocol_path, protocol)
    data.to_csv(output / 'input_45.csv', index=False)
    for root_seed in [42, 2025, 314159]:
        for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=root_seed).split(data), 1):
            dump(output / 'splits' / f'seed_{root_seed}_fold_{i}.json',
                 {'train_ids': data.iloc[tr].sample_id.tolist(), 'test_ids': data.iloc[te].sample_id.tolist()})
    if args.prepare_only:
        event('Protocol and 15 outer splits frozen', output=str(output))
        return
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tf.get_logger().setLevel('ERROR')
    event('Nested audit started', output=str(output), smoke=args.smoke)
    for root_seed in args.seed or [42, 2025, 314159]:
        for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=root_seed).split(data), 1):
            if args.fold and i not in args.fold:
                continue
            run_fold(data, output, root_seed, i, tr, te, args.smoke)
            summarize(output)
    event('Requested audit jobs finished', output=str(output))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        event('Audit stopped on error', traceback=traceback.format_exc())
        raise
