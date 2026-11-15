import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore')
SEED = 20261014
N_IMP = int(os.environ.get('N_IMP', 10))          # preregistered: 10 imputations
N_ITER = int(os.environ.get('N_ITER', 10))        # chained-equation cycles
N_CONFIGS = int(os.environ.get('N_CONFIGS', 50))  # preregistered: random search over 50 configurations
K_INNER = 5
K_OUTER = int(os.environ.get('K_OUTER', 5))
PMM_DONORS = 5

DEV_CSV = sys.argv[1] if len(sys.argv) > 1 else 'analysis/development_2011_2016.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'models'
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True); LOG.append(line)

# ---------------------------------------------------------------- variables
PREDICTORS = ['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'family_history',
              'gestational_diabetes', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_total']
CONT_IMP = ['bmi', 'waist_cm', 'mvpa_min_week_total']
BIN_IMP = ['hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation']
COMPLETE = ['age', 'male']
OUTCOME = 'y_hba1c'


def design_for_imputation(df, target, use_y, y=None):
    """Predictors used to impute `target`: all other items (mvpa on log scale) plus y when allowed."""
    cols = [c for c in COMPLETE + CONT_IMP + BIN_IMP if c != target]
    X = df[cols].copy()
    if 'mvpa_min_week_total' in X:
        X['mvpa_min_week_total'] = np.log1p(X['mvpa_min_week_total'])
    if use_y:
        X['_y'] = y
    return X.values.astype(float)


class ChainedImputer:
    """
    Multiple imputation by chained equations, one imputation per instance.
    - Continuous items: linear regression on a bootstrap sample + predictive mean matching (5 donors).
    - Binary items: logistic regression on a bootstrap sample + Bernoulli draw.
    - Gestational diabetes is imputed among women only (men are 0 by definition).
    fit_transform(train, y) imputes the training data using the outcome as a predictor.
    transform(new) imputes new data with separately stored models that do NOT use the outcome,
    so held-out folds and validation samples are imputed without their outcomes.
    """
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.models = {}       # outcome-free models for transform()
        self.donors = {}       # observed training values and their fitted means for PMM

    def _subset(self, df, var):
        return (df['male'] == 0).values if var == 'gestational_diabetes' else np.ones(len(df), bool)

    def _fit_one(self, X, yv, var):
        idx = self.rng.integers(0, len(yv), len(yv))     # bootstrap -> proper imputation
        if var in CONT_IMP:
            t = np.log1p(yv) if var == 'mvpa_min_week_total' else yv
            m = LinearRegression().fit(X[idx], t[idx])
        else:
            if len(np.unique(yv[idx])) < 2:
                return ('const', float(yv.mean()))
            m = LogisticRegression(C=1e4, max_iter=2000).fit(X[idx], yv[idx])
        return m

    def _draw(self, m, X, var, donors_x=None, donors_y=None):
        if isinstance(m, tuple):
            return self.rng.binomial(1, m[1], len(X)).astype(float)
        if var in CONT_IMP:
            pred_mis = m.predict(X)
            pred_obs = m.predict(donors_x)
            out = np.empty(len(X))
            for i, p in enumerate(pred_mis):
                nn = np.argpartition(np.abs(pred_obs - p), PMM_DONORS)[:PMM_DONORS]
                out[i] = donors_y[self.rng.choice(nn)]
            return out
        return self.rng.binomial(1, m.predict_proba(X)[:, 1]).astype(float)

    def _initial_fill(self, df, fill):
        df = df.copy()
        for v in CONT_IMP + BIN_IMP:
            miss = df[v].isna()
            if miss.any():
                df.loc[miss, v] = fill[v]
        return df

    def fit_transform(self, train, y):
        self.fill = {v: (train[v].median() if v in CONT_IMP else float(round(train[v].mean()))) for v in CONT_IMP + BIN_IMP}
        miss = {v: train[v].isna().values for v in CONT_IMP + BIN_IMP}
        cur = self._initial_fill(train, self.fill)
        cur.loc[cur.male == 1, 'gestational_diabetes'] = 0.0
        for _ in range(N_ITER):
            for v in CONT_IMP + BIN_IMP:
                sub = self._subset(cur, v)
                mv = miss[v] & sub
                if not mv.any():
                    continue
                obs = (~miss[v]) & sub
                X = design_for_imputation(cur, v, True, y)
                m = self._fit_one(X[obs], cur[v].values[obs], v)
                cur.loc[mv, v] = self._draw(m, X[mv], v, X[obs], cur[v].values[obs])
        # outcome-free models for imputing new data, fitted on the completed training data
        for v in CONT_IMP + BIN_IMP:
            sub = self._subset(cur, v)
            obs = (~miss[v]) & sub
            X = design_for_imputation(cur, v, False)
            m = self._fit_one(X[obs], cur[v].values[obs], v)
            self.models[v] = m
            self.donors[v] = (X[obs], cur[v].values[obs])
        return finalise(cur)

    def transform(self, new):
        miss = {v: new[v].isna().values for v in CONT_IMP + BIN_IMP}
        cur = self._initial_fill(new, self.fill)
        cur.loc[cur.male == 1, 'gestational_diabetes'] = 0.0
        for _ in range(N_ITER):
            for v in CONT_IMP + BIN_IMP:
                mv = miss[v] & self._subset(cur, v)
                if not mv.any():
                    continue
                X = design_for_imputation(cur, v, False)
                dx, dy = self.donors[v]
                cur.loc[mv, v] = self._draw(self.models[v], X[mv], v, dx, dy)
        return finalise(cur)


def finalise(df):
    """Restore logical consistency and recompute derived items after imputation."""
    df = df.copy()
    df.loc[df.hypertension_dx == 0, 'bp_medication'] = 0.0
    df.loc[df.male == 1, 'gestational_diabetes'] = 0.0
    df['mvpa_min_week_total'] = df['mvpa_min_week_total'].clip(0, 3360)
    df['hypertension_ada'] = ((df.hypertension_dx == 1) | (df.bp_medication == 1)).astype(float)
    df['high_glucose_history'] = ((df.told_prediabetes == 1) | (df.gestational_diabetes == 1)).astype(float)
    df['ada_score'] = ada_score(df)
    df['findrisc_adapted'] = findrisc_adapted(df)
    return df


def ada_score(d):
    age = np.select([d.age < 40, d.age < 50, d.age < 60], [0, 1, 2], 3)
    bmi = np.select([d.bmi < 25, d.bmi < 30, d.bmi < 40], [0, 1, 2], 3)
    return age + d.male + d.gestational_diabetes + d.family_history + d.hypertension_ada + (1 - d.active_recreation) + bmi


def findrisc_adapted(d):
    age = np.select([d.age < 45, d.age < 55, d.age < 65], [0, 2, 3], 4)
    bmi = np.select([d.bmi < 25, d.bmi <= 30], [0, 1], 3)
    w = d.waist_cm
    waist = np.where(d.male == 1, np.select([w < 94, w <= 102], [0, 3], 4), np.select([w < 80, w <= 88], [0, 3], 4))
    pa = np.where(d.mvpa_min_week_total >= 210, 0, 2)
    return age + bmi + waist + pa + 2 * d.bp_medication + 5 * d.high_glucose_history + 5 * d.family_history


# ---------------------------------------------------------------- logistic regression with splines
SPLINE_VARS = ['age', 'bmi', 'waist_cm', 'log_mvpa']

def rcs_basis(x, knots):
    """Restricted cubic spline (Harrell), 3 knots -> linear term + 1 nonlinear term."""
    k1, k2, k3 = knots
    pos = lambda u: np.maximum(u, 0) ** 3
    scale = (k3 - k1) ** 2
    nl = (pos(x - k1) - pos(x - k2) * (k3 - k1) / (k3 - k2) + pos(x - k3) * (k2 - k1) / (k3 - k2)) / scale
    return np.column_stack([x, nl])

def lr_design(d, knots):
    d = d.copy(); d['log_mvpa'] = np.log1p(d.mvpa_min_week_total)
    parts = [rcs_basis(d[v].values.astype(float), knots[v]) for v in SPLINE_VARS]
    binv = ['male', 'hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation']
    parts.append(d[binv].values.astype(float))
    return np.column_stack(parts)

def fit_lr(X, y):
    """Unpenalised logistic regression; Firth-type fallback (ridge-stabilised) only if it fails to converge."""
    m = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            m.fit(X, y); m._fallback = False; return m
        except Exception:
            pass
    m = LogisticRegression(C=100, max_iter=10000).fit(X, y); m._fallback = True
    return m


# ---------------------------------------------------------------- XGBoost
BOUNDS = dict(n_estimators=(100, 800), learning_rate=(0.01, 0.3), max_depth=(2, 6), min_child_weight=(1, 20),
              subsample=(0.5, 1.0), colsample_bytree=(0.5, 1.0), reg_alpha=(0.001, 10), reg_lambda=(0.01, 100))
LOG_SCALE = {'learning_rate', 'min_child_weight', 'reg_alpha', 'reg_lambda'}
HARD = dict(n_estimators=(20, 3000), learning_rate=(0.001, 1.0), max_depth=(1, 12), min_child_weight=(0.1, 200),
            subsample=(0.2, 1.0), colsample_bytree=(0.2, 1.0), reg_alpha=(1e-6, 1000), reg_lambda=(1e-5, 10000))

def sample_configs(n, seed, bounds=BOUNDS):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = {}
        for k, (lo, hi) in bounds.items():
            if k in LOG_SCALE:
                c[k] = float(10 ** r.uniform(np.log10(lo), np.log10(hi)))
            elif k in ('n_estimators', 'max_depth'):
                c[k] = int(r.integers(int(lo), int(hi) + 1))
            else:
                c[k] = float(r.uniform(lo, hi))
        out.append(c)
    return out
CONFIGS = sample_configs(N_CONFIGS, SEED)

def on_boundary(cfg, bounds=BOUNDS):
    """Parameters whose best value lies within 5% of a search-range edge that can still be extended."""
    hits = []
    for k, (lo, hi) in bounds.items():
        v = cfg[k]
        f = lambda x: np.log10(x) if k in LOG_SCALE else x
        span = f(hi) - f(lo)
        if (f(v) - f(lo)) / span < 0.05 and lo > HARD[k][0]:
            hits.append((k, 'low'))
        elif (f(hi) - f(v)) / span < 0.05 and hi < HARD[k][1]:
            hits.append((k, 'high'))
    return hits

def extend_bounds(hits, bounds=BOUNDS):
    nb = dict(bounds)
    for k, side in hits:
        lo, hi = nb[k]
        if k in LOG_SCALE:
            lo, hi = (lo / 10, hi) if side == 'low' else (lo, hi * 10)
        else:
            w = hi - lo
            lo, hi = (lo - w / 2, hi) if side == 'low' else (lo, hi + w / 2)
        lo, hi = max(lo, HARD[k][0]), min(hi, HARD[k][1])
        if k in ('n_estimators', 'max_depth'):
            lo, hi = int(round(lo)), int(round(hi))
        nb[k] = (lo, hi)
    return nb

def make_xgb(cfg):
    return xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', n_jobs=2, random_state=SEED,
                             eval_metric='logloss', **cfg)

def _search(X, y, folds, configs, offset=0):
    rows, best, best_oof = [], None, None
    for i, cfg in enumerate(configs):
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = make_xgb(cfg).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ll = log_loss(y, oof)
        rows.append(dict(config=offset + i, logloss=ll, **cfg))
        if best is None or ll < best[0]:
            best, best_oof = (ll, cfg), oof
    return best, best_oof, rows

def tune_xgb(X, y, seed):
    """Random search (5-fold CV log loss). Registered rule: if the best value of a parameter lies on a
    search-range edge, the range is extended once (50 further configurations) and the overall best is kept."""
    folds = list(StratifiedKFold(K_INNER, shuffle=True, random_state=seed).split(X, y))
    best, oof, rows = _search(X, y, folds, CONFIGS)
    hits = on_boundary(best[1])
    extension = None
    if hits:
        nb = extend_bounds(hits)
        best2, oof2, rows2 = _search(X, y, folds, sample_configs(N_CONFIGS, seed + 7, nb), offset=N_CONFIGS)
        rows += rows2
        extension = dict(hits=hits, new_bounds={k: list(v) for k, v in nb.items() if v != BOUNDS[k]},
                         improved=bool(best2[0] < best[0]))
        if best2[0] < best[0]:
            best, oof = best2, oof2
    return best[1], oof, rows, extension

def platt_fit(p, y):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return LogisticRegression(penalty=None).fit(z, y)

def platt_apply(m, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return m.predict_proba(z)[:, 1]

# ---------------------------------------------------------------- full pipeline on one train/test split
def fit_pipeline(train, ytr, seed):
    """Impute (10x), tune XGBoost on imputation 1, fit 10 XGB + 10 LR, Platt on tuning OOF. Returns fitted objects."""
    imps, Xtr_list = [], []
    for m in range(N_IMP):
        imp = ChainedImputer(seed + 1000 * m)
        done = imp.fit_transform(train, ytr)
        imps.append(imp); Xtr_list.append(done)
    knots = {v: np.percentile(np.log1p(Xtr_list[0].mvpa_min_week_total) if v == 'log_mvpa' else Xtr_list[0][v], [10, 50, 90]).tolist()
             for v in SPLINE_VARS}
    cfg, oof, table, extension = tune_xgb(Xtr_list[0][PREDICTORS].values, ytr, seed)
    platt = platt_fit(oof, ytr)
    xgbs = [make_xgb(cfg).fit(d[PREDICTORS].values, ytr) for d in Xtr_list]
    lrs = [fit_lr(lr_design(d, knots), ytr) for d in Xtr_list]
    return dict(imputers=imps, knots=knots, cfg=cfg, platt=platt, xgbs=xgbs, lrs=lrs, tuning=table, extension=extension, train_imputed=Xtr_list)

def predict_pipeline(p, new):
    gb, lr, ada, fin = [], [], [], []
    for m in range(N_IMP):
        d = p['imputers'][m].transform(new)
        gb.append(platt_apply(p['platt'], p['xgbs'][m].predict_proba(d[PREDICTORS].values)[:, 1]))
        lr.append(p['lrs'][m].predict_proba(lr_design(d, p['knots']))[:, 1])
        ada.append(d.ada_score.values); fin.append(d.findrisc_adapted.values)
    return dict(gb=np.mean(gb, 0), lr=np.mean(lr, 0), ada_imps=np.array(ada), fin_imps=np.array(fin))


def cutoff_near_sensitivity(score_imps, y, target=0.90):
    """Integer cutoff (score >= c) whose sensitivity, averaged over imputations, is nearest the target."""
    best = None
    for c in range(int(score_imps.min()), int(score_imps.max()) + 1):
        sens = np.mean([((s >= c) & (y == 1)).sum() / (y == 1).sum() for s in score_imps])
        if best is None or abs(sens - target) < abs(best[1] - target):
            best = (c, sens)
    return best

def prob_threshold_near_sensitivity(p, y, target=0.90):
    cases = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(cases)))      # threshold so that ~target of cases are >= it
    thr = cases[k]
    return float(thr), float((p[y == 1] >= thr).mean())

def summarise(name, p, y):
    return dict(model=name, auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, np.clip(p, 0, 1))) if p.max() <= 1 else None)


# ================================================================= main
def main():
    t0 = time.time()
    dev = pd.read_csv(DEV_CSV)
    assert set(dev['sample']) == {'development'}, 'This script must only see development data'
    dev = dev.reset_index(drop=True)
    y = dev[OUTCOME].values.astype(int)
    base = dev[['SEQN'] + COMPLETE + CONT_IMP + BIN_IMP].copy()
    log(f'Development sample: {len(dev)} participants, {y.sum()} cases ({y.mean():.2%}).')
    log(f'Missing items: ' + ', '.join(f'{v} {base[v].isna().mean():.1%}' for v in CONT_IMP + BIN_IMP))

    # ---------------- nested cross-validation: internal performance and out-of-fold predictions for H2
    outer = StratifiedKFold(K_OUTER, shuffle=True, random_state=SEED)
    oof = dict(gb=np.zeros(len(y)), lr=np.zeros(len(y)), ada=np.zeros((N_IMP, len(y))), fin=np.zeros((N_IMP, len(y))))
    fold_info = []
    for f, (tr, te) in enumerate(outer.split(base, y)):
        log(f'Outer fold {f + 1}/{K_OUTER}: fitting (train {len(tr)}, test {len(te)})')
        p = fit_pipeline(base.iloc[tr].reset_index(drop=True), y[tr], SEED + f)
        pr = predict_pipeline(p, base.iloc[te].reset_index(drop=True))
        oof['gb'][te] = pr['gb']; oof['lr'][te] = pr['lr']; oof['ada'][:, te] = pr['ada_imps']; oof['fin'][:, te] = pr['fin_imps']
        fold_info.append(dict(fold=f + 1, best_config=p['cfg'], range_extension=p['extension'],
                              auc_gb=float(roc_auc_score(y[te], pr['gb'])), auc_lr=float(roc_auc_score(y[te], pr['lr'])),
                              lr_fallback=any(getattr(m, '_fallback', False) for m in p['lrs'])))
        log(f'  fold {f + 1}: AUC GB {fold_info[-1]["auc_gb"]:.3f}, LR {fold_info[-1]["auc_lr"]:.3f}; range extended: {bool(p["extension"])}')

    ada_mean = oof['ada'].mean(0); fin_mean = oof['fin'].mean(0)
    internal = [summarise('gradient_boosting', oof['gb'], y), summarise('logistic_regression', oof['lr'], y),
                dict(model='ada_risk_test', auc=float(roc_auc_score(y, ada_mean)), brier=None),
                dict(model='adapted_findrisc', auc=float(roc_auc_score(y, fin_mean)), brier=None)]
    log('Internal (nested CV) AUC: ' + ', '.join(f"{r['model']} {r['auc']:.3f}" for r in internal))

    # ---------------- H2 thresholds from development out-of-fold predictions (locked)
    h2 = {}
    h2['gradient_boosting'] = dict(zip(['threshold', 'dev_sensitivity'], prob_threshold_near_sensitivity(oof['gb'], y)))
    h2['logistic_regression'] = dict(zip(['threshold', 'dev_sensitivity'], prob_threshold_near_sensitivity(oof['lr'], y)))
    c, s = cutoff_near_sensitivity(oof['ada'], y); h2['ada_risk_test'] = dict(threshold=c, dev_sensitivity=float(s))
    c, s = cutoff_near_sensitivity(oof['fin'], y); h2['adapted_findrisc'] = dict(threshold=c, dev_sensitivity=float(s))
    log('H2 locked thresholds: ' + json.dumps(h2))

    # ---------------- final models on the full development sample (locked)
    log('Fitting final locked models on the full development sample')
    final = fit_pipeline(base, y, SEED + 99)
    log(f'Final XGBoost configuration: {final["cfg"]}; range extension: {final["extension"]}')
    joblib.dump(dict(imputers=final['imputers'], knots=final['knots'], cfg=final['cfg'], platt=final['platt'],
                     xgbs=final['xgbs'], lrs=final['lrs'], predictors=PREDICTORS, seed=SEED),
                os.path.join(OUT, 'locked_models.joblib'), compress=3)
    pd.DataFrame(final['tuning']).sort_values('logloss').to_csv(os.path.join(OUT, 'xgb_tuning_final.csv'), index=False)
    pd.DataFrame(dict(SEQN=dev.SEQN, y=y, gb=oof['gb'], lr=oof['lr'], ada=ada_mean, findrisc=fin_mean)).to_csv(
        os.path.join(OUT, 'development_oof_predictions.csv'), index=False)

    lr_coef = final['lrs'][0]
    report = dict(
        created=time.strftime('%Y-%m-%d %H:%M'), seed=SEED, n=int(len(y)), cases=int(y.sum()),
        settings=dict(imputations=N_IMP, chained_iterations=N_ITER, pmm_donors=PMM_DONORS, xgb_configs=N_CONFIGS,
                      inner_folds=K_INNER, outer_folds=K_OUTER),
        predictors=PREDICTORS, spline_knots=final['knots'], final_xgb_config=final['cfg'],
        final_range_extension=final['extension'],
        lr_fallback_used=any(getattr(m, '_fallback', False) for m in final['lrs']),
        internal_performance_nested_cv=internal, outer_folds=fold_info, h2_locked_thresholds=h2,
        runtime_minutes=round((time.time() - t0) / 60, 1))
    json.dump(report, open(os.path.join(OUT, 'development_report.json'), 'w'), indent=2)
    open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(LOG) + '\n')
    log(f'Done in {report["runtime_minutes"]} min')

if __name__ == '__main__':
    main()
