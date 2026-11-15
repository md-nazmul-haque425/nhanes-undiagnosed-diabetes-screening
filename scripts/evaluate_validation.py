import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV_CSV, VAL_CSV, VAL2_CSV, MODELS, RES = sys.argv[1:6]
os.makedirs(RES, exist_ok=True)
# reuse the exact development pipeline (classes are needed to load the locked models)
sys.argv = [sys.argv[0], DEV_CSV, os.path.join(RES, '_dev_tmp')]
import fit_development_models as fdm
from fit_development_models import (ChainedImputer, finalise, lr_design, platt_apply, PREDICTORS, CONT_IMP, BIN_IMP,
                                    COMPLETE, N_IMP)
import __main__
__main__.ChainedImputer = ChainedImputer
warnings.filterwarnings('ignore')

B = 2000
SEED = 20261015
rng = np.random.default_rng(SEED)
OUT = {}
LOG = []
def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True); LOG.append(line)

locked = joblib.load(os.path.join(MODELS, 'locked_models.joblib'))
dev_report = json.load(open(os.path.join(MODELS, 'development_report.json')))
H2 = dev_report['h2_locked_thresholds']

# ------------------------------------------------------------------ prediction helpers
def predict_all(model, df):
    """Per-imputation predictions for GB, LR and both scores (validation outcomes are not used)."""
    base = df[['SEQN'] + COMPLETE + CONT_IMP + BIN_IMP].reset_index(drop=True)
    gb, lr, ada, fin = [], [], [], []
    for m in range(N_IMP):
        d = model['imputers'][m].transform(base)
        gb.append(platt_apply(model['platt'], model['xgbs'][m].predict_proba(d[model['predictors']].values)[:, 1]))
        lr.append(model['lrs'][m].predict_proba(lr_design(d, model['knots']))[:, 1])
        ada.append(d.ada_score.values); fin.append(d.findrisc_adapted.values)
    return dict(gb=np.array(gb), lr=np.array(lr), ada=np.array(ada, float), fin=np.array(fin, float))

def refer_top(p, k, tie_rng):
    """Refer exactly k participants with the highest p; ties broken at random with a fixed seed."""
    order = np.lexsort((tie_rng.random(len(p)), -p))
    out = np.zeros(len(p), bool); out[order[:k]] = True
    return out

def nearest_cutoff_rate(score, rate):
    """Integer cutoff (score >= c) whose referral rate is nearest `rate`."""
    best = None
    for c in np.arange(np.floor(score.min()), np.ceil(score.max()) + 1):
        rr = (score >= c).mean()
        if best is None or abs(rr - rate) < abs(best[1] - rate):
            best = (float(c), rr)
    return best

def fnr(ref, y): return float((~ref[y == 1]).mean())
def mcnemar_exact(a_detect, b_detect):
    """Exact McNemar on paired binary outcomes; returns b (a only), c (b only), p."""
    b = int((a_detect & ~b_detect).sum()); c = int((~a_detect & b_detect).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj.tolist()

def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

def strat_boot_idx(y, r):
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    return np.concatenate([r.choice(i1, len(i1)), r.choice(i0, len(i0))])

def wilson(x, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def newcombe_diff(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

def delong_auc_ci(y, s):
    """DeLong 95% CI for AUC."""
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    v = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    auc = v.mean(); v10 = v.mean(1); v01 = v.mean(0)
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return float(auc), [float(auc - 1.96 * se), float(auc + 1.96 * se)]

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

# ------------------------------------------------------------------ load data and predict
val = pd.read_csv(VAL_CSV).reset_index(drop=True)
assert set(val['sample']) == {'validation'}
y = val['y_hba1c'].values.astype(int)
n = len(y)
log(f'Validation 2017-March 2020: {n} participants, {y.sum()} cases')
P = predict_all(locked, val)
gb, lr = P['gb'].mean(0), P['lr'].mean(0)
ada, fin = P['ada'].mean(0), P['fin'].mean(0)
pd.DataFrame(dict(SEQN=val.SEQN, y=y, gb=gb, lr=lr, ada=ada, findrisc=fin)).to_csv(os.path.join(RES, 'validation_predictions.csv'), index=False)

# ================================================================== H1
def h1_classify(gb, lr, ada, fin, tie_seed):
    tr = np.random.default_rng(tie_seed)
    ref_ada = ada >= 5
    r = ref_ada.mean(); k = int(round(r * len(gb)))
    ref_gb = refer_top(gb, k, tr); ref_lr = refer_top(lr, k, tr)
    c_fin, r_fin = nearest_cutoff_rate(fin, r)
    ref_fin = fin >= c_fin
    return dict(r=r, k=k, fin_cutoff=c_fin, ref=dict(gb=ref_gb, lr=ref_lr, ada=ref_ada, fin=ref_fin))

H1 = h1_classify(gb, lr, ada, fin, SEED)
ref = H1['ref']
comparators = ['ada', 'fin', 'lr']
names = dict(gb='Gradient boosting', lr='Logistic regression', ada='ADA Risk Test', fin='Adapted FINDRISC')
h1 = dict(referral_rate_target=float(H1['r']), n_referred=H1['k'], findrisc_cutoff=H1['fin_cutoff'],
          referral_rates={s: float(ref[s].mean()) for s in ref}, fnr={s: fnr(ref[s], y) for s in ref},
          missed={s: int((~ref[s][y == 1]).sum()) for s in ref}, comparisons={})
pvals = []
for c in comparators:
    b, cc, p = mcnemar_exact(ref['gb'][y == 1], ref[c][y == 1])
    h1['comparisons'][c] = dict(fnr_difference=h1['fnr']['gb'] - h1['fnr'][c], gb_only_detected=b, comparator_only_detected=cc, p_exact_mcnemar=p)
    pvals.append(p)
for c, pa in zip(comparators, holm(pvals)):
    h1['comparisons'][c]['p_holm'] = pa

# bootstrap: participants (stratified by outcome) + random imputation per replicate
diffs = {c: [] for c in comparators}; fnr_b = {s: [] for s in ['gb', 'lr', 'ada', 'fin']}
for b in range(B):
    idx = strat_boot_idx(y, rng); m = rng.integers(N_IMP)
    cl = h1_classify(P['gb'][m][idx], P['lr'][m][idx], P['ada'][m][idx], P['fin'][m][idx], SEED + b)
    yb = y[idx]
    f = {s: fnr(cl['ref'][s], yb) for s in cl['ref']}
    for s in f: fnr_b[s].append(f[s])
    for c in comparators: diffs[c].append(f['gb'] - f[c])
for c in comparators:
    h1['comparisons'][c]['ci95_bootstrap'] = ci(diffs[c])
h1['fnr_ci95'] = {s: ci(fnr_b[s]) for s in fnr_b}
SESOI = -0.05
for c in comparators:
    cmp_ = h1['comparisons'][c]; lo, hi = cmp_['ci95_bootstrap']
    cmp_['supported'] = bool(cmp_['p_holm'] < 0.05 and cmp_['fnr_difference'] < 0)
    cmp_['ci_interpretation'] = ('entirely below -5 points (clinically meaningful improvement)' if hi < SESOI else
                                 'excludes 0 but not entirely below -5 points' if hi < 0 else
                                 'includes 0 and excludes -5 points (improvement of SESOI size ruled out)' if lo > SESOI else
                                 'includes 0 and -5 points (inconclusive)' if lo <= SESOI <= hi or lo <= 0 <= hi else 'favours comparator')
OUT['H1'] = h1
log('H1 FNR: ' + ', '.join(f'{names[s]} {h1["fnr"][s]:.3f}' for s in h1['fnr']) + f' at referral {H1["r"]:.3f}')
for c in comparators:
    cc = h1['comparisons'][c]
    log(f'  GB vs {names[c]}: diff {cc["fnr_difference"]:+.3f} CI {cc["ci95_bootstrap"][0]:+.3f} to {cc["ci95_bootstrap"][1]:+.3f}; p {cc["p_exact_mcnemar"]:.4f}, Holm {cc["p_holm"]:.4f}')

# ================================================================== H2
thr = dict(gb=H2['gradient_boosting']['threshold'], lr=H2['logistic_regression']['threshold'],
           ada=H2['ada_risk_test']['threshold'], fin=H2['adapted_findrisc']['threshold'])
score = dict(gb=gb, lr=lr, ada=ada, fin=fin)
ref2 = {s: score[s] >= thr[s] for s in score}
def opchar(refx, yy):
    tp = (refx & (yy == 1)).sum(); fp = (refx & (yy == 0)).sum()
    return dict(sensitivity=float(tp / (yy == 1).sum()), specificity=float((~refx & (yy == 0)).sum() / (yy == 0).sum()),
                referral_rate=float(refx.mean()), tests_per_case=float(refx.sum() / tp) if tp else None,
                npv=float((~refx & (yy == 0)).sum() / max((~refx).sum(), 1)), ppv=float(tp / max(refx.sum(), 1)))
h2 = dict(thresholds=thr, validation={s: opchar(ref2[s], y) for s in ref2}, comparisons={})
pv = []
for c in comparators:
    b, cc, p = mcnemar_exact(~ref2['gb'][y == 0], ~ref2[c][y == 0])
    h2['comparisons'][c] = dict(specificity_difference=h2['validation']['gb']['specificity'] - h2['validation'][c]['specificity'],
                                p_exact_mcnemar=p, gb_only_correct_negative=b, comparator_only_correct_negative=cc)
    pv.append(p)
for c, pa in zip(comparators, holm(pv)): h2['comparisons'][c]['p_holm'] = pa
d2 = {c: [] for c in comparators}
for b in range(B):
    idx = strat_boot_idx(y, rng); m = rng.integers(N_IMP); yb = y[idx]
    sc = dict(gb=P['gb'][m][idx], lr=P['lr'][m][idx], ada=P['ada'][m][idx], fin=P['fin'][m][idx])
    sp = {s: ((sc[s] < thr[s]) & (yb == 0)).sum() / (yb == 0).sum() for s in sc}
    for c in comparators: d2[c].append(sp['gb'] - sp[c])
for c in comparators:
    h2['comparisons'][c]['ci95_bootstrap'] = ci(d2[c])
    h2['comparisons'][c]['supported'] = bool(h2['comparisons'][c]['p_holm'] < 0.05 and h2['comparisons'][c]['specificity_difference'] > 0)
# descriptive: specificity at exactly ~90% sensitivity in validation
def spec_at_sens(s, yy, target=0.9):
    cases = np.sort(s[yy == 1]); t = cases[int(np.floor((1 - target) * len(cases)))]
    return float(t), float((s[yy == 1] >= t).mean()), float((s[yy == 0] < t).mean())
h2['descriptive_spec_at_90_sens'] = {s: dict(zip(['threshold', 'sensitivity', 'specificity'], spec_at_sens(score[s], y))) for s in score}
OUT['H2'] = h2
log('H2 validation: ' + '; '.join(f'{names[s]} sens {h2["validation"][s]["sensitivity"]:.3f} spec {h2["validation"][s]["specificity"]:.3f}' for s in score))

# ================================================================== calibration
def calib(p, yy):
    lp = logit(p)
    citl = sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(yy, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    return float(citl), float(slope), float(brier_score_loss(yy, p))
cal = {}
for s in ['gb', 'lr']:
    p = score[s]; citl, slope, br = calib(p, y)
    prec = 1 / (1 + np.exp(-(logit(p) + citl)))
    citl_r, slope_r, br_r = calib(prec, y)
    bs = []
    for b in range(1000):
        idx = strat_boot_idx(y, rng); bs.append(calib(p[idx], y[idx]))
    bs = np.array(bs)
    cal[s] = dict(locked=dict(calibration_in_the_large=citl, slope=slope, brier=br, ci95_citl=ci(bs[:, 0]), ci95_slope=ci(bs[:, 1]), ci95_brier=ci(bs[:, 2]),
                              mean_predicted=float(p.mean()), observed=float(y.mean())),
                  intercept_recalibrated=dict(calibration_in_the_large=citl_r, slope=slope_r, brier=br_r, intercept_shift=citl))
OUT['calibration'] = cal
fig, ax = plt.subplots(figsize=(5, 5))
for s, col in [('gb', '#1f4e79'), ('lr', '#c55a11')]:
    p = score[s]; lw = sm.nonparametric.lowess(y, p, frac=0.4, it=0, return_sorted=True)
    ax.plot(lw[:, 0], lw[:, 1], color=col, label=names[s])
    q = pd.qcut(p, 10, duplicates='drop'); g = pd.DataFrame(dict(p=p, y=y, q=q)).groupby('q', observed=True).mean()
    ax.scatter(g.p, g.y, color=col, s=14)
mx = max(np.percentile(gb, 99), np.percentile(lr, 99), 0.1)
ax.plot([0, mx], [0, mx], ls='--', color='grey', lw=1); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel('Predicted risk'); ax.set_ylabel('Observed proportion'); ax.set_title('Calibration, validation 2017–March 2020'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_calibration.png'), dpi=200); plt.close(fig)
log(f'Calibration GB: CITL {cal["gb"]["locked"]["calibration_in_the_large"]:+.3f}, slope {cal["gb"]["locked"]["slope"]:.3f}')

# ================================================================== H3 (descriptive)
asian = (val.race_eth == 'NH Asian').values; white = (val.race_eth == 'NH White').values
def h3_stats(refx, yy, a, w):
    ma = int((~refx & (yy == 1) & a).sum()); na = int(((yy == 1) & a).sum())
    mw = int((~refx & (yy == 1) & w).sum()); nw = int(((yy == 1) & w).sum())
    d, lo, hi = newcombe_diff(ma, na, mw, nw)
    return dict(asian_missed=ma, asian_cases=na, asian_fnr=ma / na, white_missed=mw, white_cases=nw, white_fnr=mw / nw,
                difference=float(d), ci95_newcombe=[float(lo), float(hi)])
h3 = dict(validation=h3_stats(ref['gb'], y, asian, white))
bd = []
for b in range(B):
    idx = np.concatenate([rng.choice(np.where(g)[0], g.sum()) for g in [(y == 1) & asian, (y == 1) & white, (y == 1) & ~asian & ~white, y == 0]])
    m = rng.integers(N_IMP)
    cl = h1_classify(P['gb'][m][idx], P['lr'][m][idx], P['ada'][m][idx], P['fin'][m][idx], SEED + 5000 + b)
    yb, ab, wb = y[idx], asian[idx], white[idx]
    rf = cl['ref']['gb']
    bd.append((~rf & (yb == 1) & ab).sum() / ((yb == 1) & ab).sum() - (~rf & (yb == 1) & wb).sum() / ((yb == 1) & wb).sum())
h3['validation']['ci95_bootstrap'] = ci(bd)
h3['validation_by_strategy'] = {s: h3_stats(ref[s], y, asian, white) for s in ref}
# supplementary: development out-of-fold predictions at the development ADA referral rate
dev = pd.read_csv(DEV_CSV).reset_index(drop=True)
oof = pd.read_csv(os.path.join(MODELS, 'development_oof_predictions.csv'))
assert (oof.SEQN.values == dev.SEQN.values).all()
ydev = oof.y.values; r_dev = (oof.ada >= 5).mean()
ref_dev = refer_top(oof.gb.values, int(round(r_dev * len(ydev))), np.random.default_rng(SEED))
h3['development_oof'] = h3_stats(ref_dev, ydev, (dev.race_eth == 'NH Asian').values, (dev.race_eth == 'NH White').values)
h3['development_oof']['referral_rate'] = float(r_dev)
OUT['H3'] = h3
log(f'H3 validation FNR Asian {h3["validation"]["asian_fnr"]:.3f} ({h3["validation"]["asian_missed"]}/{h3["validation"]["asian_cases"]}) vs White {h3["validation"]["white_fnr"]:.3f}; diff {h3["validation"]["difference"]:+.3f}')

# ================================================================== H4
def per1000(refx, yy): return 1000 * (refx & (yy == 1)).sum() / max(refx.sum(), 1)
h4 = dict(universal_testing=float(1000 * y.mean()), per_1000={s: float(per1000(ref[s], y)) for s in ref}, comparisons={})
pv = []
for c in ['ada', 'fin']:
    b, cc, p = mcnemar_exact(ref['gb'][y == 1], ref[c][y == 1]); pv.append(p)
    h4['comparisons'][c] = dict(difference=h4['per_1000']['gb'] - h4['per_1000'][c], p_exact_mcnemar=p)
for c, pa in zip(['ada', 'fin'], holm(pv)): h4['comparisons'][c]['p_holm'] = pa
d4 = {c: [] for c in ['ada', 'fin']}
for b in range(B):
    idx = strat_boot_idx(y, rng); m = rng.integers(N_IMP)
    cl = h1_classify(P['gb'][m][idx], P['lr'][m][idx], P['ada'][m][idx], P['fin'][m][idx], SEED + 9000 + b)
    for c in d4: d4[c].append(per1000(cl['ref']['gb'], y[idx]) - per1000(cl['ref'][c], y[idx]))
for c in d4:
    h4['comparisons'][c]['ci95_bootstrap'] = ci(d4[c])
    h4['comparisons'][c]['supported'] = bool(h4['comparisons'][c]['p_holm'] < 0.05 and h4['comparisons'][c]['difference'] > 0)
curve = []
tr_ = np.random.default_rng(SEED)
for rr in np.arange(0.10, 0.701, 0.05):
    k = int(round(rr * n))
    row = dict(referral_rate=round(float(rr), 2))
    for s in ['gb', 'lr']:
        rf = refer_top(score[s], k, tr_); row[f'{s}_per_1000'] = float(per1000(rf, y)); row[f'{s}_sensitivity'] = float(rf[y == 1].mean())
    curve.append(row)
for s in ['ada', 'fin']:
    for c in np.arange(np.floor(score[s].min()), np.ceil(score[s].max()) + 1):
        rf = score[s] >= c
        if 0.05 <= rf.mean() <= 0.75:
            curve.append(dict(referral_rate=float(rf.mean()), **{f'{s}_cutoff': float(c), f'{s}_per_1000': float(per1000(rf, y)), f'{s}_sensitivity': float(rf[y == 1].mean())}))
pd.DataFrame(curve).to_csv(os.path.join(RES, 'h4_referral_curve.csv'), index=False)
h4['curve_file'] = 'h4_referral_curve.csv'
OUT['H4'] = h4
log('H4 per 1000 tests: ' + ', '.join(f'{names[s]} {h4["per_1000"][s]:.1f}' for s in ref) + f'; universal {h4["universal_testing"]:.1f}')

# ================================================================== robustness for H1
rob = {}
items = ['bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_total']
cc_mask = val[items].notna().all(axis=1).values
def h1_point(mask, gbx, lrx, adax, finx, yy):
    cl = h1_classify(gbx[mask], lrx[mask], adax[mask], finx[mask], SEED)
    f = {s: fnr(cl['ref'][s], yy[mask]) for s in cl['ref']}
    res = dict(n=int(mask.sum()), cases=int(yy[mask].sum()), referral_rate=float(cl['r']), fnr=f, differences={c: f['gb'] - f[c] for c in comparators}, mcnemar_p={})
    for c in comparators:
        res['mcnemar_p'][c] = mcnemar_exact(cl['ref']['gb'][yy[mask] == 1], cl['ref'][c][yy[mask] == 1])[2]
    return res, cl
rob['complete_case'], _ = h1_point(cc_mask, gb, lr, ada, fin, y)
v0 = val.copy(); v0.loc[v0.gestational_diabetes.isna() & (v0.male == 0), 'gestational_diabetes'] = 0.0
P0 = predict_all(locked, v0)
rob['gdm_missing_as_no'], _ = h1_point(np.ones(n, bool), P0['gb'].mean(0), P0['lr'].mean(0), P0['ada'].mean(0), P0['fin'].mean(0), y)

def weighted_diff(refa, refb, yy, w, psu, strata):
    """Design-based (Taylor linearisation) estimate of weighted FNR difference among cases."""
    cases = yy == 1
    ma = (~refa).astype(float); mb = (~refb).astype(float)
    W = (w * cases).sum()
    fa = (w * cases * ma).sum() / W; fb = (w * cases * mb).sum() / W
    d = fa - fb
    z = w * cases * ((ma - mb) - d) / W
    df = pd.DataFrame(dict(z=z, psu=psu, st=strata)).groupby(['st', 'psu']).z.sum().reset_index()
    var = 0.0
    for _, g in df.groupby('st'):
        k = len(g)
        if k > 1: var += k / (k - 1) * ((g.z - g.z.mean()) ** 2).sum()
    se = np.sqrt(var)
    return float(fa), float(fb), float(d), [float(d - 1.96 * se), float(d + 1.96 * se)]
w = val.wt_mec.values; psu = val.psu.values; st = val.strata.values
rob['weighted'] = {'note': 'Same unweighted matched-referral classification; weighted FNR among cases, Taylor linearisation SE'}
for c in comparators:
    fa, fb, d, cci = weighted_diff(ref['gb'], ref[c], y, w, psu, st)
    rob['weighted'][c] = dict(gb_fnr=fa, comparator_fnr=fb, difference=d, ci95=cci)
# sensitivity outcome (fasting subsample)
fmask = (val.in_fasting_subsample == 1).values
y2 = np.nan_to_num(val.y_hba1c_or_fpg.values).astype(int)
r2, cl2 = h1_point(fmask, gb, lr, ada, fin, y2)
r2['weighted'] = {}
for c in comparators:
    fa, fb, d, cci = weighted_diff(cl2['ref']['gb'], cl2['ref'][c], y2[fmask], val.wt_fasting.values[fmask], psu[fmask], st[fmask])
    r2['weighted'][c] = dict(gb_fnr=fa, comparator_fnr=fb, difference=d, ci95=cci)
rob['sensitivity_outcome_hba1c_or_fpg'] = r2
rob['internal_nested_cv'] = dev_report['internal_performance_nested_cv']
OUT['robustness'] = rob
log('Robustness complete-case diffs: ' + json.dumps({c: round(v, 3) for c, v in rob['complete_case']['differences'].items()}))

# ================================================================== supporting metrics
sup = {}
for s in score:
    auc, aci = delong_auc_ci(y, score[s])
    sup[s] = dict(auc=auc, auc_ci95_delong=aci, pr_auc=float(average_precision_score(y, score[s])),
                  npv_at_h1=float((~ref[s] & (y == 0)).sum() / (~ref[s]).sum()), ppv_at_h1=float((ref[s] & (y == 1)).sum() / ref[s].sum()))
dca = []
for pt in np.arange(0.01, 0.301, 0.01):
    row = dict(threshold=round(float(pt), 2), treat_all=float(y.mean() - (1 - y.mean()) * pt / (1 - pt)))
    for s in ['gb', 'lr']:
        rf = score[s] >= pt; row[s] = float(((rf & (y == 1)).sum() - (rf & (y == 0)).sum() * pt / (1 - pt)) / n)
    for s in ['ada', 'fin']:
        rf = ref[s]; row[s] = float(((rf & (y == 1)).sum() - (rf & (y == 0)).sum() * pt / (1 - pt)) / n)
    dca.append(row)
pd.DataFrame(dca).to_csv(os.path.join(RES, 'decision_curve.csv'), index=False)
# threshold decision table: thresholds chosen in development out-of-fold predictions
tab = []
for s, col in [('gb', 'gb'), ('lr', 'lr')]:
    po = oof[col].values; cand = np.unique(np.quantile(po, np.linspace(0.01, 0.99, 400)))
    def stats_at(t, pp, yy):
        rf = pp >= t; tp = (rf & (yy == 1)).sum(); fp = (rf & (yy == 0)).sum(); fn = (~rf & (yy == 1)).sum(); tn = (~rf & (yy == 0)).sum()
        return tp, fp, fn, tn
    f1 = max(cand, key=lambda t: (lambda a: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1))(stats_at(t, po, ydev)))
    you = max(cand, key=lambda t: (lambda a: a[0] / (a[0] + a[2]) + a[3] / (a[3] + a[1]) - 1)(stats_at(t, po, ydev)))
    cost = min(cand, key=lambda t: (lambda a: 10 * a[2] + a[1])(stats_at(t, po, ydev)))
    rules = {'F1-optimal': f1, 'Youden': you, 'Cost-based (FN:FP = 10:1)': cost,
             'Fixed sensitivity 90% (H2)': thr[s], 'Fixed maximum FNR 10% (same as 90% sensitivity)': thr[s]}
    for rule, t in rules.items():
        oc = opchar(score[s] >= t, y); tab.append(dict(model=names[s], rule=rule, threshold=float(t), **oc))
    oc = opchar(ref[s], y); tab.append(dict(model=names[s], rule='Matched referral (H1)', threshold=None, **oc))
pd.DataFrame(tab).to_csv(os.path.join(RES, 'threshold_decision_table.csv'), index=False)
OUT['supporting'] = dict(discrimination=sup, decision_curve_file='decision_curve.csv', threshold_table_file='threshold_decision_table.csv')
fig, ax = plt.subplots(figsize=(6, 4))
dd = pd.DataFrame(dca)
for s, colr in [('gb', '#1f4e79'), ('lr', '#c55a11'), ('ada', '#548235'), ('fin', '#7f6000')]:
    ax.plot(dd.threshold, dd[s], label=names[s], color=colr)
ax.plot(dd.threshold, dd.treat_all, label='Test everyone', color='grey', ls='--'); ax.axhline(0, color='black', lw=0.8, label='Test no one')
ax.set_ylim(-0.005, y.mean() * 1.1); ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_decision_curve.png'), dpi=200); plt.close(fig)

# ================================================================== exploratory
exp = {}
# E1
cases = y == 1
fn_ = cases & ~ref['gb']; tp_ = cases & ref['gb']
def smd(a, b): return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
e1 = {}
for grp, mask in [('all', np.ones(n, bool)), ('NH Asian', asian), ('NH White', white)]:
    g = {}
    for v in ['age', 'bmi', 'waist_cm']:
        a = val.loc[fn_ & mask, v].dropna(); b_ = val.loc[tp_ & mask, v].dropna()
        g[v] = dict(fn_median=float(a.median()) if len(a) else None, tp_median=float(b_.median()) if len(b_) else None,
                    fn_n=int(len(a)), tp_n=int(len(b_)), smd=smd(a, b_) if len(a) > 1 and len(b_) > 1 else None,
                    mann_whitney_p=float(stats.mannwhitneyu(a, b_).pvalue) if len(a) > 0 and len(b_) > 0 else None)
    e1[grp] = g
exp['E1'] = e1
# E2: group-specific thresholds chosen in development OOF to equalise Asian and White FNR at the development referral rate
ad, wd = (dev.race_eth == 'NH Asian').values, (dev.race_eth == 'NH White').values
po = oof.gb.values; kdev = int(round(r_dev * len(po)))
best = None
for qa in np.linspace(0.05, 0.95, 181):
    ta = np.quantile(po[ad], qa)
    rest_k = kdev - (po[ad] >= ta).sum()
    if rest_k <= 0: continue
    to = np.sort(po[~ad])[::-1][min(rest_k, (~ad).sum()) - 1]
    rf = np.where(ad, po >= ta, po >= to)
    fa = (~rf & (ydev == 1) & ad).sum() / ((ydev == 1) & ad).sum(); fw = (~rf & (ydev == 1) & wd).sum() / ((ydev == 1) & wd).sum()
    if best is None or abs(fa - fw) < best[0]: best = (abs(fa - fw), ta, to)
_, ta, to = best
rf_v = np.where(asian, gb >= ta, gb >= to)
e2 = dict(dev_thresholds=dict(asian=float(ta), others=float(to)), common=h3_stats(ref['gb'], y, asian, white),
          group_specific=h3_stats(rf_v, y, asian, white), common_opchar=opchar(ref['gb'], y), group_specific_opchar=opchar(rf_v, y))
exp['E2'] = e2
# subgroup FNRs
sub = []
groups = {'Sex: male': val.male == 1, 'Sex: female': val.male == 0, 'Age 20-44': val.age < 45, 'Age 45-64': (val.age >= 45) & (val.age < 65), 'Age 65+': val.age >= 65,
          'BMI <25': val.bmi < 25, 'BMI 25-29.9': (val.bmi >= 25) & (val.bmi < 30), 'BMI >=30': val.bmi >= 30}
for r_ in val.race_eth.dropna().unique(): groups[f'Race/ethnicity: {r_}'] = val.race_eth == r_
for gname, gm in groups.items():
    gm = gm.values
    row = dict(subgroup=gname, cases=int((cases & gm).sum()))
    for s in ref: row[f'fnr_{s}'] = float((~ref[s] & cases & gm).sum() / max((cases & gm).sum(), 1))
    sub.append(row)
pd.DataFrame(sub).to_csv(os.path.join(RES, 'subgroup_fnr.csv'), index=False)
exp['subgroups_file'] = 'subgroup_fnr.csv'
# SHAP (TreeSHAP, gradient boosting, imputation 1)
import xgboost as xgb
base = val[['SEQN'] + COMPLETE + CONT_IMP + BIN_IMP].reset_index(drop=True)
d1 = locked['imputers'][0].transform(base)
contrib = locked['xgbs'][0].get_booster().predict(xgb.DMatrix(d1[PREDICTORS].values, feature_names=PREDICTORS), pred_contribs=True)[:, :-1]
shap_df = pd.DataFrame(contrib, columns=PREDICTORS)
imp = shap_df.abs().mean().sort_values(ascending=False)
imp_fn = shap_df[fn_].abs().mean().reindex(imp.index)
pd.DataFrame(dict(mean_abs_shap_all=imp, mean_abs_shap_false_negatives=imp_fn, mean_shap_false_negatives=shap_df[fn_].mean().reindex(imp.index))).to_csv(os.path.join(RES, 'shap_importance.csv'))
fig, ax = plt.subplots(figsize=(6, 4)); ax.barh(imp.index[::-1], imp.values[::-1], color='#1f4e79'); ax.set_xlabel('Mean |SHAP| (log-odds)'); ax.set_title('Gradient boosting: global importance')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_shap_importance.png'), dpi=200); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for a_, v in zip(axs, imp.index[:3]):
    a_.scatter(d1[v], shap_df[v], s=3, alpha=0.4, color='#1f4e79'); a_.set_xlabel(v); a_.set_ylabel('SHAP')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_shap_dependence.png'), dpi=200); plt.close(fig)
exp['shap_top'] = imp.head(6).round(4).to_dict()
log('SHAP top: ' + ', '.join(f'{k} {v:.3f}' for k, v in imp.head(5).items()))

# E3: reduced models refitted on development, evaluated on 2021-2023
log('E3: fitting reduced-predictor models on development')
RED = ['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_recreation']
def reduced_frame(df):
    d = df[['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_recreation']].copy()
    d = d.rename(columns={'mvpa_min_week_recreation': 'mvpa_min_week_total'})
    d['family_history'] = 0.0; d['gestational_diabetes'] = 0.0   # constant -> carries no information
    return d
fdm.PREDICTORS = ['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_total']
fdm.BIN_IMP = ['hypertension_dx', 'bp_medication', 'told_prediabetes', 'active_recreation']
_orig_lr_design = fdm.lr_design
def lr_design_red(d, knots):
    d = d.copy(); d['log_mvpa'] = np.log1p(d.mvpa_min_week_total)
    parts = [fdm.rcs_basis(d[v].values.astype(float), knots[v]) for v in fdm.SPLINE_VARS]
    parts.append(d[['male', 'hypertension_dx', 'bp_medication', 'told_prediabetes', 'active_recreation']].values.astype(float))
    return np.column_stack(parts)
fdm.lr_design = lr_design_red
devr = reduced_frame(dev)
red = fdm.fit_pipeline(devr, dev.y_hba1c.values.astype(int), SEED + 300)
val2 = pd.read_csv(VAL2_CSV).reset_index(drop=True)
y3 = val2.y_hba1c.values.astype(int)
v2r = reduced_frame(val2)
gb3, lr3 = [], []
for m in range(N_IMP):
    d = red['imputers'][m].transform(v2r)
    gb3.append(platt_apply(red['platt'], red['xgbs'][m].predict_proba(d[fdm.PREDICTORS].values)[:, 1]))
    lr3.append(red['lrs'][m].predict_proba(lr_design_red(d, red['knots']))[:, 1])
gb3, lr3 = np.array(gb3), np.array(lr3)
k3 = int(round(H1['r'] * len(y3)))
tr3 = np.random.default_rng(SEED)
rg, rl = refer_top(gb3.mean(0), k3, tr3), refer_top(lr3.mean(0), k3, tr3)
b_, c_, p_ = mcnemar_exact(rg[y3 == 1], rl[y3 == 1])
bd3 = []
for b in range(B):
    idx = strat_boot_idx(y3, rng); m = rng.integers(N_IMP); kk = int(round(H1['r'] * len(idx))); trb = np.random.default_rng(SEED + 20000 + b)
    bd3.append(fnr(refer_top(gb3[m][idx], kk, trb), y3[idx]) - fnr(refer_top(lr3[m][idx], kk, trb), y3[idx]))
exp['E3'] = dict(n=int(len(y3)), cases=int(y3.sum()), referral_rate=float(H1['r']), fnr_gb=fnr(rg, y3), fnr_lr=fnr(rl, y3),
                 difference=fnr(rg, y3) - fnr(rl, y3), ci95_bootstrap=ci(bd3), p_exact_mcnemar=p_,
                 auc_gb=float(roc_auc_score(y3, gb3.mean(0))), auc_lr=float(roc_auc_score(y3, lr3.mean(0))),
                 reduced_xgb_config=red['cfg'], range_extension=red['extension'])
fdm.lr_design = _orig_lr_design
OUT['exploratory'] = exp
log(f'E3: FNR GB {exp["E3"]["fnr_gb"]:.3f} vs LR {exp["E3"]["fnr_lr"]:.3f}')

OUT['meta'] = dict(created=time.strftime('%Y-%m-%d %H:%M'), seed=SEED, bootstrap=B, n_validation=n, cases_validation=int(y.sum()))
json.dump(OUT, open(os.path.join(RES, 'validation_results.json'), 'w'), indent=2, default=float)
open(os.path.join(RES, 'run_log.txt'), 'w').write('\n'.join(LOG) + '\n')
log('Done')
