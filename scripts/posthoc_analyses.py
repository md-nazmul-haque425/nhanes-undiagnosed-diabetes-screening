import json, os, sys, time, warnings
import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.metrics import roc_auc_score
DEV_CSV, VAL_CSV, VAL2_CSV, MODELS, RES = sys.argv[1:6]
os.makedirs(RES, exist_ok=True)
sys.argv = [sys.argv[0], DEV_CSV, os.path.join(RES, '_dev_tmp')]
import fit_development_models as fdm
from fit_development_models import ChainedImputer, platt_apply, N_IMP
import __main__; __main__.ChainedImputer = ChainedImputer
warnings.filterwarnings('ignore')
SEED = 20261114; B = 2000
rng = np.random.default_rng(SEED)
OUT = {'meta': dict(seed=SEED, bootstrap=B, created=time.strftime('%Y-%m-%d %H:%M'), status='post hoc, not preregistered')}
def log(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

def refer_top(p, k, tie_rng):
    order = np.lexsort((tie_rng.random(len(p)), -p)); out = np.zeros(len(p), bool); out[order[:k]] = True; return out
def nearest_cutoff_rate(score, rate):
    best = None
    for c in np.arange(np.floor(score.min()), np.ceil(score.max()) + 1):
        rr = (score >= c).mean()
        if best is None or abs(rr - rate) < abs(best[1] - rate): best = (float(c), rr)
    return best
def fnr(ref, y): return float((~ref[y == 1]).mean())
def mcnemar_exact(a, b):
    x = int((a & ~b).sum()); z = int((~a & b).sum())
    return x, z, float(stats.binomtest(x, x + z, 0.5).pvalue if x + z > 0 else 1.0)
def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
def strat_boot_idx(y, r):
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    return np.concatenate([r.choice(i1, len(i1)), r.choice(i0, len(i0))])
def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def newcombe_diff(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2; l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2); d = p1 - p2
    return float(d), float(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), float(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
def h1_classify(gb, lr, ada, fin, tie_seed):
    tr = np.random.default_rng(tie_seed)
    ref_ada = ada >= 5; r = ref_ada.mean(); k = int(round(r * len(gb)))
    ref_gb = refer_top(gb, k, tr); ref_lr = refer_top(lr, k, tr)
    c_fin, _ = nearest_cutoff_rate(fin, r)
    return dict(r=float(r), k=k, fin_cutoff=c_fin, ref=dict(gb=ref_gb, lr=ref_lr, ada=ref_ada, fin=fin >= c_fin))

# ------------------------------------------------------------ locked predictions (same call order as the registered run)
locked = joblib.load(os.path.join(MODELS, 'locked_models.joblib'))
val = pd.read_csv(VAL_CSV).reset_index(drop=True); y = val.y_hba1c.values.astype(int); n = len(y)
base = val[['SEQN'] + fdm.COMPLETE + fdm.CONT_IMP + fdm.BIN_IMP].reset_index(drop=True)
P = dict(gb=[], lr=[], ada=[], fin=[])
for m in range(N_IMP):
    d = locked['imputers'][m].transform(base)
    P['gb'].append(platt_apply(locked['platt'], locked['xgbs'][m].predict_proba(d[locked['predictors']].values)[:, 1]))
    P['lr'].append(locked['lrs'][m].predict_proba(fdm.lr_design(d, locked['knots']))[:, 1])
    P['ada'].append(d.ada_score.values); P['fin'].append(d.findrisc_adapted.values)
P = {k: np.array(v, float) for k, v in P.items()}
gb, lr, ada, fin = (P[k].mean(0) for k in ['gb', 'lr', 'ada', 'fin'])
saved = pd.read_csv(os.path.join(os.path.dirname(VAL_CSV), '..', 'results', 'validation_predictions.csv'))
OUT['meta']['max_abs_diff_vs_registered_predictions'] = dict(gb=float(np.abs(gb - saved.gb.values).max()), lr=float(np.abs(lr - saved.lr.values).max()),
                                                           ada=float(np.abs(ada - saved.ada.values).max()), fin=float(np.abs(fin - saved.findrisc.values).max()))
log('Reproduction check vs registered predictions: ' + json.dumps(OUT['meta']['max_abs_diff_vs_registered_predictions']))
H1 = h1_classify(gb, lr, ada, fin, 20261015)
log(f'Registered H1 reproduced: missed GB {int((~H1["ref"]["gb"][y==1]).sum())}, LR {int((~H1["ref"]["lr"][y==1]).sum())}, ADA {int((~H1["ref"]["ada"][y==1]).sum())}, FIN {int((~H1["ref"]["fin"][y==1]).sum())}')

# ------------------------------------------------------------ P2: paired DeLong tests
def delong_components(y, s):
    pos, neg = s[y == 1], s[y == 0]
    v = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return v.mean(), v.mean(1), v.mean(0)
def delong_paired(y, a, b):
    A, a10, a01 = delong_components(y, a); Bv, b10, b01 = delong_components(y, b)
    m, k = len(a10), len(a01)
    s10 = np.cov(np.vstack([a10, b10])); s01 = np.cov(np.vstack([a01, b01]))
    S = s10 / m + s01 / k
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; d = A - Bv; se = np.sqrt(var)
    z = d / se; p = 2 * stats.norm.sf(abs(z))
    return dict(auc_a=float(A), auc_b=float(Bv), difference=float(d), ci95=[float(d - 1.96 * se), float(d + 1.96 * se)], z=float(z), p=float(p))
sc = dict(gb=gb, lr=lr, ada=ada, fin=fin)
OUT['P2_paired_delong'] = {f'{a}_vs_{b}': delong_paired(y, sc[a], sc[b]) for a, b in [('gb', 'lr'), ('gb', 'ada'), ('gb', 'fin'), ('lr', 'ada'), ('lr', 'fin')]}
for k_, v in OUT['P2_paired_delong'].items(): log(f'P2 {k_}: diff {v["difference"]:+.4f} CI {v["ci95"][0]:+.4f} to {v["ci95"][1]:+.4f}, p {v["p"]:.4g}')

# ------------------------------------------------------------ P4: Asian-White gap with glucose-based outcomes (fasting subsample)
fmask = (val.in_fasting_subsample == 1).values
vf = val[fmask].reset_index(drop=True)
cl2 = h1_classify(gb[fmask], lr[fmask], ada[fmask], fin[fmask], 20261015)   # same call as the registered fasting-subsample analysis
outs = {'hba1c_or_fpg': np.nan_to_num(vf.y_hba1c_or_fpg.values).astype(int), 'fpg_only_ge126': (vf.fpg >= 126).values.astype(int),
        'hba1c_only_ge6.5': (vf.hba1c >= 6.5).values.astype(int)}
asian = (vf.race_eth == 'NH Asian').values; white = (vf.race_eth == 'NH White').values
P4 = dict(n=int(fmask.sum()), referral_rate=cl2['r'], overlap=dict(hba1c_only=int(((vf.hba1c >= 6.5) & ~(vf.fpg >= 126)).sum()), fpg_only=int((~(vf.hba1c >= 6.5) & (vf.fpg >= 126)).sum()),
          both=int(((vf.hba1c >= 6.5) & (vf.fpg >= 126)).sum())), results={})
for oname, yy in outs.items():
    P4['results'][oname] = {}
    for s in ['gb', 'lr', 'ada', 'fin']:
        miss = ~cl2['ref'][s]
        xa, na = int((miss & asian & (yy == 1)).sum()), int((asian & (yy == 1)).sum()); xw, nw = int((miss & white & (yy == 1)).sum()), int((white & (yy == 1)).sum())
        d, lo, hi = newcombe_diff(xa, na, xw, nw)
        P4['results'][oname][s] = dict(asian_missed=xa, asian_cases=na, asian_fnr=xa / na, white_missed=xw, white_cases=nw, white_fnr=xw / nw, difference=d, ci95_newcombe=[lo, hi],
                                       all_cases=int((yy == 1).sum()), all_fnr=fnr(cl2['ref'][s], yy))
    r_ = P4['results'][oname]['gb']; log(f'P4 {oname} GB: Asian {r_["asian_missed"]}/{r_["asian_cases"]} vs White {r_["white_missed"]}/{r_["white_cases"]}; diff {r_["difference"]:+.3f} ({r_["ci95_newcombe"][0]:+.3f} to {r_["ci95_newcombe"][1]:+.3f})')
OUT['P4_asian_white_glucose_outcomes'] = P4
json.dump(OUT, open(os.path.join(RES, 'posthoc_results.json'), 'w'), indent=1)

# ------------------------------------------------------------ helper: refit a reduced pipeline on development
dev = pd.read_csv(DEV_CSV).reset_index(drop=True); ydev = dev.y_hba1c.values.astype(int)
ALL = ['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation', 'mvpa_min_week_total']
def set_reduced(keep, mvpa_source='mvpa_min_week_total'):
    """Configure the development pipeline to use only `keep`; dropped items are set to a constant 0."""
    fdm.PREDICTORS = [p for p in ALL if p in keep]
    fdm.CONT_IMP = [v for v in ['bmi', 'waist_cm', 'mvpa_min_week_total'] if v in keep]
    fdm.BIN_IMP = [v for v in ['hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation'] if v in keep]
    fdm.SPLINE_VARS = [v for v, src in [('age', 'age'), ('bmi', 'bmi'), ('waist_cm', 'waist_cm'), ('log_mvpa', 'mvpa_min_week_total')] if src in keep]
    binv = [v for v in ['male', 'hypertension_dx', 'bp_medication', 'family_history', 'gestational_diabetes', 'told_prediabetes', 'active_recreation'] if v in keep]
    def lr_design_red(d, knots):
        d = d.copy(); d['log_mvpa'] = np.log1p(d.mvpa_min_week_total)
        parts = [fdm.rcs_basis(d[v].values.astype(float), knots[v]) for v in fdm.SPLINE_VARS]
        parts.append(d[binv].values.astype(float)); return np.column_stack(parts)
    fdm.lr_design = lr_design_red
    def frame(df):
        d = pd.DataFrame({c: (df[mvpa_source] if c == 'mvpa_min_week_total' else df[c]) if c in keep else 0.0 for c in ALL})
        return d
    return frame, lr_design_red
def predict_reduced(red, frame_new, lr_design_red):
    g, l = [], []
    for m in range(N_IMP):
        d = red['imputers'][m].transform(frame_new)
        g.append(platt_apply(red['platt'], red['xgbs'][m].predict_proba(d[fdm.PREDICTORS].values)[:, 1]))
        l.append(red['lrs'][m].predict_proba(lr_design_red(d, red['knots']))[:, 1])
    return np.array(g), np.array(l)

# ------------------------------------------------------------ P1: primary comparison without 'told prediabetes'
log('P1: refitting models without told_prediabetes on development')
keep1 = [p for p in ALL if p != 'told_prediabetes']
frame1, lrd1 = set_reduced(keep1)
red1 = fdm.fit_pipeline(frame1(dev), ydev, SEED + 100)
G1, L1 = predict_reduced(red1, frame1(val), lrd1)
cl = h1_classify(G1.mean(0), L1.mean(0), ada, fin, 20261015)       # scores as registered (locked pipeline)
p1 = dict(referral_rate=cl['r'], n_referred=cl['k'], xgb_config=red1['cfg'], range_extension=red1['extension'],
          auc=dict(gb=float(roc_auc_score(y, G1.mean(0))), lr=float(roc_auc_score(y, L1.mean(0)))),
          missed={s: int((~cl['ref'][s][y == 1]).sum()) for s in cl['ref']}, fnr={s: fnr(cl['ref'][s], y) for s in cl['ref']}, comparisons={})
for c in ['ada', 'fin', 'lr']:
    b_, c_, p_ = mcnemar_exact(cl['ref']['gb'][y == 1], cl['ref'][c][y == 1])
    p1['comparisons'][f'gb_vs_{c}'] = dict(fnr_difference=p1['fnr']['gb'] - p1['fnr'][c], gb_only_detected=b_, comparator_only_detected=c_, p_exact_mcnemar=p_)
for c in ['ada', 'fin']:
    b_, c_, p_ = mcnemar_exact(cl['ref']['lr'][y == 1], cl['ref'][c][y == 1])
    p1['comparisons'][f'lr_vs_{c}'] = dict(fnr_difference=p1['fnr']['lr'] - p1['fnr'][c], lr_only_detected=b_, comparator_only_detected=c_, p_exact_mcnemar=p_)
bd = {k_: [] for k_ in p1['comparisons']}
for b in range(B):
    idx = strat_boot_idx(y, rng); m = rng.integers(N_IMP)
    c2 = h1_classify(G1[m][idx], L1[m][idx], P['ada'][m][idx], P['fin'][m][idx], SEED + b); yb = y[idx]
    f = {s: fnr(c2['ref'][s], yb) for s in c2['ref']}
    for k_ in bd: a_, b2 = k_.split('_vs_'); bd[k_].append(f[a_] - f[b2])
for k_ in bd: p1['comparisons'][k_]['ci95_bootstrap'] = ci(bd[k_])
OUT['P1_without_told_prediabetes'] = p1
log('P1 missed: ' + json.dumps(p1['missed']) + '; ' + '; '.join(f'{k_} {v["fnr_difference"]:+.3f} ({v["ci95_bootstrap"][0]:+.3f} to {v["ci95_bootstrap"][1]:+.3f}) p {v["p_exact_mcnemar"]:.3f}' for k_, v in p1['comparisons'].items()))
json.dump(OUT, open(os.path.join(RES, 'posthoc_results.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------ P3: 2021-2023 without the changed items
log('P3: refitting models without changed items (activity, antihypertensive medication) and uncollected items')
keep3 = ['age', 'male', 'bmi', 'waist_cm', 'hypertension_dx', 'told_prediabetes']
frame3, lrd3 = set_reduced(keep3)
red3 = fdm.fit_pipeline(frame3(dev), ydev, SEED + 300)
val2 = pd.read_csv(VAL2_CSV).reset_index(drop=True); y3 = val2.y_hba1c.values.astype(int)
G3, L3 = predict_reduced(red3, frame3(val2), lrd3)
k3 = int(round(H1['r'] * len(y3))); tr3 = np.random.default_rng(20261015)
rg, rl = refer_top(G3.mean(0), k3, tr3), refer_top(L3.mean(0), k3, tr3)
b_, c_, p_ = mcnemar_exact(rg[y3 == 1], rl[y3 == 1])
bd3 = []
for b in range(B):
    idx = strat_boot_idx(y3, rng); m = rng.integers(N_IMP); kk = int(round(H1['r'] * len(idx))); trb = np.random.default_rng(SEED + 20000 + b)
    bd3.append(fnr(refer_top(G3[m][idx], kk, trb), y3[idx]) - fnr(refer_top(L3[m][idx], kk, trb), y3[idx]))
OUT['P3_2021_2023_without_changed_items'] = dict(n=int(len(y3)), cases=int(y3.sum()), predictors=keep3, referral_rate=float(H1['r']),
    missed_gb=int((~rg[y3 == 1]).sum()), missed_lr=int((~rl[y3 == 1]).sum()), fnr_gb=fnr(rg, y3), fnr_lr=fnr(rl, y3), difference=fnr(rg, y3) - fnr(rl, y3),
    gb_only_detected=b_, lr_only_detected=c_, p_exact_mcnemar=p_, ci95_bootstrap=ci(bd3), auc_gb=float(roc_auc_score(y3, G3.mean(0))), auc_lr=float(roc_auc_score(y3, L3.mean(0))),
    xgb_config=red3['cfg'], range_extension=red3['extension'])
r3 = OUT['P3_2021_2023_without_changed_items']
log(f'P3: missed GB {r3["missed_gb"]} vs LR {r3["missed_lr"]} of {r3["cases"]}; diff {r3["difference"]:+.3f} ({r3["ci95_bootstrap"][0]:+.3f} to {r3["ci95_bootstrap"][1]:+.3f}), p {r3["p_exact_mcnemar"]:.3f}; AUC GB {r3["auc_gb"]:.3f} LR {r3["auc_lr"]:.3f}')
json.dump(OUT, open(os.path.join(RES, 'posthoc_results.json'), 'w'), indent=1, default=float)
log('done')
