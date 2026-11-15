import os, sys
sys.argv = [sys.argv[0], 'analysis/development_2011_2016.csv', 'results/_dev_tmp']
import numpy as np, pandas as pd, joblib, xgboost as xgb
import fit_development_models as fdm
from fit_development_models import ChainedImputer, PREDICTORS, COMPLETE, CONT_IMP, BIN_IMP
import __main__
__main__.ChainedImputer = ChainedImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Serif', 'pdf.fonttype': 42, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8})
locked = joblib.load('models/locked_models.joblib')
val = pd.read_csv('analysis/validation_2017_2020.csv')
base = val[['SEQN'] + COMPLETE + CONT_IMP + BIN_IMP].reset_index(drop=True)
d1 = locked['imputers'][0].transform(base)
contrib = locked['xgbs'][0].get_booster().predict(xgb.DMatrix(d1[PREDICTORS].values, feature_names=PREDICTORS), pred_contribs=True)[:, :-1]
shap = pd.DataFrame(contrib, columns=PREDICTORS)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, v, lab in zip(axs, ['age', 'waist_cm', 'told_prediabetes'], ['Age (years)', 'Waist circumference (cm)', 'Told prediabetes (0 = no, 1 = yes)']):
    obs = base[v].notna().values  # observed values only
    x = base.loc[obs, v].values; s = shap.loc[obs, v].values
    if v == 'told_prediabetes':
        x = x + np.random.default_rng(1).uniform(-0.08, 0.08, len(x))
    ax.scatter(x, s, s=3, alpha=0.35, color='#0072B2', linewidths=0)
    ax.axhline(0, color='#888888', lw=0.6)
    ax.set_xlabel(lab); ax.set_ylabel('SHAP value (log-odds)')
    if v == 'told_prediabetes': ax.set_xticks([0, 1])
fig.tight_layout()
out = 'figures'; os.makedirs(out, exist_ok=True)
fig.savefig(f'{out}/FigA1_shap_dependence.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out}/FigA1_shap_dependence.pdf', bbox_inches='tight')
# consistency check with the reported importance
print(shap.abs().mean().sort_values(ascending=False).round(3).head(5).to_dict())
