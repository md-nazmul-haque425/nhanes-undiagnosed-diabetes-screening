import json, os
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RES = 'results'   # run from the repository root
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Serif', 'pdf.fonttype': 42, 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.8})
# Okabe-Ito (colour-vision safe)
C = {'gb': '#0072B2', 'lr': '#D55E00', 'ada': '#009E73', 'fin': '#CC79A7', 'all': '#7F7F7F'}
NAMES = {'gb': 'Gradient boosting', 'lr': 'Logistic regression', 'ada': 'ADA Risk Test', 'fin': 'Adapted FINDRISC'}

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)

# ---------------- Fig 1: participant flow
cols = [('Development\n2011–2016', [29902, 17048, 192, 2266, 11, 148, 1242, 13189, 410]),
        ('Primary validation\n2017–March 2020', [15560, 9232, 87, 1417, 5, 106, 973, 6644, 237]),
        ('Secondary validation\n2021–2023', [11933, 7809, 41, 1070, 3, 114, 1755, 4826, 128])]
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.set_xlim(0, 3); ax.set_ylim(1.4, 9.2); ax.axis('off')
def box(x, y, w, h, txt, bold=False, fc='white'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec='#333333', lw=0.8))
    ax.text(x, y, txt, ha='center', va='center', fontsize=7.6, fontweight='bold' if bold else 'normal', linespacing=1.25)
def arrow(x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333333'))
for i, (title, v) in enumerate(cols):
    x = 0.5 + i
    box(x, 8.5, 0.9, 1.05, title.replace(chr(10), ' ') + f'\nNHANES participants\nn = {v[0]:,}')
    arrow(x, 8.05, 7.6)
    box(x, 7.15, 0.86, 0.85, f'Aged ≥20 years\nn = {v[1]:,}')
    arrow(x, 6.7, 5.95)
    box(x, 4.75, 0.9, 2.2, 'Excluded\n'
        f'Pregnant: {v[2]:,}\nDiagnosed diabetes: {v[3]:,}\nStatus unknown: {v[4]:,}\n'
        f'Treated, no diagnosis: {v[5]:,}\nNo HbA1c: {v[6]:,}', fc='#F2F2F2')
    arrow(x, 3.65, 2.75)
    box(x, 2.15, 0.9, 1.15, f'Analysed\nn = {v[7]:,}\nHbA1c ≥6.5%: {v[8]} ({100 * v[8] / v[7]:.1f}%)', bold=True)
save(fig, 'Fig1_flow')

# ---------------- Fig 2: calibration
d = pd.read_csv(f'{RES}/validation_predictions.csv'); y = d.y.values
fig, ax = plt.subplots(figsize=(3.6, 3.6))
mx = 0.22
for s in ['gb', 'lr']:
    p = d[s].values
    lw = sm.nonparametric.lowess(y, p, frac=0.4, it=0, return_sorted=True)
    ax.plot(lw[:, 0], lw[:, 1], color=C[s], lw=1.4, label=NAMES[s])
    q = pd.qcut(p, 10, duplicates='drop'); g = pd.DataFrame(dict(p=p, y=y, q=q)).groupby('q', observed=True).mean()
    ax.scatter(g.p, g.y, color=C[s], s=14, zorder=3, marker='o' if s == 'gb' else 's')
ax.plot([0, mx], [0, mx], ls='--', color='#888888', lw=0.8, label='Perfect calibration')
ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel('Predicted risk'); ax.set_ylabel('Observed proportion')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
save(fig, 'Fig2_calibration')

# ---------------- Fig 3: decision curve
dc = pd.read_csv(f'{RES}/decision_curve.csv')
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for s, ls in [('gb', '-'), ('lr', '-'), ('ada', '-.'), ('fin', ':')]:
    ax.plot(dc.threshold, dc[s], color=C[s], ls=ls, lw=1.4, label=NAMES[s] + (' (≥5)' if s == 'ada' else ' (≥11)' if s == 'fin' else ''))
ax.plot(dc.threshold, dc.treat_all, color=C['all'], ls='--', lw=1, label='Test everyone')
ax.axhline(0, color='black', lw=0.8, label='Test no one')
ax.set_ylim(-0.005, 0.03); ax.set_xlim(0.01, 0.30)
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.legend(frameon=False, fontsize=7.2)
save(fig, 'Fig3_decision_curve')

# ---------------- Fig 4: SHAP importance
sh = pd.read_csv(f'{RES}/shap_importance.csv', index_col=0).sort_values('mean_abs_shap_all')
label = {'age': 'Age', 'waist_cm': 'Waist circumference', 'told_prediabetes': 'Told prediabetes',
         'family_history': 'Family history of diabetes', 'bmi': 'BMI', 'mvpa_min_week_total': 'Activity (min/week)',
         'active_recreation': 'Any recreational activity', 'male': 'Male sex', 'gestational_diabetes': 'Gestational diabetes',
         'hypertension_dx': 'Hypertension diagnosis', 'bp_medication': 'Antihypertensive medication'}
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.barh([label[i] for i in sh.index], sh.mean_abs_shap_all, color=C['gb'], height=0.6)
for i, v in enumerate(sh.mean_abs_shap_all):
    ax.text(v + 0.006, i, f'{v:.2f}', va='center', fontsize=7)
ax.set_xlabel('Mean |SHAP value| (log-odds)'); ax.set_xlim(0, 0.62)
save(fig, 'Fig4_shap_importance')
print(sorted(os.listdir(OUT)))
