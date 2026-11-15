import pandas as pd, numpy as np
files = [('Development 2011-2016', 'analysis/development_2011_2016.csv'),
         ('Primary validation 2017-March 2020', 'analysis/validation_2017_2020.csv'),
         ('Secondary validation 2021-2023', 'analysis/validation2_2021_2023.csv')]
def med(s, d=1):
    s = s.dropna(); q = s.quantile([.5, .25, .75]).round(d)
    f = (lambda x: f'{x:.0f}') if d == 0 else (lambda x: f'{x:.{d}f}')
    return f'{f(q[.5])} ({f(q[.25])}-{f(q[.75])})' if len(s) else 'Not available'
def pct(s, d=1):
    s = s.dropna(); return f'{100 * s.mean():.{d}f}' if len(s) else 'Not collected'
rows = {}
for name, f in files:
    v = pd.read_csv(f); w = v[v.male == 0]
    race = v.race_eth.value_counts(normalize=True) * 100
    rows[name] = {
        'n': f'{len(v):,}',
        'Age, years, median (IQR)': med(v.age, 0),
        'Male, %': pct(v.male),
        'BMI, kg/m2, median (IQR)': med(v.bmi),
        'Waist circumference, cm, median (IQR)': med(v.waist_cm),
        'Hypertension diagnosis, %': pct(v.hypertension_dx),
        'Family history of diabetes, %': pct(v.family_history),
        'Gestational diabetes, % of women': pct(w.gestational_diabetes),
        'Told prediabetes or borderline, %': pct(v.told_prediabetes),
        'Any recreational activity, %': pct(v.active_recreation),
        'Activity (total), min/week, median (IQR)': med(v.mvpa_min_week_total, 0),
        'NH White / NH Black / NH Asian, %': ' / '.join(f'{race.get(k, 0):.1f}' for k in ['NH White', 'NH Black', 'NH Asian']),
        'Mexican American / Other Hispanic / Other, %': ' / '.join(f'{race.get(k, 0):.1f}' for k in ['Mexican American', 'Other Hispanic', 'Other/multiracial']),
        'Undiagnosed diabetes (HbA1c >=6.5%), n (%)': f'{int(v.y_hba1c.sum())} ({100 * v.y_hba1c.mean():.1f})',
        'Missing: waist / family history / BMI, n': f'{v.waist_cm.isna().sum()} / {v.family_history.isna().sum()} / {v.bmi.isna().sum()}',
        'Missing gestational diabetes, n of women': f'{w.gestational_diabetes.isna().sum()} of {len(w)}',
    }
t = pd.DataFrame(rows); t.to_csv('results/table1.csv'); print(t.to_string())
