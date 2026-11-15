import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'analysis')
os.makedirs(OUT, exist_ok=True)

CYCLES = [  # folder, file pattern, sample, number of pooled 2-year cycles in that sample
    ('2011-2012', '{}_G', 'development', 3),
    ('2013-2014', '{}_H', 'development', 3),
    ('2015-2016', '{}_I', 'development', 3),
    ('2017-Mar2020', 'P_{}', 'validation', 1),
    ('2021-2023', '{}_L', 'validation2', 1),
]
FILES = ['DEMO', 'GHB', 'GLU', 'DIQ', 'BMX', 'BPQ', 'MCQ', 'PAQ', 'RHQ']
MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}

log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); log_lines.append(s)

def read(folder, pattern, f):
    d = pd.read_sas(os.path.join(BASE, folder, pattern.format(f) + '.xpt'), format='xport')
    num = d.select_dtypes('number').columns
    d[num] = d[num].mask(d[num].abs() < 1e-10, 0)  # SAS transport stores 0 as ~5e-79
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].apply(lambda v: v.decode().strip() if isinstance(v, bytes) else v)
    return d

def col(d, name):
    return d[name] if name in d.columns else pd.Series(np.nan, index=d.index)

def yesno(s):
    """1 -> 1, 2 -> 0, refused/don't know/missing -> NaN"""
    return s.map({1: 1.0, 2: 0.0})

def clean_num(s, codes=MISSING_CODES):
    return s.where(~s.isin(codes))

def weekly_minutes_old(d, flag, days, mins):
    """2011-2020 GPAQ items: flag (1/2), days per week, minutes per day."""
    f = col(d, flag); dd = clean_num(col(d, days)); mm = clean_num(col(d, mins))
    out = dd * mm
    out = out.where(f != 2, 0.0)
    return out.where(f.isin([1, 2]))

def weekly_minutes_new(d, q, u, mins):
    """2021-2023 items: frequency + unit (D/W/M/Y) + minutes per session."""
    qq = clean_num(col(d, q)); mm = clean_num(col(d, mins)); uu = col(d, u)
    per_week = qq * uu.map({'D': 7.0, 'W': 1.0, 'M': 7 / 30.4, 'Y': 7 / 365.0})
    per_week = per_week.where(~((qq == 0) | (uu == '')), 0.0)  # 0 times -> 0 minutes
    out = per_week * mm.fillna(0)
    return out.where(qq.notna())

frames = []
for folder, pat, sample, npool in CYCLES:
    d = read(folder, pat, 'DEMO')
    for f in FILES[1:]:
        x = read(folder, pat, f)
        if f == 'GLU':
            x = x[[c for c in x.columns if c in ('SEQN', 'LBXGLU', 'WTSAF2YR', 'WTSAFPRP')]]
        d = d.merge(x, on='SEQN', how='left', suffixes=('', '_' + f))
    n0 = len(d)
    o = pd.DataFrame({'SEQN': d.SEQN.astype(int), 'cycle': folder, 'sample': sample})

    # ---- eligibility ----
    age = d.RIDAGEYR
    adult = age >= 20
    pregnant = col(d, 'RIDEXPRG') == 1
    dx = col(d, 'DIQ010')
    diagnosed = dx == 1
    unknown_dx = dx.isin([7, 9]) | dx.isna()
    treated = (col(d, 'DIQ050') == 1) | (col(d, 'DIQ070') == 1)
    has_a1c = d.LBXGH.notna()
    keep = adult & ~pregnant & ~diagnosed & ~unknown_dx & ~treated & has_a1c
    log(f'{folder}: participants {n0}; adults 20+ {int(adult.sum())}; pregnant {int((adult & pregnant).sum())}; '
        f'diagnosed {int((adult & ~pregnant & diagnosed).sum())}; diabetes status unknown {int((adult & ~pregnant & ~diagnosed & unknown_dx).sum())}; '
        f'treated without diagnosis {int((adult & ~pregnant & ~diagnosed & ~unknown_dx & treated).sum())}; '
        f'no HbA1c {int((adult & ~pregnant & ~diagnosed & ~unknown_dx & ~treated & ~has_a1c).sum())}; analysed {int(keep.sum())}')

    # ---- outcomes ----
    o['hba1c'] = d.LBXGH
    o['fpg'] = d.LBXGLU
    o['y_hba1c'] = (d.LBXGH >= 6.5).astype(int)
    fw = col(d, 'WTSAFPRP') if sample == 'validation' else col(d, 'WTSAF2YR')
    o['in_fasting_subsample'] = ((fw > 0) & d.LBXGLU.notna()).astype(int)
    o['y_hba1c_or_fpg'] = np.where(o.in_fasting_subsample == 1, ((d.LBXGH >= 6.5) | (d.LBXGLU >= 126)).astype(int), np.nan)

    # ---- predictors ----
    o['age'] = age
    o['male'] = (d.RIAGENDR == 1).astype(int)
    o['bmi'] = d.BMXBMI
    o['waist_cm'] = d.BMXWAIST
    htn = yesno(col(d, 'BPQ020'))
    if 'BPQ050A' in d.columns:   # 2011-2020: BPQ020 -> BPQ040A -> BPQ050A
        told_rx = yesno(col(d, 'BPQ040A'))
        bpmed = yesno(col(d, 'BPQ050A'))
        bpmed = bpmed.where(~((htn == 0) | (told_rx == 0)), 0.0)
    else:                        # 2021-2023: BPQ020 -> BPQ150
        bpmed = yesno(col(d, 'BPQ150')).where(htn != 0, 0.0)
    o['hypertension_dx'] = htn
    o['bp_medication'] = bpmed
    o['hypertension_ada'] = np.where(htn.isna() & bpmed.isna(), np.nan, ((htn == 1) | (bpmed == 1)).astype(float))
    o['family_history'] = yesno(col(d, 'MCQ300C')) if 'MCQ300C' in d.columns else np.nan
    if 'RHQ162' in d.columns:
        everpreg = yesno(col(d, 'RHQ131'))
        gdm = yesno(col(d, 'RHQ162')).where(everpreg != 0, 0.0)
        gdm = gdm.where(d.RIAGENDR == 2, 0.0)  # men coded 0, as in the ADA test
        o['gestational_diabetes'] = gdm
    else:
        o['gestational_diabetes'] = np.nan
    told_pre = yesno(col(d, 'DIQ160'))
    o['told_prediabetes'] = told_pre.where(dx != 3, 1.0)  # DIQ010 = 3 'borderline' also counts
    o['high_glucose_history'] = np.where(o.told_prediabetes.isna() & o.gestational_diabetes.isna(), np.nan,
                                         ((o.told_prediabetes == 1) | (o.gestational_diabetes == 1)).astype(float))
    if 'PAQ650' in d.columns:
        rec = weekly_minutes_old(d, 'PAQ650', 'PAQ655', 'PAD660').fillna(0) + weekly_minutes_old(d, 'PAQ665', 'PAQ670', 'PAD675').fillna(0)
        rec = rec.where(col(d, 'PAQ650').isin([1, 2]) & col(d, 'PAQ665').isin([1, 2]))
        work = weekly_minutes_old(d, 'PAQ605', 'PAQ610', 'PAD615').fillna(0) + weekly_minutes_old(d, 'PAQ620', 'PAQ625', 'PAD630').fillna(0)
        trans = weekly_minutes_old(d, 'PAQ635', 'PAQ640', 'PAD645').fillna(0)
        o['active_recreation'] = np.where(rec.isna(), np.nan, ((col(d, 'PAQ650') == 1) | (col(d, 'PAQ665') == 1)).astype(float))
        o['mvpa_min_week_recreation'] = rec.clip(upper=3360)
        o['mvpa_min_week_total'] = (rec + work + trans).where(rec.notna()).clip(upper=3360)  # preregistered cap: 8 h/day
    else:
        rec = weekly_minutes_new(d, 'PAD790Q', 'PAD790U', 'PAD800').fillna(0) + weekly_minutes_new(d, 'PAD810Q', 'PAD810U', 'PAD820').fillna(0)
        rec = rec.where(clean_num(col(d, 'PAD790Q')).notna() | clean_num(col(d, 'PAD810Q')).notna())
        o['active_recreation'] = np.where(rec.isna(), np.nan, (rec > 0).astype(float))
        o['mvpa_min_week_recreation'] = rec.clip(upper=3360)
        o['mvpa_min_week_total'] = np.nan  # work and transport activity not collected in 2021-2023
    o['sedentary_min_day'] = clean_num(col(d, 'PAD680'))

    # ---- subgroup and descriptive ----
    o['race_eth'] = d.RIDRETH3.map({1: 'Mexican American', 2: 'Other Hispanic', 3: 'NH White', 4: 'NH Black', 6: 'NH Asian', 7: 'Other/multiracial'})
    o['nh_asian'] = (d.RIDRETH3 == 6).astype(int)
    o['education'] = clean_num(col(d, 'DMDEDUC2'))
    o['income_poverty_ratio'] = d.INDFMPIR

    # ---- survey design ----
    if sample == 'validation':
        o['wt_mec'] = d.WTMECPRP; o['wt_fasting'] = col(d, 'WTSAFPRP')
    else:
        o['wt_mec'] = d.WTMEC2YR / npool; o['wt_fasting'] = col(d, 'WTSAF2YR') / npool
    o['psu'] = d.SDMVPSU; o['strata'] = d.SDMVSTRA

    # ---- ADA Risk Test (Bang et al. 2009) ----
    pts_age = pd.cut(age, [0, 39.999, 49.999, 59.999, 200], labels=[0, 1, 2, 3]).astype(float)
    pts_bmi = pd.cut(d.BMXBMI, [0, 24.999, 29.999, 39.999, 200], labels=[0, 1, 2, 3]).astype(float)
    ada_parts = [pts_age, o.male.astype(float), o.gestational_diabetes, o.family_history, o.hypertension_ada,
                 1 - o.active_recreation, pts_bmi]
    o['ada_score'] = sum(ada_parts) if sample != 'validation2' else np.nan
    o['ada_positive'] = np.where(o.ada_score.isna(), np.nan, (o.ada_score >= 5).astype(float))

    # ---- Adapted FINDRISC (Lindstrom & Tuomilehto 2003; fruit/vegetable item omitted, max 25) ----
    f_age = pd.cut(age, [0, 44.999, 54.999, 64.999, 200], labels=[0, 2, 3, 4]).astype(float)
    f_bmi = pd.cut(d.BMXBMI, [0, 24.999, 30, 200], labels=[0, 1, 3]).astype(float)
    w = d.BMXWAIST
    f_waist = np.where(d.RIAGENDR == 1, np.select([w < 94, w <= 102, w > 102], [0, 3, 4], np.nan),
                       np.select([w < 80, w <= 88, w > 88], [0, 3, 4], np.nan))
    f_waist = pd.Series(f_waist, index=d.index).where(w.notna())
    pa_total = o.mvpa_min_week_total if sample != 'validation2' else o.mvpa_min_week_recreation
    f_pa = np.where(pa_total.isna(), np.nan, np.where(pa_total >= 210, 0, 2))
    f_bpmed = o.bp_medication * 2
    f_glu = o.high_glucose_history * 5
    f_fam = o.family_history * 5   # MCQ300C = close (first-degree) relative
    o['findrisc_adapted'] = (f_age + f_bmi + f_waist + f_pa + f_bpmed + f_glu + f_fam) if sample != 'validation2' else np.nan

    frames.append(o[keep.values])

all_ = pd.concat(frames, ignore_index=True)
names = {'development': 'development_2011_2016.csv', 'validation': 'validation_2017_2020.csv', 'validation2': 'validation2_2021_2023.csv'}
for s, fn in names.items():
    part = all_[all_['sample'] == s]
    part.to_csv(os.path.join(OUT, fn), index=False)
    log(f'{fn}: {len(part)} rows, {int(part.y_hba1c.sum())} HbA1c cases')

codebook = [
 ('SEQN','NHANES participant ID','SEQN'), ('cycle','NHANES cycle',''), ('sample','development / validation / validation2',''),
 ('hba1c','HbA1c (%)','LBXGH'), ('fpg','Fasting plasma glucose (mg/dL)','LBXGLU'),
 ('y_hba1c','Primary outcome: undiagnosed diabetes, HbA1c >= 6.5%','LBXGH'),
 ('in_fasting_subsample','1 if in morning fasting subsample with valid weight and FPG','WTSAF2YR / WTSAFPRP, LBXGLU'),
 ('y_hba1c_or_fpg','Sensitivity outcome: HbA1c >= 6.5% or FPG >= 126 (fasting subsample only, else blank)','LBXGH, LBXGLU'),
 ('age','Age in years (80 = 80+)','RIDAGEYR'), ('male','1 male, 0 female','RIAGENDR'), ('bmi','Body mass index (kg/m2)','BMXBMI'),
 ('waist_cm','Waist circumference (cm)','BMXWAIST'), ('hypertension_dx','Ever told high blood pressure','BPQ020'),
 ('bp_medication','Taking prescribed blood-pressure medication (0 if never told hypertension or never advised)','BPQ040A/BPQ050A; BPQ150 in 2021-2023'),
 ('hypertension_ada','ADA item: diagnosed hypertension or BP medication','BPQ020, BPQ050A/BPQ150'),
 ('family_history','Close relative with diabetes (blank in 2021-2023: dropped)','MCQ300C'),
 ('gestational_diabetes','Women: ever told diabetes in pregnancy (0 if never pregnant; men 0; blank in 2021-2023)','RHQ131, RHQ162'),
 ('told_prediabetes','Ever told prediabetes, or borderline diabetes (DIQ010 = 3)','DIQ160, DIQ010'),
 ('high_glucose_history','FINDRISC item: prediabetes/borderline or gestational diabetes','DIQ160, DIQ010, RHQ162'),
 ('active_recreation','Any moderate or vigorous recreational activity','PAQ650, PAQ665; PAD790-PAD820 in 2021-2023'),
 ('mvpa_min_week_recreation','Recreational moderate + vigorous minutes per week','PAQ655 x PAD660 + PAQ670 x PAD675; PAD790/800/810/820'),
 ('mvpa_min_week_total','Work + transport + recreation minutes per week, capped at 3360 (blank in 2021-2023)','PAQ605-PAD675'),
 ('sedentary_min_day','Sedentary minutes per day','PAD680'),
 ('race_eth','Race and ethnicity','RIDRETH3'), ('nh_asian','1 if non-Hispanic Asian','RIDRETH3 = 6'),
 ('education','Education (adults 20+), 1-5','DMDEDUC2'), ('income_poverty_ratio','Family income to poverty ratio','INDFMPIR'),
 ('wt_mec','Exam weight for analysis (2011-2016 divided by 3)','WTMEC2YR / WTMECPRP'),
 ('wt_fasting','Fasting subsample weight (2011-2016 divided by 3)','WTSAF2YR / WTSAFPRP'),
 ('psu','Masked variance PSU','SDMVPSU'), ('strata','Masked variance stratum','SDMVSTRA'),
 ('ada_score','ADA Risk Test, 0-11 (Bang 2009); blank in 2021-2023','derived'),
 ('ada_positive','ADA score >= 5','derived'),
 ('findrisc_adapted','Adapted FINDRISC, 0-25 (fruit/vegetable item omitted; PA >= 210 min/week of work, transport and recreation); blank in 2021-2023','derived'),
]
pd.DataFrame(codebook, columns=['variable', 'description', 'NHANES source']).to_csv(os.path.join(OUT, 'codebook.csv'), index=False)
open(os.path.join(OUT, 'build_log.txt'), 'w').write('\n'.join(log_lines) + '\n')
