import sys, json
sys.argv=[sys.argv[0],'analysis/development_2011_2016.csv','results/_dev_tmp']
import numpy as np, pandas as pd, joblib
import fit_development_models as fdm
from fit_development_models import ChainedImputer, lr_design, COMPLETE, CONT_IMP, BIN_IMP, N_IMP
import __main__; __main__.ChainedImputer=ChainedImputer
from sklearn.metrics import roc_auc_score
L=joblib.load('models/locked_models.joblib')
names=[]
for v in ['age','bmi','waist_cm','log_mvpa']: names+= [f'{v}', f'{v}_nl']
names+=['male','hypertension_dx','bp_medication','family_history','gestational_diabetes','told_prediabetes','active_recreation']
B=np.array([m.coef_[0] for m in L['lrs']]); A=np.array([m.intercept_[0] for m in L['lrs']])
print('fallback used:', [getattr(m,'_fallback',None) for m in L['lrs']])
tab=pd.DataFrame({'term':['intercept']+names,'mean':np.r_[A.mean(),B.mean(0)],'min':np.r_[A.min(),B.min(0)],'max':np.r_[A.max(),B.max(0)]})
print(tab.to_string()); print('knots',L['knots'])
val=pd.read_csv('analysis/validation_2017_2020.csv'); y=val.y_hba1c.values
base=val[['SEQN']+COMPLETE+CONT_IMP+BIN_IMP].reset_index(drop=True)
pav=[];ppool=[]
for m in range(N_IMP):
    d=L['imputers'][m].transform(base); X=lr_design(d,L['knots'])
    pav.append(L['lrs'][m].predict_proba(X)[:,1])
    ppool.append(1/(1+np.exp(-(A.mean()+X@B.mean(0)))))
pav=np.mean(pav,0); ppool=np.mean(ppool,0)
print('AUC averaged-prob', roc_auc_score(y,pav), 'AUC pooled-eq', roc_auc_score(y,ppool), 'max abs diff', np.abs(pav-ppool).max(), 'corr', np.corrcoef(pav,ppool)[0,1])
tab.to_csv('results/lr_coefficients.csv',index=False)
json.dump(L['knots'],open('results/lr_knots.json','w'))
