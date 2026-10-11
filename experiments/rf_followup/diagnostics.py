"""Read-only model analysis on validation data; no training or model selection."""
import argparse, json, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
import followup as f

def run(base,out):
    b=joblib.load(out/'followup_predictor.joblib')
    with np.load(base/'results/dataset_arrays.npz') as z: x=z['x'];y=z['y']
    qs=json.loads((base/'results/queries.json').read_text())
    va=np.array([q['room'] in b['baseline']['validation_rooms'] and q['kind']=='random' for q in qs])
    x=x[va];y=y[va];rows=[]
    groups={'absolute_tx_rx':list(range(6)), 'relative_link':list(range(6,14)),
       'room_boundaries':list(range(14,29)), 'wood':list(range(29,36)),
       'metal':list(range(36,43)), 'concrete':list(range(43,50))}
    for model,fn in [('original',lambda xx:f.baseline_predict(b['baseline'],xx)),('followup',lambda xx:f.predict(b,xx))]:
        p=fn(x);ref=np.abs(p[:,f.PRIMARY]-y[:,f.PRIMARY]).mean(0)
        for group,cols in groups.items():
            for repeat in range(5):
                xp=x.copy();perm=np.random.default_rng(f.SEED+repeat).permutation(len(x));xp[:,cols]=x[perm][:,cols]
                change=np.abs(fn(xp)[:,f.PRIMARY]-y[:,f.PRIMARY]).mean(0)-ref
                for j,k in enumerate(f.PRIMARY):rows.append({'model':model,'group':group,'repeat':repeat,'target':b['targets'][k],'increase_in_validation_mae':change[j]})
    df=pd.DataFrame(rows);df.to_csv(out/'validation_group_permutation_repeats.csv',index=False)
    df.groupby(['model','target','group']).increase_in_validation_mae.agg(['mean','std']).reset_index().to_csv(out/'validation_group_importance.csv',index=False)
    # Saved tree importance is descriptive only, never interpreted causally.
    for k,h in b['heads'].items():
        if hasattr(h['model'],'feature_importances_'):
            pd.DataFrame({'feature':[b['feature_names'][i] for i in f.columns(h['spec']['features'])],
              'importance':h['model'].feature_importances_}).sort_values('importance',ascending=False).to_csv(out/('importance_'+b['targets'][k]+'.csv'),index=False)
    cv=pd.read_csv(out/'training_configuration_cv.csv')
    summary=cv.groupby(['model','target'])[['mae','r2']].mean();summary.to_csv(out/'training_cv_summary.csv')
    metrics=pd.read_csv(out/'test_metrics_all16.csv');emb=pd.read_csv(out/'embedding_comparison.csv')
    print('PER-ROOM PRIMARY TEST\n',metrics[(metrics.model.isin(['original_fast','followup']))&(metrics.room!='ALL')&metrics.target.isin([b['targets'][k] for k in f.PRIMARY])].to_string(index=False))
    print('CV MEAN\n',summary.to_string());print('EMBEDDINGS\n',emb.to_string(index=False))
    print('ABLATION\n',pd.read_csv(out/'delay_feature_ablation.csv').to_string(index=False))
    (out/'DIAGNOSTICS_NOTES.md').write_text('Permutation importance measures validation prediction sensitivity, not physical causality. Correlated/derived features can substitute for one another; shuffling feature groups can create physically inconsistent combinations. Constant features cannot reveal material effects that were not varied. Ablations refit a fixed delay model on training only. CV settings were fixed after validation selection; these seven training folds are a sensitivity diagnostic, not nested independent performance estimates.\n')
    print('DIAGNOSTICS COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.base,a.out)
