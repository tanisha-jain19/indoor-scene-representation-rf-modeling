"""Exploratory fast deployment candidate, selected on validation configurations."""
import json,time,os,subprocess,sys
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
import experiment as ex

def main():
    qs,X,Y,sim_seconds,_=ex.load_data()
    bundle=joblib.load(ex.OUT/'predictor.joblib')
    tr=np.array([q['room'] in ex.TRAIN and q['kind']=='random' for q in qs])
    va=np.array([q['room'] in ex.VAL and q['kind']=='random' for q in qs])
    ids=np.flatnonzero(va)[::8]
    geom={r:ex.geometry(r)[0] for r in ex.VAL}
    z=bundle['target_scaler'].transform(ex.transform_y(Y))
    scale=np.maximum(Y[tr][:,ex.PRIMARY].std(0),1e-8)
    options=[];fitted={}
    for leaf in [2,5]:
        model=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=1.,random_state=ex.SEED,n_jobs=1)
        start=time.perf_counter();model.fit(X[tr],z[tr]);train_seconds=time.perf_counter()-start
        candidate=dict(bundle,model=model,selected=f'extra_trees_leaf{leaf}_single_thread')
        p=ex.predict(candidate,X[va]);mae=np.abs(p[:,ex.PRIMARY]-Y[va][:,ex.PRIMARY]).mean(0)
        elapsed=[]
        for i in ids:
            q=qs[i];start=time.perf_counter()
            ex.predict(candidate,ex.features(geom[q['room']],q['tx'],q['rx']))
            elapsed.append(time.perf_counter()-start)
        latency=float(np.median(elapsed))
        row=dict(model=candidate['selected'],validation_normalized_mae=float(np.mean(mae/scale)),
                 validation_gain_mae_db=float(mae[0]),validation_delay_mae_ns=float(mae[1]),
                 validation_fraction_mae=float(mae[2]),median_seconds=latency,
                 validation_speedup=float(np.median(sim_seconds[ids])/latency),train_seconds=train_seconds)
        options.append(row);fitted[candidate['selected']]=candidate;print(row,flush=True)
    eligible=[r for r in options if r['validation_speedup']>=3]
    chosen=min(eligible,key=lambda r:r['validation_normalized_mae']) if eligible else min(options,key=lambda r:r['median_seconds'])
    joblib.dump(fitted[chosen['model']],ex.OUT/'fast_predictor.joblib',compress=3)
    ex.save_json(ex.OUT/'fast_model_selection.json',dict(candidates=options,selected=chosen['model'],
        rule='Lowest validation normalized MAE among >=3x candidates; otherwise fastest',test_labels_used_for_selection=False,
        note='Exploratory follow-up after initial HGB evaluation. Validation eligibility uses recorded simulation times; final speed requires same-runtime benchmark.'))
    env=dict(os.environ,RF_MODEL_FILE='fast_predictor.joblib',RF_REPORT_SUBDIR='fast_model')
    subprocess.run([sys.executable,str(ex.ROOT/'experiment.py'),'evaluate'],env=env,check=True)

if __name__=='__main__':main()
