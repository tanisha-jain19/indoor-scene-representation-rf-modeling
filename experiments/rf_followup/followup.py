"""Separate, reproducible follow-up. Original directory is only read.

Usage: python followup.py --base ORIGINAL --out NEW_DIRECTORY
No new simulation labels, test-guided selection or edits to the baseline.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, hashlib, importlib.util, json, platform, shutil, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

SEED = 20260923
PRIMARY = [1, 4, 2]

def engineered(x):
    x = np.atleast_2d(x)
    d = x[:, 12:13]
    # Method of images for six ideal planar walls. No obstruction/material tracing.
    excess = (np.sqrt(d*d + 4*x[:,17:23]*x[:,23:29])-d)/0.299792458
    fractions = x[:, [33,40,47]]/np.maximum(d, 1e-12)
    return np.c_[x, excess, excess.min(1), excess.mean(1), excess.max(1), fractions]

def columns(mode):
    if mode == 'original': return np.arange(50)
    if mode == 'engineered': return np.arange(62)
    if mode == 'no_material': return np.arange(29)
    if mode == 'no_absolute_xyz': return np.arange(6,50)
    raise ValueError(mode)

def inputs(x, mode):
    return (engineered(x) if mode == 'engineered' else np.atleast_2d(x))[:,columns(mode)]

def transform(y):
    z = np.array(y, copy=True)
    z[:,[0,3,4,5]] = np.log1p(np.maximum(z[:,[0,3,4,5]],0))
    return z

def inverse(z):
    y = np.array(z, copy=True)
    y[:,[0,3,4,5]] = np.maximum(np.expm1(y[:,[0,3,4,5]]),0)
    y[:,2] = np.clip(y[:,2],0,1)
    y[:,[10,11,13,14]] = np.clip(y[:,[10,11,13,14]],-1,1)
    return y

def baseline_predict(bundle,x):
    return inverse(bundle['target_scaler'].inverse_transform(bundle['model'].predict(np.atleast_2d(x))))

def make(spec):
    if spec['family'] == 'et':
        return ExtraTreesRegressor(n_estimators=250, min_samples_leaf=spec['leaf'],
             max_features=spec['fraction'], max_depth=spec.get('depth'),
             random_state=SEED, n_jobs=1)
    return HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15,
             l2_regularization=5, early_stopping=False, random_state=SEED,
             loss=spec.get('loss','squared_error'))

def fit(spec,x,y,k=None):
    m=make(spec); sc=None
    if k is None:
        sc=StandardScaler().fit(transform(y)); target=sc.transform(transform(y))
    else:
        target=np.log1p(np.maximum(y[:,k],0)) if spec.get('log') else y[:,k]
    m.fit(inputs(x,spec['features']),target)
    return {'model':m,'scaler':sc,'spec':spec,'target':k}

def predict_fit(b,x):
    p=b['model'].predict(inputs(x,b['spec']['features']))
    if b['target'] is None: return inverse(b['scaler'].inverse_transform(p))
    if b['spec'].get('log'): p=np.expm1(p)
    if b['target']==4: p=np.maximum(p,0)
    if b['target']==2: p=np.clip(p,0,1)
    return p

def predict(b,x):
    p=baseline_predict(b['baseline'],x) if b['joint'] is None else predict_fit(b['joint'],x)
    for k,head in b['heads'].items(): p[:,k]=predict_fit(head,x)
    return p

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def pool(a): return np.stack([a.mean(0),a.std(0),a.min(0),a.max(0),np.median(a,0)],axis=1).reshape(-1)

def run(base,out):
    base=base.resolve(); out=out.resolve()
    assert base != out and base not in out.parents, 'Output must be outside baseline'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(), 'Finished run exists; do not overwrite it'
    r=base/'results'
    protected=[r/'dataset_arrays.npz',r/'queries.json',r/'fast_predictor.joblib',
               r/'predictor.joblib',r/'fast_model/embedding_preprocessing.joblib',
               base/'FINAL_REPORT.md',base/'source/experiment.py',base/'source/geometry_io.py']
    before={str(p.relative_to(base)):sha(p) for p in protected}
    save_json(out/'original_hashes_before.json',before)
    specs_joint=[dict(name='joint_leaf1',family='et',leaf=1,fraction=1.,features='original'),
      dict(name='joint_leaf4_mf07',family='et',leaf=4,fraction=.7,features='original'),
      dict(name='joint_leaf2_engineered',family='et',leaf=2,fraction=1.,features='engineered')]
    specs_head=[dict(name='et_leaf2',family='et',leaf=2,fraction=1.,features='original'),
      dict(name='et_leaf1_mf07',family='et',leaf=1,fraction=.7,features='original'),
      dict(name='et_leaf4_engineered',family='et',leaf=4,fraction=.7,features='engineered'),
      dict(name='hgb_direct',family='hgb',features='original',log=False),
      dict(name='hgb_log',family='hgb',features='original',log=True),
      dict(name='hgb_engineered_absolute',family='hgb',features='engineered',loss='absolute_error',log=False)]
    protocol={'seed':SEED,'joint_candidates':specs_joint,'head_candidates':specs_head,
      'head_log_rule':'ET delay uses log1p; HGB log candidate only for delay',
      'selection':'Minimum validation primary MAE/train SD; baseline and best joint and baseline with independently selected primary heads compete',
      'test_policy':'Previously examined test set; exploratory final comparison, not independent confirmation',
      'cv_policy':'7 training-configuration folds for fixed selected settings; diagnostic only',
      'ablation_policy':'Fixed ET delay log leaf2, original/no_material/no_absolute_xyz; validation only, not selected',
      'no_new_sionna_labels':True,'original_directory_read_only':str(base)}
    save_json(out/'protocol.json',protocol)
    qt=(r/'queries.json').read_text(); qs=json.loads(qt)
    with np.load(r/'dataset_arrays.npz') as z:
        assert str(z['query_sha256'])==hashlib.sha256(qt.encode()).hexdigest()
        x=z['x']; y=z['y']
    assert x.shape==(1815,50) and y.shape==(1815,16)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    old=joblib.load(r/'fast_predictor.joblib'); old['model'].set_params(n_jobs=1)
    old_hgb=joblib.load(r/'predictor.joblib')
    targets=old['targets']; names=old['feature_names']
    train=old['train_rooms']; val=old['validation_rooms']; test=old['test_rooms']
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    rooms=np.array([q['room'] for q in qs]); random=np.array([q['kind']=='random' for q in qs])
    tr=random&np.isin(rooms,train); va=random&np.isin(rooms,val); te=random&np.isin(rooms,test)
    assert (tr.sum(),va.sum(),te.sum(),(~random).sum())==(1120,320,320,55)
    save_json(out/'data_audit.json',{'shape_x':list(x.shape),'shape_y':list(y.shape),
       'train':train,'validation':val,'test':test,'counts':[1120,320,320,55],
       'unique_room_tx_rx':len({(q['room'],tuple(q['tx']),tuple(q['rx'])) for q in qs}),
       'constant_training_features':[names[i] for i in np.where(x[tr].std(0)<1e-12)[0]],
       'versions':{'python':sys.version,'sklearn':sklearn.__version__,'numpy':np.__version__}})
    assert len({(q['room'],tuple(q['tx']),tuple(q['rx'])) for q in qs})==1815
    sd=np.maximum(y[tr][:,PRIMARY].std(0),1e-8)
    pv=baseline_predict(old,x[va]); rows=[]; joint={}; heads={}
    def record(name,p,duration=0):
        errors=np.abs(y[va][:,PRIMARY]-p[:,PRIMARY]).mean(0)
        row={'model':name,'score':float(np.mean(errors/sd)),'gain_mae':errors[0],
             'delay_mae':errors[1],'fraction_mae':errors[2],'fit_seconds':duration}
        rows.append(row); pd.DataFrame(rows).to_csv(out/'validation_models.csv',index=False)
        print('VALIDATION',row,flush=True); return row
    record('original_fast',pv)
    record('original_hgb',baseline_predict(old_hgb,x[va]))
    for spec in specs_joint:
        t=time.perf_counter(); b=fit(spec,x[tr],y[tr]); duration=time.perf_counter()-t
        joint[spec['name']]=b; record(spec['name'],predict_fit(b,x[va]),duration)
    head_rows=[]; hybrid=pv.copy()
    for k in PRIMARY:
        best_error=mean_absolute_error(y[va,k],pv[:,k]); chosen=None
        head_rows.append({'target':targets[k],'candidate':'original_fast','mae':best_error,'r2':r2_score(y[va,k],pv[:,k])})
        for raw in specs_head:
            if raw['name']=='hgb_log' and k!=4: continue
            spec=dict(raw)
            if spec['family']=='et': spec['log']=(k==4)
            b=fit(spec,x[tr],y[tr],k); p=predict_fit(b,x[va]); error=mean_absolute_error(y[va,k],p)
            head_rows.append({'target':targets[k],'candidate':spec['name'],'mae':error,'r2':r2_score(y[va,k],p)})
            if error<best_error: best_error=error; chosen=b; hybrid[:,k]=p
        if chosen is not None: heads[k]=chosen
        pd.DataFrame(head_rows).to_csv(out/'validation_heads.csv',index=False)
        print('HEAD',targets[k], chosen['spec'] if chosen else 'original',best_error,flush=True)
    record('hybrid',hybrid)
    eligible=[row for row in rows if row['model']!='original_hgb']
    winner=min(eligible,key=lambda row:row['score'])['model']
    selected={'baseline':old,'joint':joint.get(winner),'heads':heads if winner=='hybrid' else {},
              'selected':winner,'targets':targets,'feature_names':names,'protocol':protocol}
    save_json(out/'selection.json',{'winner':winner,'joint':joint[winner]['spec'] if winner in joint else None,
       'heads':{str(k):v['spec'] for k,v in selected['heads'].items()},'test_evaluated_yet':False})
    joblib.dump(selected,out/'followup_predictor.joblib',compress=3)
    # Fixed-setting diagnostic across the seven training configurations only.
    cv=[]
    baseline_spec=dict(name='baseline',family='et',leaf=2,fraction=1.,features='original')
    for room in train:
        a=tr&(rooms!=room); b=tr&(rooms==room)
        refit=fit(baseline_spec,x[a],y[a]); bp=predict_fit(refit,x[b]); cp=bp.copy()
        if selected['joint'] is not None: cp=predict_fit(fit(selected['joint']['spec'],x[a],y[a]),x[b])
        for k,h in selected['heads'].items(): cp[:,k]=predict_fit(fit(h['spec'],x[a],y[a],k),x[b])
        for label,p in [('original_settings',bp),('selected_settings',cp)]:
            for k in PRIMARY: cv.append({'heldout_training_room':room,'model':label,'target':targets[k],
              'mae':mean_absolute_error(y[b,k],p[:,k]),'r2':r2_score(y[b,k],p[:,k])})
        pd.DataFrame(cv).to_csv(out/'training_configuration_cv.csv',index=False)
        print('CV',room,'saved',flush=True)
    abl=[]
    for mode in ['original','no_material','no_absolute_xyz']:
        spec=dict(name=mode,family='et',leaf=2,fraction=1.,features=mode,log=True)
        p=predict_fit(fit(spec,x[tr],y[tr],4),x[va]);
        abl.append({'features':mode,'delay_mae':mean_absolute_error(y[va,4],p),'delay_r2':r2_score(y[va,4],p)})
    pd.DataFrame(abl).to_csv(out/'delay_feature_ablation.csv',index=False)
    # Validate original predictions against saved metrics before new test reporting.
    pred_old=baseline_predict(old,x[te]); old_metrics=pd.read_csv(r/'fast_model/test_metrics.csv')
    for k in PRIMARY:
        reported=old_metrics[(old_metrics.target==targets[k])&(old_metrics.method=='predictor')].mae.iloc[0]
        assert np.isclose(mean_absolute_error(y[te,k],pred_old[:,k]),reported,atol=1e-6)
    loaded=joblib.load(out/'followup_predictor.joblib')
    assert np.allclose(predict(loaded,x[va]),predict(selected,x[va]))
    pp={'original_fast':pred_old,'followup':predict(loaded,x[te]),
        'original_hgb':baseline_predict(old_hgb,x[te]),
        'training_mean':np.tile(y[tr].mean(0),(te.sum(),1))}
    metrics=[]; records=[]; ids=np.where(te)[0]
    for label,p in pp.items():
        for room in ['ALL']+list(test):
            mask=np.ones(len(ids),bool) if room=='ALL' else rooms[te]==room
            for k,n in enumerate(targets): metrics.append({'model':label,'room':room,'target':n,
              'mae':mean_absolute_error(y[te][mask,k],p[mask,k]),'r2':r2_score(y[te][mask,k],p[mask,k])})
        for j,i in enumerate(ids):
            records.append({'model':label,'query_id':int(i),'room':rooms[i],
              **{'reference_'+n:float(y[i,k]) for k,n in enumerate(targets)},
              **{'predicted_'+n:float(p[j,k]) for k,n in enumerate(targets)}})
    met=pd.DataFrame(metrics); met.to_csv(out/'test_metrics_all16.csv',index=False)
    pd.DataFrame(records).to_csv(out/'test_predictions.csv',index=False)
    main=met[(met.room=='ALL')&met.target.isin([targets[k] for k in PRIMARY])]
    main.to_csv(out/'main_comparison.csv',index=False)
    print('FINAL EXPLORATORY TEST\n'+main.to_string(index=False),flush=True)
    prep=joblib.load(r/'fast_model/embedding_preprocessing.joblib'); room_order=train+val+test
    pooled_ref=[]; pooled_old=[]; pooled_new=[]
    for room in room_order:
        ix=(~random)&(rooms==room); assert ix.sum()==5
        pooled_ref.append(pool(y[ix]));pooled_old.append(pool(baseline_predict(old,x[ix])));pooled_new.append(pool(predict(loaded,x[ix])))
    em=lambda a:normalize(prep['scaler'].transform(a)*prep['family_weights'])
    er=em(pooled_ref);eo=em(pooled_old);en=em(pooled_new)
    pd.DataFrame({'room':room_order,'split':['train']*7+['validation']*2+['test']*2,
        'original_cosine':np.sum(er*eo,axis=1),'followup_cosine':np.sum(er*en,axis=1)}).to_csv(out/'embedding_comparison.csv',index=False)
    for label,a in [('reference',er),('original',eo),('followup',en)]:
        pd.DataFrame(a,index=room_order).to_csv(out/(label+'_80d_embeddings.csv'))
    # Copy code into the new output so imported modules never create baseline __pycache__.
    src=out/'baseline_source';src.mkdir(exist_ok=True)
    for name in ['experiment.py','geometry_io.py']: shutil.copy2(base/'source'/name,src/name)
    sys.path.insert(0,str(src)); os.environ['RF_SURROGATE_OUT']=str(out/'unused_simulation_output')
    import experiment as ex
    ex.SCENES=base/'source/scenes'
    geometries={room:ex.geometry(room)[0] for room in test}
    benchids=ids[np.linspace(0,len(ids)-1,30,dtype=int)]
    timing=[]
    for i in benchids:
        q=qs[i]; xx=ex.features(geometries[q['room']],q['tx'],q['rx'])
        assert np.allclose(xx,x[i],rtol=1e-9,atol=1e-9), 'Feature mismatch'
        for repeat in range(5):
            order=['original_fast','followup'] if repeat%2==0 else ['followup','original_fast']
            for label in order:
                t=time.perf_counter(); xx=ex.features(geometries[q['room']],q['tx'],q['rx'])
                baseline_predict(old,xx) if label=='original_fast' else predict(loaded,xx)
                timing.append({'query_id':int(i),'repeat':repeat,'model':label,'seconds':time.perf_counter()-t})
    timing=pd.DataFrame(timing);timing.to_csv(out/'runtime_repeats.csv',index=False)
    med=timing.groupby(['query_id','model']).seconds.median().unstack();med.to_csv(out/'runtime_per_query.csv')
    bm={'queries':30,'repeats':5,'original_median_ms':float(med.original_fast.median()*1000),
       'followup_median_ms':float(med.followup.median()*1000),
       'old_over_new_speed_ratio':float(med.original_fast.median()/med.followup.median()),
       'scope':'Warm CPU geometry feature extraction plus full16 inference, same runtime; excludes loading, training, disk and simulation',
       'new_sionna_speedup_measured':False,'platform':platform.platform(),'processor':platform.processor()}
    save_json(out/'runtime_comparison.json',bm)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,label in enumerate(['original_fast','followup']):
        for ax,k in zip(axes[row],PRIMARY):
            ax.scatter(y[te,k],pp[label][:,k],s=10,alpha=.5)
            lo=min(y[te,k].min(),pp[label][:,k].min());hi=max(y[te,k].max(),pp[label][:,k].max())
            ax.plot([lo,hi],[lo,hi],color='black');ax.set(title=label+' / '+targets[k],xlabel='Sionna reference',ylabel='Prediction')
    fig.tight_layout();fig.savefig(out/'quality_comparison.png',dpi=150);plt.close(fig)
    after={str(p.relative_to(base)):sha(p) for p in protected}
    assert before==after,'Original artifact changed'
    save_json(out/'original_preservation_verified.json',{'key_files_unchanged':True,'hashes':after})
    report='# RF surrogate follow-up: exploratory comparison\n\n'
    report+='Original artifacts preserved. Same 1,815 labels; 1,120 train, 320 validation, 320 test, 55 auxiliary anchors. All scenes share the original room family. No new independent rooms or RF measurements.\n\n'
    report+='Selection used validation only; test had previously been examined. Selected: '+winner+'.\n\n'
    report+=main.to_markdown(index=False)+'\n\nRuntime: '+json.dumps(bm,indent=2)+'\n\n'
    report+='The historical 2.54x Sionna speedup is not a new measurement. Grouped CV is a fixed-setting training-room diagnostic, not nested independent validation. Embeddings reuse original train-only preprocessing and five anchors per configuration. Better descriptor MAE alone does not establish reliable embeddings, physical consistency, real-world validity or new-building generalization.\n\n'
    report+='Files: validation_models.csv and validation_heads.csv (selection); training_configuration_cv.csv (configuration robustness); delay_feature_ablation.csv; main_comparison.csv and test_metrics_all16.csv; test_predictions.csv; runtime_comparison.json; embedding_comparison.csv; followup_predictor.joblib; protocol.json; original_preservation_verified.json.\n'
    (out/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    save_json(out/'COMPLETE.json',{'completed':True,'selected':winner,'original_preserved':True,'exploratory_test':True})
    print('RUNTIME',bm,flush=True);print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    with threadpool_limits(limits=1): run(a.base,a.out)
