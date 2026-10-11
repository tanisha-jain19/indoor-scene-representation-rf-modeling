"""Explainable RF surrogate. All stages are explicit, reproducible and resumable."""
import os, sys, json, time, hashlib, argparse, platform
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RF_SURROGATE_OUT', str(ROOT/'results')))
SCENES=ROOT/'scenes'
SEED=20260923
TRAIN=['fullroom','nobedroom','nopartitionroom','room1','room2','room3','room5']
VAL=['fullyfurnishedroom','room4']
TEST=['nochairroom','nosofaroom']
TARGETS=['path_count','path_gain_db','dominant_fraction','mean_delay_ns','rms_delay_ns','delay_span_ns',
         'theta_rx_mean','theta_rx_std','theta_tx_mean','theta_tx_std',
         'phi_rx_sin','phi_rx_cos','phi_rx_spread','phi_tx_sin','phi_tx_cos','phi_tx_spread']
PRIMARY=[1,4,2]
FEATURE_NAMES=[]

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def geometry(room):
    from geometry_io import mesh_bounds
    start=time.perf_counter();root=SCENES/room
    bounds=mesh_bounds(root/'meshes');xml=ET.parse(root/f'{room}_materials.xml')
    mats={Path(s.find('string').get('value')).stem:s.find('ref').get('id') for s in xml.findall('shape')}
    return {'bounds':bounds,'materials':mats}, time.perf_counter()-start

def clear(p,g):
    bounds=g['bounds'];lo,hi=np.array(bounds['walls'])
    if np.any(p<=lo+.25) or np.any(p>=hi-.25):return False
    for n,(a,b) in bounds.items():
        if n in ('walls','floor'):continue
        if np.all(p>=np.array(a)-.15) and np.all(p<=np.array(b)+.15):return False
    return True

def segment_fraction(tx,rx,lo,hi):
    """Fraction of the segment inside an AABB, not an exact mesh LOS test."""
    d=rx-tx;t0,t1=0.,1.
    for k in range(3):
        if abs(d[k])<1e-12:
            if tx[k]<lo[k] or tx[k]>hi[k]:return 0.
        else:
            a,b=sorted(((lo[k]-tx[k])/d[k],(hi[k]-tx[k])/d[k]))
            t0=max(t0,a);t1=min(t1,b)
            if t1<=t0:return 0.
    return t1-t0

def features(g,tx,rx):
    global FEATURE_NAMES
    tx=np.array(tx);rx=np.array(rx);delta=rx-tx;dist=np.linalg.norm(delta)
    lo,hi=np.array(g['bounds']['walls']);vals=[];names=[]
    def add(label,x):
        for i,v in enumerate(np.atleast_1d(x)):names.append(f'{label}_{i}');vals.append(float(v))
    add('tx',tx);add('rx',rx);add('delta',delta);add('direction',delta/dist)
    add('distance',[dist,np.log10(dist)]);add('room_size',hi-lo)
    for label,p in [('tx',tx),('rx',rx)]:add(label+'_boundary',np.r_[p-lo,hi-p])
    for material in ['wood','metal','concrete']:
        count=volume=area=hits=length=0.;near_tx=[];near_rx=[]
        for n,(a,b) in g['bounds'].items():
            if n in ('floor','walls') or material not in g['materials'][n]:continue
            a=np.array(a);b=np.array(b);s=b-a;count+=1;volume+=np.prod(s)
            area+=2*(s[0]*s[1]+s[0]*s[2]+s[1]*s[2])
            fraction=segment_fraction(tx,rx,a,b);hits+=fraction>0;length+=fraction*dist
            near_tx.append(np.linalg.norm(np.maximum(np.maximum(a-tx,tx-b),0)))
            near_rx.append(np.linalg.norm(np.maximum(np.maximum(a-rx,rx-b),0)))
        add(material,[count,volume,area,hits,length,min(near_tx,default=20),min(near_rx,default=20)])
    FEATURE_NAMES=names
    return np.array(vals,dtype=np.float64)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg={'seed':SEED,'frequency_hz':3.5e9,'random_links_per_room':160,'samples_per_src':200000,
         'max_depth':3,'train_rooms':TRAIN,'validation_rooms':VAL,'test_rooms':TEST,
         'targets':TARGETS,'primary_indices':PRIMARY,'antenna':'single isotropic V at TX and RX',
         'power_definition':'10log10(sum(abs(path_coefficient)^2)); noncoherent path gain, not dBm',
         'scope':'held-out configurations of supplied room family, not new building generalization',
         'coordinate_system':'exported coordinates in metres; Y up',
         'effects':'LOS and specular reflection only; no diffuse, refraction or diffraction'}
    hashes={str(p.relative_to(SCENES)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SCENES.rglob('*')) if p.is_file()}
    cfg['assets_sha256']=hashes
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==cfg,'Configuration changed; use new output folder'
    save_json(OUT/'config.json',cfg)
    queries=[];rng=np.random.default_rng(SEED);audit={}
    anchors=np.array([[1,1.2,-4],[3,1.2,-4],[5,1.2,-4],[7,1.2,-4],[4,1.2,-1.]])
    for room in TRAIN+VAL+TEST:
        g,sec=geometry(room);audit[room]={**g,'load_seconds':sec}
        def sample():
            lo,hi=np.array(g['bounds']['walls'])
            for _ in range(100000):
                p=rng.uniform(lo+.35,hi-.35);p[1]=rng.uniform(.8,2.)
                if clear(p,g):return p
            raise RuntimeError('No valid positions: '+room)
        for i in range(cfg['random_links_per_room']):
            tx=sample();rx=sample()
            while np.linalg.norm(tx-rx)<.8:rx=sample()
            queries.append({'room':room,'kind':'random','index':i,'tx':tx.tolist(),'rx':rx.tolist()})
        # Explicitly identified auxiliary queries for 80-D pooled embeddings.
        tx=np.array([.6,1.8,-.6])
        if not clear(tx,g):tx=sample()
        for i,rx in enumerate(anchors):
            if not clear(rx,g):
                raise RuntimeError(f'Anchor RX {i} intersects conservative bounds in {room}; revise common anchors before simulation')
            queries.append({'room':room,'kind':'anchor','index':i,'tx':tx.tolist(),'rx':rx.tolist()})
    save_json(OUT/'geometry_audit.json',audit);save_json(OUT/'queries.json',queries)
    features(g,tx,rx);save_json(OUT/'feature_names.json',FEATURE_NAMES)
    print('PREPARED',len(queries),'queries;',len(FEATURE_NAMES),'features',flush=True)
    return cfg,queries

def scene_for(room=None):
    from sionna.rt import load_scene,PlanarArray,Transmitter,Receiver
    start=time.perf_counter()
    scene=load_scene(None if room is None else str(SCENES/room/f'{room}_materials.xml'),merge_shapes=False)
    scene.frequency=3.5e9
    scene.tx_array=PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
    scene.rx_array=PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
    scene.add(Transmitter(name='tx',position=[1,1.2,-1]))
    scene.add(Receiver(name='rx',position=[2,1.2,-1]))
    return scene,time.perf_counter()-start

def descriptors(a,tau,angles):
    a=np.asarray(a).reshape(-1);tau=np.asarray(tau).reshape(-1)
    assert len(a)==len(tau),(a.shape,tau.shape)
    valid=(tau>=0)&np.isfinite(tau)&np.isfinite(a)&(np.abs(a)>0)
    p=np.abs(a[valid])**2;t=tau[valid]*1e9
    if not len(p) or p.sum()<=0:raise ValueError('No valid propagation path')
    w=p/p.sum();mean=np.sum(w*t)
    out=[len(p),10*np.log10(p.sum()),p.max()/p.sum(),t.mean(),np.sqrt(np.sum(w*(t-mean)**2)),np.ptp(t)]
    theta_r,phi_r,theta_t,phi_t=[np.asarray(v).reshape(-1)[valid] for v in angles]
    out.extend([theta_r.mean(),theta_r.std(),theta_t.mean(),theta_t.std()])
    for phi in [phi_r,phi_t]:
        s,c=np.sin(phi).mean(),np.cos(phi).mean();spread=np.sqrt(max(0,-2*np.log(max(np.hypot(s,c),1e-12))))
        out.extend([s,c,spread])
    y=np.array(out);assert y.shape==(16,) and np.isfinite(y).all()
    return y

def solve(scene,tx,rx,seed,rays=200000,depth=3):
    from sionna.rt import PathSolver
    import drjit as dr
    scene.get('tx').position=tx;scene.get('rx').position=rx
    dr.sync_thread();start=time.perf_counter()
    paths=PathSolver()(scene,max_depth=depth,samples_per_src=rays,max_num_paths_per_src=200000,
        synthetic_array=True,los=True,specular_reflection=depth>0,diffuse_reflection=False,
        refraction=False,diffraction=False,seed=int(seed))
    a,tau=paths.cir(normalize_delays=False,out_type='numpy')
    if isinstance(a,(tuple,list)):a=np.asarray(a[0])+1j*np.asarray(a[1])
    a=np.asarray(a).reshape(-1);tau=np.asarray(tau).reshape(-1)
    angles=[np.asarray(v).reshape(-1) for v in (paths.theta_r,paths.phi_r,paths.theta_t,paths.phi_t)]
    y=descriptors(a,tau,angles);dr.sync_thread();seconds=time.perf_counter()-start
    return dict(y=y,a=a,tau=tau,theta_r=angles[0],phi_r=angles[1],theta_t=angles[2],phi_t=angles[3],seconds=seconds)

def numerical_checks():
    scene,_=scene_for();tx=[1.,1.2,-1.];rx=[3.,1.2,-1.]
    out=solve(scene,tx,rx,SEED,rays=1000,depth=0)
    expected=20*np.log10((299792458/3.5e9)/(4*np.pi*2))
    err=abs(out['y'][1]-expected)
    assert err<.1,(expected,out['y'][1])
    assert abs(out['tau'][out['tau']>=0].min()-2/299792458)<1e-10
    assert out['y'][4]<1e-5 and abs(out['y'][2]-1)<1e-6
    save_json(OUT/'free_space_check.json',{'passed':True,'power_error_db':float(err),'expected_gain_db':float(expected)})
    print('PASS free-space gain and delay checks',flush=True)

def generate(pilot=False):
    cfg,queries=prepare();numerical_checks();raw=OUT/'raw';raw.mkdir(exist_ok=True)
    times={};convergence=[]
    for room in TRAIN+VAL+TEST:
        scene,load_sec=scene_for(room);g,_=geometry(room)
        material={n:{'relative_permittivity':float(np.asarray(m.relative_permittivity).item()),'conductivity':float(np.asarray(m.conductivity).item())} for n,m in scene.radio_materials.items()}
        save_json(OUT/'materials'/f'{room}.json',material)
        if any('metal' in m for m in g['materials'].values()):
            assert any(v['conductivity']>1e6 for v in material.values()),'Expected metal assignment missing'
        assert any(v['relative_permittivity']>1 for v in material.values()),'Non-vacuum materials missing'
        qs=[q for q in queries if q['room']==room]
        if pilot:qs=qs[:2]
        times[room]=load_sec
        for j,q in enumerate(qs):
            dest=raw/f'{room}_{q["kind"]}_{q["index"]:04d}.npz'
            if dest.exists():continue
            t=time.perf_counter();x=features(g,q['tx'],q['rx']);feature_sec=time.perf_counter()-t
            out=solve(scene,q['tx'],q['rx'],SEED+q['index'])
            tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,x=x,feature_seconds=feature_sec,**out);tmp.replace(dest)
            if j%20==0:print(room,j+1,'/',len(qs),'paths',int(out['y'][0]),'seconds',round(out['seconds'],3),flush=True)
        # Check the first two fixed queries per scene; do not alter settings based on test prediction errors.
        checkfile=OUT/'convergence'/f'{room}.json'
        if not checkfile.exists():
            rows=[]
            for q in qs[:2]:
                ref=np.load(raw/f'{room}_{q["kind"]}_{q["index"]:04d}.npz')['y']
                high=solve(scene,q['tx'],q['rx'],SEED+q['index'],rays=400000)['y']
                diff=np.abs(high-ref)
                passed=bool(diff[1]<.5 and diff[4]<max(.5,.05*high[4]) and diff[2]<.05)
                rows.append({'query':q['index'],'absolute_diff':diff.tolist(),'passed':passed})
            save_json(checkfile,rows)
        rows=json.loads(checkfile.read_text());convergence.extend(rows)
        if not all(r['passed'] for r in rows):raise RuntimeError(f'Ray convergence check failed in {room}; inspect before continuing')
        print('Completed',room,'pilot' if pilot else 'dataset',flush=True)
    save_json(OUT/'scene_load_seconds.json',times)
    save_json(OUT/'generation_status.json',{'pilot':pilot,'raw_files':len(list(raw.glob('*.npz'))),'convergence_checks':len(convergence),'complete':not pilot})

def load_data():
    cache=OUT/'dataset_arrays.npz'
    query_text=(OUT/'queries.json').read_text()
    if cache.exists():
        with np.load(cache) as z:
            assert str(z['query_sha256'])==hashlib.sha256(query_text.encode()).hexdigest(),'Dataset cache/query mismatch'
            return json.loads(query_text),z['x'],z['y'],z['seconds'],z['feature_seconds']
    queries=json.loads((OUT/'queries.json').read_text());xs=[];ys=[];seconds=[];ft=[]
    for q in queries:
        z=np.load(OUT/'raw'/f'{q["room"]}_{q["kind"]}_{q["index"]:04d}.npz')
        xs.append(z['x']);ys.append(z['y']);seconds.append(float(z['seconds']));ft.append(float(z['feature_seconds']))
    return queries,np.array(xs),np.array(ys),np.array(seconds),np.array(ft)

def transform_y(y):
    z=np.array(y,copy=True)
    for k in [0,3,4,5]:z[:,k]=np.log1p(z[:,k])
    return z

def inverse_y(z):
    y=np.array(z,copy=True)
    for k in [0,3,4,5]:y[:,k]=np.maximum(0,np.expm1(y[:,k]))
    y[:,2]=np.clip(y[:,2],0,1)
    for k in [10,11,13,14]:y[:,k]=np.clip(y[:,k],-1,1)
    return y

def train():
    import joblib,pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    qs,x,y,seconds,ft=load_data();rooms=np.array([q['room'] for q in qs]);random=np.array([q['kind']=='random' for q in qs])
    tr=np.isin(rooms,TRAIN)&random;va=np.isin(rooms,VAL)&random;te=np.isin(rooms,TEST)&random
    assert not (tr&va).any() and not(tr&te).any()
    zy=transform_y(y);yscale=StandardScaler().fit(zy[tr]);z=yscale.transform(zy)
    candidates={
      'extra_trees_leaf2':ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,max_features=1.,random_state=SEED,n_jobs=2),
      'extra_trees_leaf5':ExtraTreesRegressor(n_estimators=250,min_samples_leaf=5,max_features=1.,random_state=SEED,n_jobs=2),
      'hist_gradient':MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=SEED)),
      'lookup_knn5':make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5,weights='distance'))}
    scales=np.maximum(y[tr][:,PRIMARY].std(0),1e-8);rows=[];models={}
    for name,model in candidates.items():
        start=time.perf_counter();model.fit(x[tr],z[tr]);duration=time.perf_counter()-start
        p=inverse_y(yscale.inverse_transform(model.predict(x[va])))
        mae=np.abs(p[:,PRIMARY]-y[va][:,PRIMARY]).mean(0);score=float(np.mean(mae/scales))
        rows.append({'model':name,'validation_normalized_mae':score,'gain_mae_db':mae[0],'rms_delay_mae_ns':mae[1],'dominant_fraction_mae':mae[2],'train_seconds':duration})
        models[name]=model;print(rows[-1],flush=True)
    selected=min((r for r in rows if r['model']!='lookup_knn5'),key=lambda r:r['validation_normalized_mae'])['model']
    pd.DataFrame(rows).to_csv(OUT/'validation_models.csv',index=False)
    bundle={'model':models[selected],'target_scaler':yscale,'selected':selected,'lookup':models['lookup_knn5'],'mean_y':y[tr].mean(0),'feature_names':FEATURE_NAMES or json.loads((OUT/'feature_names.json').read_text()),'targets':TARGETS,'train_rooms':TRAIN,'validation_rooms':VAL,'test_rooms':TEST}
    joblib.dump(bundle,OUT/'predictor.joblib',compress=3)
    save_json(OUT/'training_status.json',{'selected':selected,'train_examples':int(tr.sum()),'validation_examples':int(va.sum()),'test_examples':int(te.sum()),'test_not_used_for_selection':True})
    print('SAVED predictor. Run evaluate in a fresh process.',flush=True)

def predict(bundle,x,lookup=False):
    model=bundle['lookup' if lookup else 'model']
    return inverse_y(bundle['target_scaler'].inverse_transform(model.predict(np.atleast_2d(x))))

def evaluate():
    report_dir=OUT/os.environ.get('RF_REPORT_SUBDIR','')
    report_dir.mkdir(parents=True,exist_ok=True)
    import joblib,pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error,r2_score
    start=time.perf_counter();b=joblib.load(OUT/os.environ.get('RF_MODEL_FILE','predictor.joblib'));model_load=time.perf_counter()-start
    qs,x,y,sec,ft=load_data();test=np.array([q['room'] in TEST and q['kind']=='random' for q in qs]);ids=np.where(test)[0]
    predictions={'predictor':predict(b,x[test]),'lookup_knn5':predict(b,x[test],True),'training_mean':np.tile(b['mean_y'],(sum(test),1))}
    rows=[]
    for name,p in predictions.items():
        for k in PRIMARY:
            rows.append({'method':name,'target':TARGETS[k],'mae':mean_absolute_error(y[test,k],p[:,k]),'r2':r2_score(y[test,k],p[:,k])})
    pd.DataFrame(rows).to_csv(report_dir/'test_metrics.csv',index=False)
    per_room=[];records=[]
    for room in TEST:
        subset=np.array([qs[i]['room']==room for i in ids])
        for k in PRIMARY:per_room.append({'room':room,'target':TARGETS[k],'mae':float(np.abs(predictions['predictor'][subset,k]-y[ids[subset],k]).mean())})
    pd.DataFrame(per_room).to_csv(report_dir/'test_metrics_by_room.csv',index=False)
    for j,i in enumerate(ids):
        rec={**qs[i]}
        for k,n in enumerate(TARGETS):rec['reference_'+n]=y[i,k];rec['predicted_'+n]=predictions['predictor'][j,k]
        records.append(rec)
    pd.DataFrame(records).to_csv(report_dir/'test_predictions.csv',index=False)
    # Each timed query includes feature extraction and a one-query prediction.
    geom={room:geometry(room) for room in TEST};dur=[]
    predict(b,x[ids[0]])
    benchmark_ids=ids[np.linspace(0,len(ids)-1,min(100,len(ids)),dtype=int)]
    for i in benchmark_ids:
        q=qs[i];start=time.perf_counter();xx=features(geom[q['room']][0],q['tx'],q['rx']);predict(b,xx);dur.append(time.perf_counter()-start)
    timing={'model_load_seconds':model_load,'geometry_load_seconds':{r:g[1] for r,g in geom.items()},
        'warm_predictor_single_query_median_seconds':float(np.median(dur)),
        'sionna_matched_queries_median_seconds':float(np.median(sec[benchmark_ids])),
        'warm_median_speedup':float(np.median(sec[benchmark_ids])/np.median(dur)),
        'sionna_scene_load_seconds':json.loads((OUT/'scene_load_seconds.json').read_text()),
        'dataset_solve_seconds':float(sec.sum()),'timing_scope':'same Colab host; synchronized GPU simulation plus descriptors versus CPU features plus one-query prediction; loaded scene/model; compilation and cold setup reported separately'}
    save_json(report_dir/'timing.json',timing)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,k in zip(axes,PRIMARY):
        ax.scatter(y[test,k],predictions['predictor'][:,k],s=12,alpha=.5)
        lo=min(y[test,k].min(),predictions['predictor'][:,k].min());hi=max(y[test,k].max(),predictions['predictor'][:,k].max())
        ax.plot([lo,hi],[lo,hi],color='black');ax.set(xlabel='Sionna reference',ylabel='Prediction',title=TARGETS[k])
    fig.tight_layout();fig.savefig(report_dir/'prediction_quality.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,k in zip(axes,PRIMARY):
        rr=[r for r in rows if r['target']==TARGETS[k]];ax.bar([r['method'] for r in rr],[r['mae'] for r in rr]);ax.set_title(TARGETS[k]+' MAE');ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(report_dir/'baseline_comparison.png',dpi=150);plt.close(fig)
    if hasattr(b['model'],'feature_importances_'):
        importance=pd.DataFrame({'feature':b['feature_names'],'importance':b['model'].feature_importances_}).sort_values('importance',ascending=False)
        importance.to_csv(report_dir/'feature_importances.csv',index=False)
    # New transparent embedding definition; fitted on reference anchors of training rooms only.
    roomnames=TRAIN+VAL+TEST;ref=[];pred=[]
    def pool(a):return np.stack([a.mean(0),a.std(0),a.min(0),a.max(0),np.median(a,0)],axis=1).reshape(-1)
    for room in roomnames:
        ix=[i for i,q in enumerate(qs) if q['room']==room and q['kind']=='anchor']
        assert len(ix)==5
        ref.append(pool(y[ix]));pred.append(pool(predict(b,x[ix])))
    from sklearn.preprocessing import StandardScaler,normalize
    sc=StandardScaler().fit(np.array(ref)[:len(TRAIN)])
    # Balance four feature families (count, power, delay, angular) by coordinate counts.
    weights=np.repeat([1,1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]+[1/np.sqrt(10)]*10,5)
    er=normalize(sc.transform(ref)*weights);ep=normalize(sc.transform(pred)*weights)
    pd.DataFrame(er,index=roomnames).to_csv(report_dir/'reference_80d_embeddings.csv');pd.DataFrame(ep,index=roomnames).to_csv(report_dir/'predicted_80d_embeddings.csv')
    pd.DataFrame({'room':roomnames,'reference_prediction_cosine':np.sum(er*ep,axis=1),'split':['train']*len(TRAIN)+['validation']*len(VAL)+['test']*len(TEST)}).to_csv(report_dir/'embedding_quality.csv',index=False)
    joblib.dump({'scaler':sc,'family_weights':weights,'pool_order':'per feature: mean,std,min,max,median','targets':TARGETS},report_dir/'embedding_preprocessing.joblib')
    pd.DataFrame(rows).to_json(report_dir/'metrics.json',orient='records',indent=2)
    save_json(report_dir/'evaluation_complete.json',{'fresh_process_reload':True,'held_out_predictions':int(sum(test)),'model':b['selected'],'completed':True})
    report='# RF surrogate experiment\n\n'
    report+='The predictor estimates 16 Sionna descriptors from 50 geometry/link features. It receives no RF input. Primary evaluation concerns path gain (not dBm), RMS delay spread and dominant-path fraction.\n\n'
    report+=f'Selected model: {b["selected"]}. Training rooms: {TRAIN}. Validation: {VAL}. Final test: {TEST}. All scenes share the supplied room family. This is configuration generalization, not a claim about unseen buildings.\n\n'
    report+=pd.DataFrame(rows).to_string(index=False)+'\n\n'+json.dumps(timing,indent=2)
    report+='\n\nThe additional 80-D embeddings use five anchor receivers and train-only scaling. They are a newly documented construction, not guaranteed identical to historical embeddings. No real-world RF measurements were used. AABB obstruction features approximate geometry. Simulation omits diffuse reflection, diffraction and refraction. Low errors or high speed alone do not establish deployment suitability.\n'
    (report_dir/'MENTOR_REPORT.md').write_text(report,encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(timing,indent=2));print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','generate','train','evaluate']);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage in ('pilot','generate'):generate(args.stage=='pilot')
    elif args.stage=='train':train()
    else:evaluate()
