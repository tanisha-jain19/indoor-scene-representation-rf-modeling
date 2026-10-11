"""Frozen-model, matched-query benchmark. No fitting or model selection."""
import argparse, os, sys, json, time, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd, joblib
from threadpoolctl import threadpool_limits
import followup as f

def run(base,out):
    sys.path.insert(0,str(out/'baseline_source'));os.environ['RF_SURROGATE_OUT']=str(out/'benchmark_aux')
    import experiment as ex
    ex.SCENES=base/'source/scenes'
    start=time.perf_counter();b=joblib.load(out/'followup_predictor.joblib');load=time.perf_counter()-start
    old=b['baseline'];old['model'].set_params(n_jobs=1)
    qs=json.loads((base/'results/queries.json').read_text())
    with np.load(base/'results/dataset_arrays.npz') as z:y=z['y']
    ids=np.array([i for i,q in enumerate(qs) if q['room'] in old['test_rooms'] and q['kind']=='random'])
    ids=ids[np.linspace(0,len(ids)-1,30,dtype=int)];rows=[];loads={}
    for room in old['test_rooms']:
        g,_=ex.geometry(room);scene,loads[room]=ex.scene_for(room)
        subset=[i for i in ids if qs[i]['room']==room];q=qs[subset[0]]
        ex.solve(scene,q['tx'],q['rx'],ex.SEED+q['index'])
        for i in subset:
            q=qs[i];sim=ex.solve(scene,q['tx'],q['rx'],ex.SEED+q['index'])
            assert np.allclose(sim['y'][f.PRIMARY],y[i,f.PRIMARY],rtol=1e-3,atol=.01),'Sionna reference mismatch'
            dt={'original':[],'followup':[]}
            for repeat in range(5):
                for label in (['original','followup'] if repeat%2==0 else ['followup','original']):
                    start=time.perf_counter();xx=ex.features(g,q['tx'],q['rx'])
                    f.baseline_predict(old,xx) if label=='original' else f.predict(b,xx)
                    dt[label].append(time.perf_counter()-start)
            rows.append({'room':room,'query_id':int(i),'sionna_seconds':sim['seconds'],
               'original_seconds':float(np.median(dt['original'])),'followup_seconds':float(np.median(dt['followup']))})
        pd.DataFrame(rows).to_csv(out/'same_runtime_sionna_benchmark.csv',index=False)
        print('BENCHMARK',room,'saved',flush=True)
    df=pd.DataFrame(rows);sim=float(df.sionna_seconds.median());oldtime=float(df.original_seconds.median());newtime=float(df.followup_seconds.median())
    summary={'queries':len(df),'sionna_median_ms':1000*sim,'original_median_ms':1000*oldtime,
       'followup_median_ms':1000*newtime,'original_sionna_speedup':sim/oldtime,'followup_sionna_speedup':sim/newtime,
       'original_over_followup_speed_ratio':oldtime/newtime,'model_load_seconds':load,'scene_load_seconds':loads,
       'reference_reproduction_passed':True,'scope':'Same runtime: synchronized Sionna solve+16 descriptors vs CPU features+16 predictions; warm, no loading/compilation/disk/training',
       'python':platform.python_version(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()}
    f.save_json(out/'same_runtime_sionna_benchmark.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.base,a.out)
