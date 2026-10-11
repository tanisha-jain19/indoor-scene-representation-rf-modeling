"""Repeat matched simulator and full16 predictor queries on one runtime."""
import time,json,platform,subprocess
import numpy as np
import pandas as pd
import joblib
import experiment as ex

def main():
    qs,X,Y,_,_=ex.load_data()
    start=time.perf_counter();fast=joblib.load(ex.OUT/'fast_predictor.joblib');model_load=time.perf_counter()-start
    geom={r:ex.geometry(r)[0] for r in ex.TEST}
    ids=np.array([i for i,q in enumerate(qs) if q['room'] in ex.TEST and q['kind']=='random'])
    ids=ids[np.linspace(0,len(ids)-1,30,dtype=int)]
    rows=[];loads={}
    for room in ex.TEST:
        scene,loads[room]=ex.scene_for(room)
        subset=[i for i in ids if qs[i]['room']==room]
        q=qs[subset[0]];ex.solve(scene,q['tx'],q['rx'],ex.SEED+q['index'])
        for i in subset:
            q=qs[i];reference=ex.solve(scene,q['tx'],q['rx'],ex.SEED+q['index'])
            assert np.allclose(reference['y'][ex.PRIMARY],Y[i,ex.PRIMARY],rtol=1e-3,atol=.01),'Reference reproducibility failed'
            elapsed=[]
            for _ in range(3):
                start=time.perf_counter();ex.predict(fast,ex.features(geom[room],q['tx'],q['rx']))
                elapsed.append(time.perf_counter()-start)
            rows.append(dict(room=room,query_index=q['index'],sionna_seconds=reference['seconds'],predictor_seconds=float(np.median(elapsed))))
    df=pd.DataFrame(rows);df.to_csv(ex.OUT/'same_runtime_benchmark.csv',index=False)
    summary=dict(queries=len(rows),simulator_median_seconds=float(df.sionna_seconds.median()),
        full16_predictor_median_seconds=float(df.predictor_seconds.median()),
        warm_full16_speedup=float(df.sionna_seconds.median()/df.predictor_seconds.median()),
        reference_reproduction_passed=True,model_load_seconds=model_load,scene_load_seconds=loads,
        includes='Predictor geometry/link features plus full16 inference; synchronized simulator solve plus full16 descriptor extraction',
        excludes='One-time imports, model/scene loading, compilation, disk writes',
        python=platform.python_version(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip())
    ex.save_json(ex.OUT/'same_runtime_benchmark.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
