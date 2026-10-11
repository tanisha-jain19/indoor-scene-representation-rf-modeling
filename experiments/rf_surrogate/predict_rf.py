"""Use the saved predictor without importing or running Sionna.

Example:
python predict_rf.py --model results/predictor.joblib --scene scenes/nochairroom \
  --tx 1 1.2 -1 --rx 3 1.2 -4

The scene directory must contain <directory-name>_materials.xml and meshes/*.ply.
Predictions outside the training room family are experimental, not validated.
"""
from pathlib import Path
import argparse,json,time
import joblib
import numpy as np
import experiment as e

class RFSession:
    """Load once, then answer repeated TX/RX queries without simulation."""
    def __init__(self,model_path,scene_directory):
        start=time.perf_counter()
        scene=Path(scene_directory).resolve();e.SCENES=scene.parent
        self.geometry,_=e.geometry(scene.name)
        self.bundle=joblib.load(model_path)
        self.load_seconds=time.perf_counter()-start

    def predict(self,tx,rx):
        start=time.perf_counter();tx=np.asarray(tx,dtype=float);rx=np.asarray(rx,dtype=float)
        if not e.clear(tx,self.geometry) or not e.clear(rx,self.geometry):
            raise ValueError('Position fails conservative clearance checks')
        if np.linalg.norm(tx-rx)<.8:raise ValueError('Minimum supported TX/RX separation is 0.8 m')
        y=e.predict(self.bundle,e.features(self.geometry,tx,rx))[0]
        return {'rf_descriptors':dict(zip(e.TARGETS,map(float,y))),
                'query_seconds':time.perf_counter()-start}

def predict_rf(model_path,scene_directory,tx,rx):
    start=time.perf_counter()
    scene=Path(scene_directory).resolve()
    e.SCENES=scene.parent
    g,geometry_seconds=e.geometry(scene.name)
    tx=np.asarray(tx,dtype=float);rx=np.asarray(rx,dtype=float)
    if not e.clear(tx,g) or not e.clear(rx,g):raise ValueError('Position fails conservative room/furniture clearance checks')
    if np.linalg.norm(tx-rx)<.8:raise ValueError('TX/RX separation is outside the trained range (minimum 0.8 m)')
    bundle=joblib.load(model_path)
    values=e.predict(bundle,e.features(g,tx,rx))[0]
    return {'rf_descriptors':dict(zip(e.TARGETS,map(float,values))),
            'total_seconds_including_loads':time.perf_counter()-start,
            'geometry_load_seconds':geometry_seconds,
            'scope':'fixed 3.5 GHz, single iso/V antennas, supplied room family; approximate simulated descriptors',
            'power_units':'path_gain_db is noncoherent path gain in dB, not received dBm'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--tx',nargs=3,type=float,required=True);p.add_argument('--rx',nargs=3,type=float,required=True)
    a=p.parse_args();print(json.dumps(predict_rf(a.model,a.scene,a.tx,a.rx),indent=2))
