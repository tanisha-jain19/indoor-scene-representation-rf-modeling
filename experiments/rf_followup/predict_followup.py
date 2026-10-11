"""Predict 16 descriptors from room geometry and TX/RX without importing Sionna."""
import argparse, json, os, sys
from pathlib import Path
import joblib
import followup as f
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path)
    p.add_argument('--scenes',type=Path,help='Portable scene directory; replaces the original --base/source/scenes layout')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--room',required=True)
    p.add_argument('--tx',nargs=3,type=float,required=True);p.add_argument('--rx',nargs=3,type=float,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.out/'baseline_source'))
    os.environ['RF_SURROGATE_OUT']=str(a.out/'unused_simulation_output')
    import experiment as ex
    if not a.scenes and not a.base:p.error('Provide --scenes or the original --base directory')
    ex.SCENES=a.scenes if a.scenes else a.base/'source/scenes'
    b=joblib.load(a.out/'followup_predictor.joblib');g,_=ex.geometry(a.room)
    with threadpool_limits(limits=1): y=f.predict(b,ex.features(g,a.tx,a.rx))[0]
    assert not any(m=='sionna' or m.startswith('sionna.') for m in sys.modules)
    print(json.dumps({'room':a.room,'tx':a.tx,'rx':a.rx,'prediction':dict(zip(b['targets'],map(float,y))),
        'sionna_imported':False,'scope':'Same supplied room family, frequency and simulated physics only'},indent=2))

if __name__=='__main__':main()
