"""Assemble reports and verified archives without fitting or changing predictions."""
import argparse, hashlib, json, platform, subprocess, sys, zipfile
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(base,out):
    assert (out/'COMPLETE.json').exists()
    before=json.loads((out/'original_hashes_before.json').read_text())
    after={name:sha(base/name) for name in before}
    assert before==after
    (out/'original_preservation_verified.json').write_text(json.dumps({'key_files_unchanged':True,'hashes':after},indent=2))
    benchmark=json.loads((out/'same_runtime_sionna_benchmark.json').read_text())
    assert benchmark['reference_reproduction_passed']
    main=pd.read_csv(out/'main_comparison.csv')
    assert len(main)==12 and main[['mae','r2']].notna().all().all()
    standalone=json.loads((out/'standalone_prediction.json').read_text())
    assert standalone['sionna_imported'] is False and len(standalone['prediction'])==16
    required=['followup_predictor.joblib','validation_models.csv','validation_heads.csv',
       'training_configuration_cv.csv','delay_feature_ablation.csv','validation_group_importance.csv',
       'test_metrics_all16.csv','test_predictions.csv','embedding_comparison.csv','same_runtime_sionna_benchmark.csv']
    assert all((out/p).is_file() and (out/p).stat().st_size for p in required)
    summary=(out/'MENTOR_SUMMARY.md').read_text()
    (out/'FINAL_REPORT.md').write_text(summary+'\n\n## Saved evidence\n\n'+
       '\n'.join('- `'+p+'`' for p in required)+
       '\n\nThe proposed changes were selected on validation; no subsequent test-guided model changes were made. Original work is retained as baseline. No new independent-scene generalization claim.\n',encoding='utf-8')
    (out/'environment_final.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    state={'status':'Complete: exploratory follow-up and matched T4 benchmark saved',
      'notebook_url':'https://colab.research.google.com/drive/1Nyv6Va1BYxsEKEiiAIbIWwtkxkdsCi6V',
      'original_unchanged':True,'selected_candidate':'hybrid','recommended_baseline':'original_fast',
      'overall_upgrade':False,'reason':'Gain/fraction MAE improve; delay, test embeddings and full16 speed regress',
      'test_is_exploratory':True,'training_complete':True,'diagnostics_complete':True,'gpu_benchmark_complete':True,
      'results_folder':str(out),'original_folder':str(base),'new_training_needed':False}
    (out/'STATUS.json').write_text(json.dumps(state,indent=2))
    (out/'RESUME.md').write_text('# Completed follow-up\n\n'+json.dumps(state,indent=2)+'\n\nDo not retrain automatically. Read FINAL_REPORT.md and comparison CSVs. The original experiment remains unchanged. All expected work is complete.\n')
    # A clean review notebook reads saved results and can run a standalone prediction.
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
    md('# RF surrogate follow-up — saved-results review\nNo retraining. The original remains the baseline. See FINAL_REPORT.md for the mixed outcome.')
    code("from google.colab import drive\ndrive.mount('/content/drive')\nfrom pathlib import Path\nimport json, pandas as pd\nOUT=Path('/content/drive/MyDrive/RF_Surrogate_Followup_20261004')\nassert (OUT/'STATUS.json').exists()\nprint((OUT/'FINAL_REPORT.md').read_text())")
    md('## Accuracy and configuration differences\nLower MAE is better; R2 compares squared prediction error with a constant reference based on that evaluation set. Negative R2 is worse than that reference. Test results are exploratory.')
    code("display(pd.read_csv(OUT/'main_comparison.csv'))\ndisplay(pd.read_csv(OUT/'test_metrics_all16.csv').query(\"model in ['original_fast','followup'] and target in ['path_gain_db','rms_delay_ns','dominant_fraction']\"))")
    md('## Selection and diagnostics\nOnly validation selected the model. Seven training-configuration folds are diagnostics, not nested CV.')
    code("display(pd.read_csv(OUT/'validation_models.csv'))\ndisplay(pd.read_csv(OUT/'validation_heads.csv'))\ndisplay(pd.read_csv(OUT/'training_cv_summary.csv'))\ndisplay(pd.read_csv(OUT/'delay_feature_ablation.csv'))")
    md('## Embeddings and timing\nOriginal anchor pooling and scaling were preserved. The timing includes full16 predictions and features, with loaded models/scenes.')
    code("display(pd.read_csv(OUT/'embedding_comparison.csv'))\nprint((OUT/'same_runtime_sionna_benchmark.json').read_text())\nfrom IPython.display import Image, display\ndisplay(Image(filename=str(OUT/'quality_comparison.png')))\nprint((OUT/'original_preservation_verified.json').read_text())")
    (out/'REVIEW_RESULTS.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=2))
    files=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['MANIFEST_SHA256.json','ARCHIVE_VERIFIED.json']]
    manifest={str(p.relative_to(out)):sha(p) for p in files}
    (out/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2))
    archive=out.parent/'RF_Surrogate_Followup_20261004.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'MANIFEST_SHA256.json']:z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items(): assert hashlib.sha256(z.read(name)).hexdigest()==h
    verified={'archive':str(archive),'files':len(manifest)+1,'bytes':archive.stat().st_size,'sha256':sha(archive),'crc_and_member_hashes_verified':True}
    (out/'ARCHIVE_VERIFIED.json').write_text(json.dumps(verified,indent=2))
    print(json.dumps(state,indent=2));print(json.dumps(verified,indent=2));print('FINAL SAVE AND VERIFICATION COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.base,a.out)
