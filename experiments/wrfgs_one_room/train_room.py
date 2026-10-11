"""Single-room adapter using the pinned authors' Gaussian/Deform models and CUDA renderer.

Changes: explicit float data, receiver-centred coordinates, smaller bounded initial
Gaussian count, validation-only selection, complete checkpoints and final-test export.
This is an adaptation, not a claim to reproduce the authors' benchmark scores.
"""
import argparse, csv, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor'/'WRF-GSplus'
sys.path.insert(0,str(VENDOR))
OUT=Path(os.environ.get('WRF_RUN_DIR',str(ROOT/'run_fullroom')))

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def imports():
    global torch, GaussianModel, DeformModel, render, generate_new_cam, l1_loss, torch_ssim
    import torch
    from scene import GaussianModel, DeformModel
    from gaussian_renderer import render
    from utils.generate_camera import generate_new_cam
    from utils.loss_utils import l1_loss, ssim as torch_ssim
    assert torch.cuda.is_available(),'Select a Colab NVIDIA GPU runtime first.'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False

def defaults():
    from arguments import OptimizationParams, PipelineParams
    parser=argparse.ArgumentParser(); op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args([]);opt=op.extract(args);pipe=pp.extract(args)
    opt.iterations=30000;opt.position_lr_max_steps=30000;opt.deform_lr_max_steps=30000
    opt.densify_until_iter=6000
    return opt,pipe

def initial_model(n,seed=2026):
    from utils.graphics_utils import BasicPointCloud
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    g=GaussianModel(0,'default')
    # Virtual sources can lie outside the physical room. Cover a surrounding volume.
    xyz=np.random.default_rng(seed).uniform([-10,-5,-10],[12,6,10],(n,3)).astype('float32')
    pcd=BasicPointCloud(points=xyz,colors=np.full((n,3),.05,dtype='float32'))
    g.create_from_pcd(pcd,[],5)
    opt,pipe=defaults();g.training_setup(opt)
    d=DeformModel();d.train_setting(opt)
    cam=generate_new_cam(np.eye(3,dtype='float32'),[0,0,0])
    return g,d,opt,pipe,cam

def predict(g,d,pipe,cam,tx):
    pos=tx.expand(g.get_xyz.shape[0],-1)
    offsets=d.step(g.get_xyz.detach(),pos)
    # separate_sh=False is important: the published branch adds the deformation signal here.
    result=render(cam,g,pipe,torch.zeros(3,device='cuda'),*offsets,separate_sh=False)
    image=result['render']
    pred=torch.linalg.vector_norm(image[:2],dim=0)
    assert pred.shape==(90,360),pred.shape
    return pred,result

def inputs():
    assert (OUT/'data_ready.json').exists(),'Run full data generation first.'
    ready=json.loads((OUT/'data_ready.json').read_text())
    assert hashlib.sha256((OUT/'spectra.npy').read_bytes()).hexdigest()==ready['spectra_sha256']
    x=np.load(OUT/'tx_model.npy').astype(np.float32);y=np.load(OUT/'spectra.npy').astype(np.float32)
    splits={s:np.load(OUT/(s+'_indices.npy')) for s in ('train','val','test')}
    all_ids=np.concatenate(list(splits.values()))
    assert len(np.unique(all_ids))==len(y)==600
    return x,y,splits,ready['dataset_id']+':'+ready['spectra_sha256']

def save_checkpoint(path,g,d,opt,iteration,best,data_id):
    # Both the Gaussians AND deformation network are required to reproduce predictions.
    state=dict(gaussians=g.capture(),deform=d.deform.state_dict(),
        deform_optimizer=d.optimizer.state_dict(),iteration=iteration,best_val=best,
        data_id=data_id,opt=vars(opt),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def restore(path,g,d,opt,data_id):
    # Load only checkpoints created by this notebook in your own output folder.
    state=torch.load(path,map_location='cuda',weights_only=False)
    assert state['data_id']==data_id,'Checkpoint belongs to a different dataset.'
    assert state['opt']==vars(opt),'Training settings differ from checkpoint.'
    g.restore(state['gaussians'],opt);d.deform.load_state_dict(state['deform'])
    d.optimizer.load_state_dict(state['deform_optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state

def validation(g,d,pipe,cam,x,y,ids):
    from skimage.metrics import structural_similarity
    errors=[];scores=[]
    with torch.no_grad():
        for i in ids:
            pred,_=predict(g,d,pipe,cam,x[i]);a=pred.cpu().numpy();b=y[i]
            errors.append(float(np.abs(a-b).mean()))
            scores.append(float(structural_similarity(a,b,data_range=1)))
    return float(np.mean(errors)),float(np.mean(scores))

def train(args):
    imports();x,y,splits,data_id=inputs()
    model_dir=OUT/'model';model_dir.mkdir(exist_ok=True)
    g,d,opt,pipe,cam=initial_model(args.points)
    xgpu=torch.tensor(x,device='cuda');start=0;best=float('inf')
    last=model_dir/'last.pt'
    if args.resume:
        assert last.exists(),'No last.pt checkpoint to resume.'
        state=restore(last,g,d,opt,data_id);start=state['iteration'];best=state['best_val']
    elif last.exists(): raise RuntimeError('Existing training found. Use --resume or a NEW WRF_RUN_DIR.')
    assert start<args.iterations<=30000,'Total iterations must increase and be <=30000 for this schedule.'
    history=model_dir/'history.csv';write_header=not history.exists()
    with history.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['iteration','train_loss','val_mae','val_ssim','gaussians','elapsed_this_session_s'])
        if write_header:writer.writeheader()
        timer=time.perf_counter()
        for step in range(start+1,args.iterations+1):
            g.update_learning_rate(step);d.update_learning_rate(step)
            i=int(np.random.choice(splits['train']));target=torch.tensor(y[i],device='cuda')
            pred,pkg=predict(g,d,pipe,cam,xgpu[i])
            loss=.8*l1_loss(pred,target)+.2*(1-torch_ssim(pred[None,None],target[None,None]))
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at iteration {step}; inspect last checkpoint.')
            loss.backward()
            visible=pkg['radii']>0
            if not bool(visible.any()):raise RuntimeError('Renderer has no visible Gaussians.')
            with torch.no_grad():
                if step<opt.densify_until_iter:
                    g.max_radii2D[visible]=torch.maximum(g.max_radii2D[visible],pkg['radii'][visible])
                    g.add_densification_stats(pkg['viewspace_points'],visible)
            g.optimizer.step();d.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True);d.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if opt.densify_from_iter<step<opt.densify_until_iter and step%200==0 and len(g.get_xyz)<16000:
                    g.densify_and_prune(opt.densify_grad_threshold,.005,10,None,pkg['radii'])
                    if len(g.get_xyz)>16000:
                        keep=torch.topk(g.get_opacity.flatten(),16000).indices
                        mask=torch.ones(len(g.get_xyz),device='cuda',dtype=torch.bool);mask[keep]=False
                        g.tmp_radii=torch.zeros(len(g.get_xyz),device='cuda');g.prune_points(mask);g.tmp_radii=None
                if len(g.get_xyz)<16: raise RuntimeError('Too few Gaussians remain; inspect optimisation.')
            if step%100==0:print(f'iteration {step}/{args.iterations}; loss={loss.item():.5f}; Gaussians={len(g.get_xyz)}',flush=True)
            if step%500==0 or step==args.iterations:
                mae,score=validation(g,d,pipe,cam,xgpu,y,splits['val'])
                if mae<best:
                    best=mae;save_checkpoint(model_dir/'best.pt',g,d,opt,step,best,data_id)
                save_checkpoint(last,g,d,opt,step,best,data_id)
                writer.writerow(dict(iteration=step,train_loss=loss.item(),val_mae=mae,val_ssim=score,
                    gaussians=len(g.get_xyz),elapsed_this_session_s=time.perf_counter()-timer));f.flush()
                print(f'VALIDATION MAE={mae:.5f}; SSIM={score:.4f}; saved complete checkpoint.',flush=True)
    dump(model_dir/'training_status.json',{'iteration':args.iterations,'best_val_mae':best,
        'status':'TRAINED; fresh checkpoint reload and final test still required'})

def evaluate(args):
    imports();x,y,splits,data_id=inputs();g,d,opt,pipe,cam=initial_model(32)
    checkpoint=OUT/'model'/'best.pt';state=restore(checkpoint,g,d,opt,data_id)
    xgpu=torch.tensor(x,device='cuda');test=splits['test'];train=splits['train']
    from skimage.metrics import structural_similarity
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'evaluation';folder.mkdir(exist_ok=True)
    rows=[];predictions=[];mean_image=y[train].mean(0)
    with torch.no_grad():
        for _ in range(10):predict(g,d,pipe,cam,xgpu[test[0]])
        for i in test:
            torch.cuda.synchronize();t=time.perf_counter()
            pred,_=predict(g,d,pipe,cam,xgpu[i]);torch.cuda.synchronize()
            ms=(time.perf_counter()-t)*1000
            a=pred.cpu().numpy();b=y[i];predictions.append(a)
            nearest=train[np.argmin(np.linalg.norm(x[train]-x[i],axis=1))]
            for method,image in [('wrfgsplus',a),('nearest_train_position',y[nearest]),('mean_train_image',mean_image)]:
                rows.append(dict(sample=int(i),method=method,mae=float(np.abs(image-b).mean()),
                    ssim=float(structural_similarity(image,b,data_range=1)),
                    inference_ms=ms if method=='wrfgsplus' else ''))
    np.save(folder/'predictions.npy',np.stack(predictions));np.save(folder/'test_indices.npy',test)
    with (folder/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for name in ('wrfgsplus','nearest_train_position','mean_train_image'):
        rr=[r for r in rows if r['method']==name]
        summary[name]={k:float(np.mean([r[k] for r in rr])) for k in ('mae','ssim')}
    rr=[r for r in rows if r['method']=='wrfgsplus']
    summary['median_inference_ms']=float(np.median([r['inference_ms'] for r in rr]))
    summary['best_checkpoint_iteration']=state['iteration'];summary['test_count']=len(test)
    summary['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary['data_id']=data_id
    summary['better_mae_than_both_baselines']=all(summary['wrfgsplus']['mae']<summary[k]['mae'] for k in ('nearest_train_position','mean_train_image'))
    summary['status']='COMPLETED: checkpoint reloaded and final test evaluated; inspect accuracy before claiming a useful predictor'
    dump(folder/'summary.json',summary)
    # Predetermined samples avoid selecting only impressive examples.
    fig,axes=plt.subplots(6,3,figsize=(12,12))
    for row,j in enumerate(np.linspace(0,len(test)-1,6,dtype=int)):
        for col,(label,img) in enumerate([('Sionna reference',y[test[j]]),('WRF-GS+ prediction',predictions[j]),('Absolute error',np.abs(y[test[j]]-predictions[j]))]):
            ax=axes[row,col];ax.imshow(img,origin='lower',vmin=0,vmax=1,aspect='auto',extent=(-180,180,-90,90));ax.set_title(f'TX {test[j]}: {label}',fontsize=9)
            ax.set_xlabel('longitude');ax.set_ylabel('latitude')
    fig.tight_layout();fig.savefig(folder/'comparisons.png',dpi=150);plt.close(fig)
    import pandas as pd
    history=pd.read_csv(OUT/'model'/'history.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(history.iteration,history.val_mae);axes[0].set(xlabel='iteration',ylabel='validation MAE')
    axes[1].plot(history.iteration,history.val_ssim);axes[1].set(xlabel='iteration',ylabel='validation SSIM')
    fig.tight_layout();fig.savefig(folder/'validation_curve.png',dpi=150);plt.close(fig)
    report=f'''# One-room WRF-GS+ experiment\n\n{summary['status']}\n\nScene: fullroom, existing 8.15 m by 5 m mesh, Y-up.\nOne fixed 4x4 RX, 600 TX positions at Y=1.2 m; 420 train / 90 val / 90 test.\nCarrier: 3.5 GHz. LOS and up to three specular interactions; no transmission, diffuse scattering or diffraction.\nTarget: per-image peak-normalised linear Bartlett spectrum, 90x360.\n\nBest iteration (selected by validation MAE): {state['iteration']}.\nFinal test MAE: {summary['wrfgsplus']['mae']:.6f}.\nFinal test SSIM: {summary['wrfgsplus']['ssim']:.6f}.\nMedian warmed inference time: {summary['median_inference_ms']:.3f} ms (GPU model/render only).\nNearest-position baseline MAE: {summary['nearest_train_position']['mae']:.6f}.\nMean-image baseline MAE: {summary['mean_train_image']['mae']:.6f}.\nLower MAE than BOTH baselines: {summary['better_mae_than_both_baselines']}.\n\nThis is an adaptation of the published implementation, not a reproduction of its real-world benchmark.\nPredictions were produced in a separate process by reloading both Gaussians and deformation weights.\nSuccess of execution does not by itself establish useful prediction quality.\nNo absolute RSSI, real-world accuracy, cross-room generalisation, or superiority to ray tracing is claimed.\n\nSources: https://github.com/wenchaozheng/WRF-GSplus ; https://arxiv.org/abs/2412.04832 ; https://github.com/NVlabs/sionna-rt/tree/v1.2.1\n'''
    (folder/'mentor_report.md').write_text(report,encoding='utf-8');print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate'])
    p.add_argument('--iterations',type=int,default=10000);p.add_argument('--points',type=int,default=8000)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stage=='train':train(a)
    else:evaluate(a)
