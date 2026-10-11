"""GPU smoke test using the authors' supplied data and unchanged model/renderer."""
import json, os, subprocess
from pathlib import Path
import numpy as np
import train_room as tr

tr.imports()
import torch
from scene.dataloader import Spectrum_dataset
data=tr.VENDOR/'data_test200'
index=data/'train_index.txt'
dataset=Spectrum_dataset(str(data),str(index))
target,tx=dataset[0]
assert target.shape==(90,360),target.shape
g,d,opt,pipe,cam=tr.initial_model(1024)
# Use the supplied receiver pose for this example only.
import yaml
from scipy.spatial.transform import Rotation
info=yaml.safe_load((data/'gateway_info.yml').read_text())['gateway1']
cam=tr.generate_new_cam(torch.tensor(Rotation.from_quat(info['orientation']).as_matrix(),dtype=torch.float32),info['position'])
losses=[]
for _ in range(3):
    pred,_=tr.predict(g,d,pipe,cam,tx.cuda())
    loss=.8*tr.l1_loss(pred,target.cuda())+.2*(1-tr.torch_ssim(pred[None,None],target.cuda()[None,None]))
    assert torch.isfinite(loss)
    loss.backward()
    assert g._features_dc.grad is not None and bool(g._features_dc.grad.abs().sum()>0),'No Gaussian gradient'
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in d.deform.parameters()),'No deformation gradient'
    losses.append(float(loss));g.optimizer.step();d.optimizer.step()
    g.optimizer.zero_grad(set_to_none=True);d.optimizer.zero_grad(set_to_none=True)
tr.OUT.mkdir(parents=True,exist_ok=True)
record={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'smoke_losses':losses,'status':'PASS: authors sample, CUDA forward/backward and both optimisers; not a trained benchmark'}
(tr.OUT/'runtime_check.json').write_text(json.dumps(record,indent=2))
with (tr.OUT/'pip_freeze.txt').open('w') as f:subprocess.run([os.sys.executable,'-m','pip','freeze'],stdout=f,check=True)
print(json.dumps(record,indent=2))
