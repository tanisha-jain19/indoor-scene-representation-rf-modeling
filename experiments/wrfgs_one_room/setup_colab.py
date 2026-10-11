"""Run with Colab's default Python; builds an isolated GPU environment."""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ENV=Path('/content/wrfgs_env')
PY=str(ENV/'bin/python')

def run(cmd,**kw):
    print('Running:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True,**kw)

assert Path('/content').exists(),'This setup script is intended for Google Colab Linux.'
run(['nvidia-smi'])
nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
version=subprocess.check_output([nvcc,'--version'],text=True)
print(version)
match=re.search(r'release (\d+)\.(\d+)',version)
assert match and int(match[1])==12,('This prepared environment requires a CUDA 12.x compiler. '
    'This runtime has a different compiler; share this output before continuing.')
if not Path('/usr/bin/g++-12').exists():
    run(['apt-get','update','-qq'])
    run(['apt-get','install','-y','-qq','gcc-12','g++-12'])
run([sys.executable,'-m','pip','install','-q','uv==0.6.17'])
run([sys.executable,'-m','uv','python','install','3.11'])
if not (ENV/'bin/python').exists():
    run([sys.executable,'-m','uv','venv','--python','3.11','--seed',ENV])
run([PY,'-m','pip','install','pip==25.0.1','setuptools==75.8.2','wheel==0.45.1','ninja==1.11.1.3'])
run([PY,'-m','pip','install','torch==2.6.0','torchvision==0.21.0','--index-url','https://download.pytorch.org/whl/cu124'])
run([PY,'-m','pip','install','numpy==1.26.4','scipy==1.15.3','matplotlib==3.10.1',
    'pillow==11.1.0','sionna-rt==1.2.1','plyfile==1.1','pandas==2.2.3',
    'imageio==2.37.0','scikit-image==0.25.2','PyYAML==6.0.2','einops==0.8.1',
    'opencv-python-headless==4.11.0.86','tqdm==4.67.1','tensorboard==2.19.0'])
cap=subprocess.check_output([PY,'-c','import torch; assert torch.cuda.is_available(); print("%s.%s" % torch.cuda.get_device_capability())'],text=True).strip()
assert cap in ('7.5','8.0','8.6','8.9','9.0'), 'Select T4 (or a supported Ampere/Hopper GPU) for this pinned environment.'
env=os.environ.copy();env['MAX_JOBS']='2';env['TORCH_CUDA_ARCH_LIST']=cap
env['PATH']=str(ENV/'bin')+os.pathsep+env['PATH']
env['CC']='/usr/bin/gcc-12'
env['CXX']='/usr/bin/g++-12'
env['CUDA_HOME']=str(Path(nvcc).resolve().parent.parent)
for module in ['simple-knn','diff-gaussian-rasterization']:
    run([PY,'-m','pip','install','-v','--no-build-isolation',ROOT/'vendor/WRF-GSplus/submodules'/module],env=env)
run([PY,'-m','pip','check'])
print('SETUP COMPLETE. All experiment commands must use:',PY)
