"""Fetch the original WRF-GS+ implementation; keep upstream licenses intact."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/wrfgs_one_room/vendor/WRF-GSplus'
COMMIT='e564f3184463717ae08660a8b7232c6318c73ca6'
if __name__=='__main__':
    if DEST.exists():raise SystemExit('Destination already exists; inspect it instead of overwriting.')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','https://github.com/wenchaozheng/WRF-GSplus',str(DEST)],check=True)
    subprocess.run(['git','-C',str(DEST),'checkout',COMMIT],check=True)
    subprocess.run(['git','-C',str(DEST),'submodule','update','--init','--recursive'],check=True)
    print('Fetched pinned upstream sources. Read their licenses before use.')
