"""Replay the original distance-matrix and Fourier methods on supplied CSVs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

def compute(folder):
    frames={p.stem:pd.read_csv(p) for p in sorted(Path(folder).glob('*.csv'))}
    if not frames: raise ValueError('No room CSVs found')
    for name,df in frames.items():
        if not {'Object','X','Y','Z'}<=set(df):raise ValueError(f'{name}: expected Object,X,Y,Z')
        if df.Object.duplicated().any():raise ValueError(f'{name}: duplicate object identifiers')
        if not np.isfinite(df[['X','Y','Z']].to_numpy(dtype=float)).all():raise ValueError(f'{name}: invalid coordinates')
    objects=sorted(set().union(*(set(df.Object) for df in frames.values())))
    names=list(frames);distance=[];fourier=[]
    for df in frames.values():
        coords=df.set_index('Object')[['X','Y','Z']].reindex(objects).fillna(0).to_numpy(dtype=np.float32)
        distance.append(cdist(coords,coords).flatten().astype(np.float32))
        # Preserve original order: coordinate, frequency, sine/cosine.
        encoded=[]
        for value in coords.flatten():
            for k in range(6):encoded.extend([np.sin((2.0**k)*np.pi*value),np.cos((2.0**k)*np.pi*value)])
        vec=np.array(encoded,dtype=np.float32);fourier.append(vec/np.linalg.norm(vec))
    return names,objects,{'distance':cosine_similarity(distance),'fourier':cosine_similarity(fourier)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/fourier/ExtractedCSV')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results/geometry_replay')
    args=parser.parse_args();names,objects,matrices=compute(args.data);args.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    for method,matrix in matrices.items():
        frame=pd.DataFrame(matrix,index=names,columns=names);frame.to_csv(args.out/f'{method}_similarity.csv')
        fig,ax=plt.subplots(figsize=(11,9));sns.heatmap(frame,annot=True,fmt='.3f',cmap='viridis',ax=ax)
        ax.set_title(f'{method.title()} room similarity — cosine');fig.tight_layout();fig.savefig(args.out/f'{method}_heatmap.png',dpi=150);plt.close(fig)
    print(f'Replayed {len(names)} rooms with {len(objects)} aligned objects; outputs: {args.out}')

if __name__=='__main__':main()
