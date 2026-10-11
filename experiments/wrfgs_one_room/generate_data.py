"""Prepare one existing room and generate a reproducible Sionna dataset."""
import argparse, hashlib, json, os, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from rf_math import *

ROOT=Path(__file__).resolve().parent
OUT=Path(os.environ.get('WRF_RUN_DIR',str(ROOT/'run_fullroom')))

def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); temp.replace(path)

def configuration():
    return dict(scene='fullroom', seed=2026, frequency_hz=3.5e9,
        n_samples=600, rx_world=[.85,1.2,-4.2], tx_height_y=1.2,
        array='4x4, half-wavelength, yz plane, zero orientation, iso/V',
        coordinate_system='original exported coordinates; Y-up; no geometry rotation',
        model_coordinates='world minus receiver position, in metres',
        target='per-image peak-normalised linear Bartlett power at carrier',
        angular_grid='pinned CUDA mapping: atan2(x,z), asin(y/r); 90x360 full sphere',
        max_depth=3, samples_per_src=200000, max_num_paths_per_src=200000,
        los=True,specular_reflection=True,diffuse_reflection=False,refraction=False,
        diffraction=False, synthetic_array=True,
        scope='single-scene interpolation on one TX-height plane; simulated targets only')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=configuration()
    meshdir=ROOT/'scene_fullroom'/'meshes'
    bounds=mesh_bounds(meshdir)
    cfg['scene_sha256']={str(p.relative_to(ROOT/'scene_fullroom')):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scene_fullroom').rglob('*')) if p.is_file()}
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if (OUT/'config.json').exists():
        if json.loads((OUT/'config.json').read_text()) != cfg:
            raise RuntimeError('Settings changed. Set WRF_RUN_DIR to a new experiment folder; do not mix datasets.')
    write_json(OUT/'config.json',cfg)
    (OUT/'dataset_id.txt').write_text(digest)
    rx,xyz,splits=sample_positions(bounds,cfg['n_samples'],cfg['seed'])
    np.save(OUT/'tx_world.npy',xyz); np.save(OUT/'tx_model.npy',xyz-rx)
    for split,indices in splits.items(): np.save(OUT/(split+'_indices.npy'),indices)
    write_json(OUT/'geometry_bounds.json',bounds)
    materials={}
    xml=ET.parse(ROOT/'scene_fullroom'/'fullroom_materials.xml').getroot()
    for shape in xml.findall('shape'):
        name=Path(shape.find("string[@name='filename']").get('value')).stem
        materials[name]=shape.find('ref').get('id')
    write_json(OUT/'materials.json',materials)
    # Validate the array itself, not only its centre, against object bounds.
    wavelength=299792458/cfg['frequency_hz']
    for y in np.arange(4)*.5-.75:
        for z in np.arange(4)*.5-.75:
            assert clear_position(rx+np.array([0,y,z])*wavelength,bounds,.10)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(11,7))
    for name,(lo,hi) in bounds.items():
        if name in ('floor','walls'): continue
        ax.add_patch(Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],alpha=.15,color='gray'))
        ax.text((lo[0]+hi[0])/2,(lo[2]+hi[2])/2,name,fontsize=7,ha='center')
    for split,color in [('train','#2458c6'),('val','#b58000'),('test','#c23752')]:
        p=xyz[splits[split]]; ax.scatter(p[:,0],p[:,2],s=9,c=color,label=f'{split}: {len(p)}')
    ax.scatter(rx[0],rx[2],marker='*',s=220,c='black',label='fixed RX')
    ax.set(xlim=(0,8.15),ylim=(-5,0),xlabel='Exported X (m)',ylabel='Exported Z (m)',
           title='fullroom: all TX/RX centres at Y=1.2 m; shaded object footprints')
    ax.set_aspect('equal'); ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    fig.tight_layout(); fig.savefig(OUT/'room_plan.png',dpi=160); plt.close(fig)
    print('Prepared:',OUT,'\n420 training / 90 validation / 90 final-test positions.')
    return cfg

def configured_scene(empty=False):
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
    cfg=json.loads((OUT/'config.json').read_text())
    scene=load_scene(None if empty else str(ROOT/'scene_fullroom'/'fullroom_materials.xml'),merge_shapes=False)
    scene.frequency=cfg['frequency_hz']
    scene.tx_array=PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='V')
    scene.rx_array=PlanarArray(num_rows=4,num_cols=4,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    scene.add(Transmitter(name='tx',position=[2,1.2,-4.0],orientation=[0,0,0]))
    scene.add(Receiver(name='rx',position=cfg['rx_world'],orientation=[0,0,0]))
    return scene,cfg

def solve(scene,cfg,index,samples=None,empty=False):
    from sionna.rt import PathSolver
    solver=PathSolver()
    start=time.perf_counter()
    paths=solver(scene,max_depth=0 if empty else cfg['max_depth'],
        samples_per_src=samples or cfg['samples_per_src'],
        max_num_paths_per_src=cfg['max_num_paths_per_src'],
        synthetic_array=True,los=True,specular_reflection=not empty,
        diffuse_reflection=False,refraction=False,diffraction=False,seed=cfg['seed']+int(index))
    # cir() adds the carrier-delay phase exactly once. paths.a alone is passband.
    a,tau=paths.cir(normalize_delays=False,out_type='numpy')
    if isinstance(a,(tuple,list)): a=np.asarray(a[0])+1j*np.asarray(a[1])
    a=np.asarray(a)
    assert a.ndim==6 and a.shape[:4]==(1,16,1,1), f'Unexpected CIR shape {a.shape}'
    a=a[0,:,0,0,:,0]
    if a.shape[-1]==0: raise RuntimeError(f'No paths for sample {index}; inspect placement/solver.')
    channel=a.sum(axis=-1)
    offsets=np.asarray(scene.rx_array.positions(float(np.asarray(scene.wavelength).item()))).T
    assert offsets.shape==(16,3),offsets.shape
    steering=steering_matrix(offsets,float(np.asarray(scene.wavelength).item()))
    power,peak=spectrum_from_channel(channel,steering)
    duration=time.perf_counter()-start
    return dict(image=power,peak=peak,channel=channel,a_baseband=a,tau=np.asarray(tau),
                theta_r=np.asarray(paths.theta_r),phi_r=np.asarray(paths.phi_r),
                seconds=duration,offsets=offsets)

def validate_los():
    scene,cfg=configured_scene(empty=True)
    rx=np.asarray(cfg['rx_world']); tx=rx+np.array([1.8,.4,-1.0])
    scene.get('tx').position=tx.tolist()
    item=solve(scene,cfg,0,empty=True)
    direction=(tx-rx)/np.linalg.norm(tx-rx)
    # Compare full spectrum against analytic single-path array response.
    wavelength=cfg['frequency_hz']; wavelength=299792458/wavelength
    a=steering_matrix(item['offsets'],wavelength)
    analytic_h=np.exp(2j*np.pi/wavelength*(item['offsets']@direction))
    expected,_=spectrum_from_channel(analytic_h,a)
    mae=float(np.mean(np.abs(expected-item['image'])))
    if mae>.01: raise RuntimeError(f'LOS steering/coordinate check failed: MAE={mae}. Do not generate full data.')
    write_json(OUT/'los_check.json',{'passed':True,'analytic_spectrum_mae':mae})
    print('PASS: free-space direction, steering sign and array ordering; MAE=',mae)

def generate(limit=None):
    import mitsuba as mi
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    cfg=prepare(); scene,_=configured_scene()
    # Ensure named ITU assignments have actually loaded with non-vacuum properties.
    material_info={name:{'relative_permittivity':float(np.asarray(m.relative_permittivity).item()),
                        'conductivity':float(np.asarray(m.conductivity).item())}
                   for name,m in scene.radio_materials.items()}
    write_json(OUT/'loaded_materials.json',material_info)
    assert len(scene.objects)==10, f'Expected 10 shapes; got {len(scene.objects)}'
    assert any(v['conductivity']>1e6 for v in material_info.values()), 'Metal did not load'
    if not (OUT/'los_check.json').exists(): validate_los()
    xyz=np.load(OUT/'tx_world.npy'); n=len(xyz) if limit is None else min(limit,len(xyz))
    rawdir=OUT/'raw'; rawdir.mkdir(exist_ok=True)
    for i in range(n):
        dest=rawdir/f'{i:04d}.npz'
        if dest.exists(): continue
        scene.get('tx').position=xyz[i].tolist()
        item=solve(scene,cfg,i)
        tmp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**item); tmp.replace(dest)
        if i%10==0 or i+1==n:
            print(f'RF example {i+1}/{n}; paths={item["a_baseband"].shape[-1]}; seconds={item["seconds"]:.2f}',flush=True)
    # Compare a second, higher-ray solve of an existing example before full training.
    # This is a basic numerical convergence check, not a full simulator validation.
    if not (OUT/'convergence.json').exists():
        i=int(np.argmax(xyz[:n,0])); scene.get('tx').position=xyz[i].tolist()
        reference=np.load(rawdir/f'{i:04d}.npz')['image']
        high=solve(scene,cfg,i,samples=cfg['samples_per_src']*2)
        error=float(np.mean(np.abs(reference-high['image'])))
        write_json(OUT/'convergence.json',{'sample':i,'mae':error,'passed':error<=.08,
            'description':'same seed; doubled launch rays; normalised image MAE'})
        if error>.08: raise RuntimeError('Pilot did not converge. Increase launch rays in configuration and use a NEW WRF_RUN_DIR.')
    assert json.loads((OUT/'convergence.json').read_text())['passed']
    fig,axes=plt.subplots(2,4,figsize=(14,5))
    for ax,i in zip(axes.flat,range(min(n,8))):
        item=np.load(rawdir/f'{i:04d}.npz'); ax.imshow(item['image'],vmin=0,vmax=1,origin='lower',aspect='auto',extent=(-180,180,-90,90))
        ax.set_title(f'TX {i}; {item["a_baseband"].shape[-1]} paths'); ax.set_xlabel('longitude (deg)'); ax.set_ylabel('latitude (deg)')
    fig.tight_layout(); fig.savefig(OUT/'pilot_rf_images.png',dpi=140); plt.close(fig)
    if n<len(xyz):
        print('Pilot complete. Inspect room_plan.png and pilot_rf_images.png before the full dataset.'); return
    images=np.stack([np.load(rawdir/f'{i:04d}.npz')['image'] for i in range(n)])
    assert images.shape==(600,90,360) and np.isfinite(images).all()
    variation=float(np.std(images,axis=0).mean())
    assert variation>.005, 'RF targets hardly vary; inspect coordinate/data-generation setup.'
    np.save(OUT/'spectra.npy',images)
    # Export stock-compatible PNG/CSV as well as float32 arrays used by our adapter.
    stock=OUT/'stock_format'; (stock/'spectrum').mkdir(parents=True,exist_ok=True)
    for i,img in enumerate(images): Image.fromarray(np.rint(img*255).astype('uint8')).save(stock/'spectrum'/f'{i+1}.png')
    np.savetxt(stock/'tx_pos.csv',np.load(OUT/'tx_model.npy'),delimiter=',',header='x,y,z',comments='')
    for split in ('train','val','test'):
        np.savetxt(stock/(split+'_index.txt'),np.load(OUT/(split+'_indices.npy'))+1,fmt='%d')
    (stock/'gateway_info.yml').write_text('dataset_name: fullroom\ngateway1:\n  position: [0, 0, 0]\n  orientation: [0, 0, 0, 1]\n')
    write_json(OUT/'data_ready.json',{'count':n,'mean_pixel_std_across_samples':variation,
        'dataset_id':(OUT/'dataset_id.txt').read_text(),
        'spectra_sha256':hashlib.sha256((OUT/'spectra.npy').read_bytes()).hexdigest(),
        'status':'DATA GENERATED; training/evaluation still required'})
    print('All 600 RF images saved. Data generation complete; this is NOT a trained model.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','all']);args=p.parse_args()
    if args.stage=='prepare': prepare()
    else: generate(8 if args.stage=='pilot' else None)
