"""Numerical conventions shared by generation, testing and documentation."""
import numpy as np

HEIGHT, WIDTH = 90, 360

def direction_grid():
    # Exact inverse of the pinned WRF-GSplus CUDA point3ToLonlatScreen + ndc2Pix.
    # World/model axes stay Y-up, receiver camera rotation is identity.
    lon = ((np.arange(WIDTH) + .5) / WIDTH * 2 - 1) * np.pi
    lat = ((np.arange(HEIGHT) + .5) / HEIGHT * 2 - 1) * np.pi / 2
    lo, la = np.meshgrid(lon, lat)
    return np.stack([np.cos(la)*np.sin(lo), np.sin(la),
                     np.cos(la)*np.cos(lo)], axis=-1).reshape(-1, 3)

def direction_pixel(direction):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    return np.stack([(lon/np.pi+1)*WIDTH/2-.5,
                     (lat*2/np.pi+1)*HEIGHT/2-.5], axis=-1)

def steering_matrix(offsets_m, wavelength):
    """Unit-modulus array steering, positive phase as in Sionna synthetic RX."""
    return np.exp(2j*np.pi/wavelength * (direction_grid() @ np.asarray(offsets_m).T)).astype(np.complex64)

def spectrum_from_channel(channel, steering):
    """Narrowband Bartlett power: |a(direction)^H h / M|^2.

    channel contains the COHERENT sum of BASEBAND paths for each RX antenna.
    Normalisation discards absolute power. It is part of the target definition,
    applied independently to each example, not a fitted dataset statistic.
    """
    h = np.asarray(channel).reshape(-1)
    power = np.abs(steering.conj() @ h / len(h))**2
    peak = float(power.max())
    if not np.isfinite(power).all() or peak <= 1e-30:
        raise ValueError('Empty/nonfinite RF spectrum; inspect scene and positions.')
    return (power/peak).reshape(HEIGHT, WIDTH).astype(np.float32), peak

def read_ply_vertices(path):
    """Read the bundled binary little-endian PLY vertex table, without extra deps."""
    types = {'float':'<f4', 'double':'<f8', 'uchar':'u1', 'int':'<i4'}
    with open(path, 'rb') as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            line = line.decode('ascii').strip()
            lines.append(line)
            if line == 'end_header': break
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Expected bundled binary little-endian PLY')
        k = next(i for i,s in enumerate(lines) if s.startswith('element vertex '))
        n = int(lines[k].split()[-1]); props=[]
        for s in lines[k+1:]:
            if s.startswith('element ') or s == 'end_header': break
            if s.startswith('property '):
                _,typ,name = s.split(); props.append((name, types[typ]))
        a = np.fromfile(f, dtype=props, count=n)
        if len(a) != n: raise ValueError('Truncated vertex data')
    return np.column_stack([a[k] for k in ('x','y','z')])

def mesh_bounds(root):
    return {p.stem: [v.min(0).tolist(), v.max(0).tolist()]
            for p in sorted(root.glob('*.ply')) for v in [read_ply_vertices(p)]}

def clear_position(p, bounds, margin=.20):
    p=np.asarray(p)
    if not (.35 < p[0] < 7.8 and -4.65 < p[2] < -.35 and .3 < p[1] < 2.7):
        return False
    for name,(lo,hi) in bounds.items():
        if name in ('floor','walls'): continue
        if np.all(p >= np.asarray(lo)-margin) and np.all(p <= np.asarray(hi)+margin):
            return False
    return True

def sample_positions(bounds, n=600, seed=2026):
    rx=np.array([.85,1.2,-4.2], dtype=float)
    assert clear_position(rx,bounds), 'Receiver intersects expanded object bounds'
    rng=np.random.default_rng(seed); points=[]
    for _ in range(200000):
        p=np.array([rng.uniform(.4,7.75),1.2,rng.uniform(-4.6,-.4)])
        if not clear_position(p,bounds) or np.linalg.norm(p-rx)<.7: continue
        if points and np.min(np.linalg.norm(np.asarray(points)-p,axis=1)) < .15: continue
        points.append(p)
        if len(points)==n: break
    if len(points)!=n: raise RuntimeError('Could not place all transmitters safely')
    xyz=np.asarray(points,dtype=np.float32)
    order=rng.permutation(n)
    return rx,xyz, {'train':order[:int(.7*n)],
                    'val':order[int(.7*n):int(.85*n)],
                    'test':order[int(.85*n):]}
