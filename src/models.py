"""Released ink models on the three reconstructions of the same labelled patches.

All of these models were trained on delta/beta = 1000 data, and ink_9um was trained on native 9.362 um renders of
exactly these five segments (its checkpoint config lists w035, w039, w040, w041, w044). So this is not a fair
contest between the reconstructions: it asks whether a model trained on the standard reconstruction still reads
the others. src/fair.py has the fair comparisons.

Models (paths from the environment):
  ink_9um s42 / s43  $INK9UM_S42, $INK9UM_S43   organisers' release, hybrid 3D-2D, 17 centre layers
  Reader v2          $READER_V2                  github.com/DomRusso2/reader-v2 (MIT)
  (both through villa's ink inference CLI: $VILLA_SRC = <villa>/vesuvius/src, both layer orders)
  v8-in              $V8IN_DIR                   huggingface.co/YoussefMoNader/ink-8um-v8in (MIT), outward -> inward
Layer order: for the villa models, chosen once per segment on the standard reconstruction (the better of the two)
and kept for all three; for v8-in, fixed by geometry (layers ordered outward -> inward).
Labels: the published 9.362 um labels on the published mesh's canvas, no shift. AUC = labelled ink vs unlabelled
papyrus within 2 mm of it, inside the render (32 px from its edge).
usage: python src/models.py results/renders.json results/models.json"""
import json, os, subprocess, sys
import numpy as np
import tifffile, zarr
from scipy import ndimage

sys.path.insert(0, 'src')
from labcmp import get_label  # noqa: E402

OUT = os.environ.get('PAG_OUT', 'out')
os.environ.setdefault('XFORM_OUT', OUT)
RECS = ('standard', 'pag50', 'pag0')
VILLA = {'ink9um_s42': os.environ.get('INK9UM_S42'), 'ink9um_s43': os.environ.get('INK9UM_S43'),
         'reader_v2': os.environ.get('READER_V2')}


def auc(score, pos, neg):
    sp, sn = score[pos], score[neg]
    if sp.size < 50 or sn.size < 50:
        return float('nan')
    rng = np.random.default_rng(0)
    sp = rng.choice(sp, min(sp.size, 200000), replace=False); sn = rng.choice(sn, min(sn.size, 200000), replace=False)
    rk = np.concatenate([sp, sn]).argsort(kind='mergesort').argsort() + 1
    return float((rk[:sp.size].sum() - sp.size * (sp.size + 1) / 2) / (sp.size * sn.size))


def masks(seg, r, shape):
    H, W = shape[1:]
    r0, r1, c0, c1 = r['window_cells']
    L = get_label(seg, '9.362um-volume-20250728140407')[r0 * 20:r1 * 20, c0 * 20:c1 * 20][:H, :W] > 0
    return L


def villa_maps(npy, name, ck):
    zp = npy.replace('.npy', '.zarr')
    if not os.path.exists(os.path.join(zp, '.zarray')) and not os.path.exists(os.path.join(zp, 'zarr.json')):
        s = np.load(npy)
        z = zarr.open_array(zp, mode='w', shape=s.shape, chunks=(s.shape[0], 256, 256), dtype='u1')
        z[:] = s
    out = npy.replace('.npy', f'_{name}.tif')
    if not os.path.exists(out.replace('.tif', '_reverse.tif')):
        env = dict(os.environ, PYTHONPATH=os.environ['VILLA_SRC'])
        subprocess.check_call([sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', zp, ck, out, '--overlap', '0.5',
                               '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'both', '--no-compile'],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    maps = []
    for p in (out, out.replace('.tif', '_reverse.tif')):
        a = tifffile.imread(p).astype(np.float32)
        maps.append(a / 255 if a.max() > 1.5 else a)
    return maps


def v8in_map(npy, inward_plus, model):
    out = npy.replace('.npy', '_v8in.npy')
    if not os.path.exists(out):
        from ink8um import predict_stack
        s = np.load(npy)
        st = np.ascontiguousarray(np.clip(s, 0, 200).astype(np.uint8).transpose(1, 2, 0))
        np.save(out, predict_stack(model, st, reverse=not inward_plus, stride=21, batch_size=8, num_workers=0).astype(np.float32))
    return np.load(out)


def main():
    rp, out = sys.argv[1], sys.argv[2]
    res = json.load(open(rp))
    rep = json.load(open(out)) if os.path.exists(out) else {}
    v8 = None
    for seg, r in sorted(res.items()):
        d = os.path.join(OUT, seg)
        s0 = np.load(os.path.join(d, 'standard.npy'), mmap_mode='r')
        valid = ndimage.binary_erosion(np.asarray(s0[s0.shape[0] // 2]) > 0, iterations=32)
        for n in RECS[1:]:
            valid &= np.asarray(np.load(os.path.join(d, n + '.npy'), mmap_mode='r')[s0.shape[0] // 2]) > 0
        L = masks(seg, r, s0.shape)
        pos = L & valid
        near = (ndimage.distance_transform_edt(~L) <= 2.0 / 9.362e-3) & ~L & valid
        e = rep.setdefault(seg, {'n_ink_px': int(pos.sum()), 'n_papyrus_px': int(near.sum())})
        for m, ck in VILLA.items():
            if m in e or not ck:
                continue
            maps = {n: villa_maps(os.path.join(d, n + '.npy'), m, ck) for n in RECS}
            H, W = s0.shape[1:]
            a_std = [auc(mp[:H, :W], pos, near) for mp in maps['standard']]
            k = int(np.argmax(a_std))
            e[m] = {'layer_order': ['as_rendered', 'reversed'][k], 'standard_auc_other_order': round(a_std[1 - k], 4)}
            for n in RECS:
                e[m][n] = round(auc(maps[n][k][:H, :W], pos, near), 4)
            json.dump(rep, open(out, 'w'), indent=1)
            print(seg.split('-')[1][:4], m, e[m], flush=True)
        if 'v8in' not in e and os.environ.get('V8IN_DIR'):
            if v8 is None:
                sys.path.insert(0, os.environ['V8IN_DIR'])
                from ink8um import InkDetector
                v8 = InkDetector.from_pretrained(os.environ['V8IN_DIR']).cuda().eval()
            e['v8in'] = {n: round(auc(v8in_map(os.path.join(d, n + '.npy'), r['layers_ascend_inward'], v8), pos, near), 4) for n in RECS}
            json.dump(rep, open(out, 'w'), indent=1)
            print(seg.split('-')[1][:4], 'v8in', e['v8in'], flush=True)


if __name__ == '__main__':
    main()
