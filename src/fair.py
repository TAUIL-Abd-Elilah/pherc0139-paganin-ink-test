"""Fair comparisons of the three reconstructions for ink (no model trained on one of them is involved):
  - clipping: fraction of rendered voxels at 0 or 255 (the uint8 export windows differ between the volumes);
  - raw ink contrast: Cohen's d between labelled ink and unlabelled papyrus within 2 mm of it, per layer, after
    in-plane Gaussian smoothing sigma = 0 / 1 / 2 / 4 / 8 px (the noise vs sharpness trade-off);
  - how the three depth profiles compare: least-squares scale k (profile = k x standard profile), shape
    correlation r, and the contrast (L2 norm of d over layers) near the traced surface (|offset| <= 4 voxels,
    where the ink is) and far from it (|offset| >= 7 voxels) as a ratio to the standard volume.
A model trained on each reconstruction is in src/fair_cnn.py.
usage: python src/fair.py results/renders.json results/fair.json"""
import json, os, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, 'src')
from models import OUT, RECS, masks  # noqa: E402


def main():
    rp, out = sys.argv[1], sys.argv[2]
    res = json.load(open(rp))
    segs = sorted(res)
    rep = {'segments': {}}
    for seg in segs:
        d = os.path.join(OUT, seg)
        st = {n: np.load(os.path.join(d, n + '.npy')).astype(np.float32) for n in RECS}
        valid = ndimage.binary_erosion(st['standard'][st['standard'].shape[0] // 2] > 0, iterations=32)
        for n in RECS[1:]:
            valid &= st[n][st[n].shape[0] // 2] > 0
        L = masks(seg, res[seg], st['standard'].shape)
        pos = L & valid
        near = (ndimage.distance_transform_edt(~L) <= 2.0 / 9.362e-3) & ~L & valid
        e = {'n_ink_px': int(pos.sum()), 'n_papyrus_px': int(near.sum())}
        for n in RECS:
            s = st[n]
            v3 = np.broadcast_to(valid, s.shape)
            q = {'clip_low_frac': round(float((s[v3] == 0).mean()), 4), 'clip_high_frac': round(float((s[v3] == 255).mean()), 4)}
            for sig in (0, 1, 2, 4, 8):
                dk = []
                for k in range(s.shape[0]):
                    a = ndimage.gaussian_filter(s[k], sig) if sig else s[k]
                    x, y = a[pos], a[near]
                    dk.append(float((x.mean() - y.mean()) / np.sqrt(0.5 * (x.var() + y.var()) + 1e-9)))
                kb = int(np.argmax(np.abs(dk)))
                q[f'd_sigma{sig}'] = {'best_layer': kb, 'd': round(dk[kb], 3), 'by_layer': [round(x, 3) for x in dk]}
            e[n] = q
        for n in RECS[1:]:
            e[n]['vs_standard'] = {}
            for sig in (0, 1, 2, 4, 8):
                a0 = np.array(e['standard'][f'd_sigma{sig}']['by_layer']); b0 = np.array(e[n][f'd_sigma{sig}']['by_layer'])
                off = np.arange(len(a0)) - (len(a0) - 1) / 2
                core, outer = np.abs(off) <= 4, np.abs(off) >= 7
                e[n]['vs_standard'][f'sigma{sig}'] = {'scale_k': round(float((a0 * b0).sum() / (a0 * a0).sum()), 3),
                                                     'shape_r': round(float(np.corrcoef(a0, b0)[0, 1]), 3),
                                                     'l2_ratio': round(float(np.linalg.norm(b0) / np.linalg.norm(a0)), 3),
                                                     'core_l2_ratio': round(float(np.linalg.norm(b0[core]) / np.linalg.norm(a0[core])), 3),
                                                     'outer_l2_ratio': round(float(np.linalg.norm(b0[outer]) / np.linalg.norm(a0[outer])), 3)}
        rep['segments'][seg] = e
        print(seg.split('-')[1][:4], {n: (e[n]['d_sigma0']['d'], e[n]['d_sigma2']['d'], e[n]['clip_low_frac']) for n in RECS}, flush=True)
    rep['summary'] = {}
    for n in RECS[1:]:
        for sig in (0, 2, 4, 8):
            ks = [rep['segments'][g][n]['vs_standard'][f'sigma{sig}']['scale_k'] for g in segs]
            rs = [rep['segments'][g][n]['vs_standard'][f'sigma{sig}']['shape_r'] for g in segs]
            cr = [rep['segments'][g][n]['vs_standard'][f'sigma{sig}']['core_l2_ratio'] for g in segs]
            orr = [rep['segments'][g][n]['vs_standard'][f'sigma{sig}']['outer_l2_ratio'] for g in segs]
            rep['summary'][f'{n}_sigma{sig}'] = {'scale_k_mean': round(float(np.mean(ks)), 3), 'scale_k_range': [min(ks), max(ks)],
                                                'shape_r_mean': round(float(np.mean(rs)), 3), 'shape_r_min': min(rs),
                                                'core_ratio_median': round(float(np.median(cr)), 3), 'core_ratio_range': [min(cr), max(cr)],
                                                'core_segments_at_or_above_standard': int(sum(c >= 1.0 for c in cr)),
                                                'outer_ratio_median': round(float(np.median(orr)), 3), 'outer_ratio_range': [min(orr), max(orr)]}
    print(json.dumps(rep['summary'], indent=1))
    json.dump(rep, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
