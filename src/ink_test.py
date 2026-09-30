"""Render the same labelled patch of PHerc0139 from the three reconstructions of its 9.362 um scan.

For each PHerc0139 segment with human ink labels on the 9.362 um scan (w035, w039, w040, w041, w044):
  1. take the densest labelled 60 x 170-cell window of the published 9.362 um mesh (11.2 x 31.8 mm);
  2. align pag0 and pag50 to the standard volume: a first offset from 3-D correlation at the window's centre
     vertex (src/align_check.py), then render, register the render to the standard render (mid layer, sub-pixel),
     turn the residual into a volume offset along the render's row / column directions and re-render, until the
     residual is under 0.1 px (the offset varies across the scroll, so one 3-D box is not enough);
  3. render 24 layers (-11.5 .. +11.5 voxels along the smoothed mesh normal; $PAG_LAYERS to change) from each reconstruction
     (src/render_crop.py, level 0, the sampler validated against vc_render_tifxyz);
  4. record which way the layers run (towards the scroll centre or away from it).
Writes $PAG_OUT/<segment>/{standard,pag50,pag0}.npy (uint8, 24 x H x W) and results/renders.json.
usage: python src/ink_test.py results/renders.json <segment> [...]"""
import json, os, subprocess, sys, urllib.request
import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, 'src')
from volumes import REC, centre  # noqa: E402
from align_check import offset  # noqa: E402
from labcmp import get_label  # noqa: E402

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
OUT = os.environ.get('PAG_OUT', 'out')
WIN_H, WIN_W = 60, 170
LAYERS = int(os.environ.get('PAG_LAYERS', '24'))  # 24 = -11.5 .. +11.5 voxels
os.environ.setdefault('XFORM_OUT', OUT)


def mesh(seg, d):
    os.makedirs(d, exist_ok=True)
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f'{B}/PHerc0139/segments/{seg}/mesh/{seg.split("-")[0]}-on-20250728140407-9.362um.tifxyz/{f}', p)
    return np.stack([tifffile.imread(os.path.join(d, c + '.tif')).astype(np.float64) for c in 'xyz'], -1)


def shifted_mesh(d, name, dzyx):
    """Copy of the mesh with (z, y, x) + dzyx on valid points, so the render samples pag where standard has it."""
    o = f'{d}_{name}'
    os.makedirs(o, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(d, 'meta.json'), o)
    valid = tifffile.imread(os.path.join(d, 'z.tif')) > -0.5
    for c, s in zip('zyx', dzyx):
        a = tifffile.imread(os.path.join(d, c + '.tif'))
        tifffile.imwrite(os.path.join(o, c + '.tif'), np.where(valid, a + s, a).astype(a.dtype))
    return o


def residual(a, b, m=3):
    """Sub-pixel (dy, dx) with a[i, j] ~ b[i - dy, j - dx]: high-passed layers, correlation over valid pixels."""
    ok = (a > 0) & (b > 0)
    a = a.astype(np.float32); b = b.astype(np.float32)
    a -= ndimage.gaussian_filter(a, 4); b -= ndimage.gaussian_filter(b, 4)
    g = np.full((2 * m + 1, 2 * m + 1), -2.0)
    for i, dy in enumerate(range(-m, m + 1)):
        for j, dx in enumerate(range(-m, m + 1)):
            bs = np.roll(np.roll(b, dy, 0), dx, 1)
            k = ok & np.roll(np.roll(ok, dy, 0), dx, 1)
            k[:m] = k[-m:] = False; k[:, :m] = k[:, -m:] = False
            g[i, j] = np.corrcoef(a[k], bs[k])[0, 1]
    i, j = np.unravel_index(np.argmax(g), g.shape)
    sub = []
    for t, (u, v, w) in ((i, (g[i - 1, j], g[i, j], g[i + 1, j]) if 0 < i < 2 * m else (0, 0, 0)),
                         (j, (g[i, j - 1], g[i, j], g[i, j + 1]) if 0 < j < 2 * m else (0, 0, 0))):
        den = u - 2 * v + w
        sub.append(t - m + (0.5 * (u - w) / den if den < 0 else 0.0))
    return sub, float(g.max())


def render(md, vol, o, rows, cols):
    subprocess.check_call([sys.executable, 'src/render_crop.py', md, f'{B}/{vol}', o, '--rows', *map(str, rows),
                           '--cols', *map(str, cols), '--layers', str(LAYERS), '--level', '0', '--smooth-normals', '1.0'],
                          stdout=subprocess.DEVNULL)


def main():
    out, segs = sys.argv[1], sys.argv[2:]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for seg in segs:
        if seg in res:
            continue
        d = os.path.join(OUT, seg); os.makedirs(d, exist_ok=True)
        lab = get_label(seg, '9.362um-volume-20250728140407') > 0
        X = mesh(seg, os.path.join(d, 'mesh'))
        Hc, Wc = lab.shape[0] // 20, lab.shape[1] // 20
        cells = lab[:Hc * 20, :Wc * 20].reshape(Hc, 20, Wc, 20).mean((1, 3), dtype=np.float32)
        dens = ndimage.uniform_filter(cells, (WIN_H, WIN_W), mode='constant')
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
        r0 = int(np.clip(cy - WIN_H // 2, 0, max(0, Hc - WIN_H))); c0 = int(np.clip(cx - WIN_W // 2, 0, max(0, Wc - WIN_W)))
        r1, c1 = min(Hc, r0 + WIN_H), min(Wc, c0 + WIN_W)
        g = X[r0:r1, c0:c1]; v = g[..., 0] > -0.5
        P = g[v]
        mid = [int(round(float(np.median(P[:, i])))) for i in (2, 1, 0)]                 # z, y, x
        # layer direction: render_crop's layers ascend along cross(dv, du)
        du = np.gradient(g, axis=1); dv = np.gradient(g, axis=0); n = np.cross(dv, du)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        cyv, cxv = centre(REC['standard'], mid[0])
        to_c = np.stack([cxv - P[:, 0], cyv - P[:, 1], np.zeros(len(P))], 1)
        to_c /= np.linalg.norm(to_c, axis=1, keepdims=True) + 1e-9
        cos_in = float(np.median((n[v] * to_c).sum(1)))
        r = {'window_cells': [r0, r1, c0, c1], 'window_centre_zyx': mid, 'label_frac': round(float(lab[r0 * 20:r1 * 20, c0 * 20:c1 * 20].mean()), 4),
             'layers_ascend_inward': cos_in > 0, 'median_cos_layer_axis_to_centre': round(cos_in, 3), 'offset_vs_standard': {}}
        # the centre vertex (on the sheet) and the render's per-pixel steps there, in volume coordinates (z, y, x):
        # a 2-D residual in the render is turned into a 3-D offset with these local tangents
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        vi = np.argwhere(v); k = np.argmin(((vi - [rc - r0, cc - c0]) ** 2).sum(1))
        on_sheet = [int(round(float(g[tuple(vi[k])][i]))) for i in (2, 1, 0)]
        i0, j0 = vi[k]
        nb = g[max(0, i0 - 3):i0 + 4, max(0, j0 - 3):j0 + 4]; nv = nb[..., 0] > -0.5
        row_step = np.nanmean(np.where((nv[1:] & nv[:-1])[..., None], np.diff(nb, axis=0), np.nan).reshape(-1, 3), 0)[::-1] / 20
        col_step = np.nanmean(np.where((nv[:, 1:] & nv[:, :-1])[..., None], np.diff(nb, axis=1), np.nan).reshape(-1, 3), 0)[::-1] / 20
        r['centre_vertex_zyx'] = on_sheet
        r['render_step_vox'] = {'row_zyx': [round(float(x), 3) for x in row_step], 'col_zyx': [round(float(x), 3) for x in col_step]}
        o_std = os.path.join(d, 'standard')
        if not os.path.exists(o_std + '.npy'):
            render(os.path.join(d, 'mesh'), REC['standard'], o_std, (r0, r1), (c0, c1))
        s0 = np.load(o_std + '.npy', mmap_mode='r')
        c = s0.shape[0] // 2
        for name, vol in REC.items():
            if name == 'standard':
                continue
            o = os.path.join(d, name)
            s3, rp, rz = offset(on_sheet, vol)
            e = {'from_3d_box': {'dz_dy_dx': s3, 'r_peak': rp, 'r_at_zero': rz}, 'refinements': []}
            off = np.array(s3, float)
            if 'pag50' in r['offset_vs_standard']:          # same raw data and processing: start pag0 from pag50's result
                off = np.array(r['offset_vs_standard']['pag50']['final_dz_dy_dx'], float)
            for it in range(3):
                render(shifted_mesh(os.path.join(d, 'mesh'), name, off), vol, o, (r0, r1), (c0, c1))
                (dy, dx), rr = residual(np.asarray(s0[c]), np.asarray(np.load(o + '.npy', mmap_mode='r')[c]))
                e['refinements'].append({'offset_dz_dy_dx': [round(float(x), 3) for x in off], 'residual_px_dy_dx': [round(dy, 3), round(dx, 3)], 'r': round(rr, 3)})
                if abs(dy) < 0.1 and abs(dx) < 0.1:
                    break
                off = off - dy * row_step - dx * col_step
            e['final_dz_dy_dx'] = e['refinements'][-1]['offset_dz_dy_dx']
            r['offset_vs_standard'][name] = e
        res[seg] = r
        json.dump(res, open(out, 'w'), indent=1)
        print(seg.split('-')[1][:4], r, flush=True)


if __name__ == '__main__':
    main()
