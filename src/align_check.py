"""Are the three reconstructions of PHerc0139's 9.362 um scan voxel-aligned?

At a point (z, y, x), cut a 64 x 192 x 192 box from the standard volume (delta/beta 1000), high-pass it (minus a
sigma-4 Gaussian), and correlate it with the same box of the pag0 / pag50 volume shifted by integers
dz in [-3, 3], dy, dx in [-2, 2]. The peak is refined to sub-voxel precision per axis with a parabola through the
peak and its two neighbours. Result: pag[z + dz, y + dy, x + dx] matches standard[z, y, x].
usage: python src/align_check.py results/alignment.json [z y x ...]"""
import json, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, 'src')
import zfetch as zf  # noqa: E402
from volumes import REC  # noqa: E402

MZ, MYX, BZ, BYX = 3, 2, 32, 96


def _box(vol, c, mz, myx):
    zc, yc, xc = c
    b = zf.read_box(vol, 0, (zc - BZ - mz, yc - BYX - myx, xc - BYX - myx), (zc + BZ + mz, yc + BYX + myx, xc + BYX + myx))
    b = b.astype(np.float32)
    return b - ndimage.gaussian_filter(b, 4)


def _parabola(rm, r0, rp):
    den = rm - 2 * r0 + rp
    return 0.0 if den >= 0 else float(np.clip(0.5 * (rm - rp) / den, -0.5, 0.5))


def offset(point, vol):
    """(dz, dy, dx) sub-voxel, r at the integer peak, r at (0,0,0)."""
    a = _box(REC['standard'], point, 0, 0)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b0 = _box(vol, point, MZ, MYX)
    sz, sy = 2 * BZ, 2 * BYX
    grid = np.full((2 * MZ + 1, 2 * MYX + 1, 2 * MYX + 1), -2.0)
    for i, dz in enumerate(range(-MZ, MZ + 1)):
        for j, dy in enumerate(range(-MYX, MYX + 1)):
            for k, dx in enumerate(range(-MYX, MYX + 1)):
                b = b0[MZ + dz:MZ + dz + sz, MYX + dy:MYX + dy + sy, MYX + dx:MYX + dx + sy]
                grid[i, j, k] = float((a * (b - b.mean())).mean() / (b.std() + 1e-9))
    i, j, k = np.unravel_index(np.argmax(grid), grid.shape)
    sub = []
    for ax, (idx, m) in enumerate(zip((i, j, k), (MZ, MYX, MYX))):
        if 0 < idx < grid.shape[ax] - 1:
            sl = [i, j, k]
            vals = []
            for t in (idx - 1, idx, idx + 1):
                sl[ax] = t
                vals.append(grid[tuple(sl)])
            sub.append(idx - m + _parabola(*vals))
        else:
            sub.append(float(idx - m))
    return [round(s, 2) for s in sub], round(float(grid[i, j, k]), 3), round(float(grid[MZ, MYX, MYX]), 3)


def main():
    out = sys.argv[1]
    pts = [int(v) for v in sys.argv[2:]] or [6000, 3300, 2600, 12000, 2500, 3300]
    res = []
    for p in zip(pts[0::3], pts[1::3], pts[2::3]):
        for k in ('pag0', 'pag50'):
            s, r, r0 = offset(p, REC[k])
            e = {'point_zyx': list(p), 'volume': k, 'shift_dz_dy_dx': s, 'r_peak': r, 'r_at_zero': r0}
            res.append(e)
            print(e, flush=True)
            json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
