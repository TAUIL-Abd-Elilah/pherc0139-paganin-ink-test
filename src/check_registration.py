"""Check that the pag0 / pag50 renders show the same papyrus as the standard render, in three parts of each window
(left, middle, right thirds of its width): sub-pixel shift of the middle layer that best matches the standard
render (src/ink_test.py:residual). One offset per window cannot follow an offset that changes along the window,
so this is the residual that remains.
usage: python src/check_registration.py results/renders.json results/registration.json"""
import json, os, sys
import numpy as np

sys.path.insert(0, 'src')
from ink_test import residual, OUT  # noqa: E402


def main():
    ren = json.load(open(sys.argv[1]))
    rep = {}
    for seg in sorted(ren):
        d = os.path.join(OUT, seg)
        a = np.load(os.path.join(d, 'standard.npy'), mmap_mode='r')
        c, H, W = a.shape[0] // 2, a.shape[1], a.shape[2]
        e = {}
        for n in ('pag50', 'pag0'):
            b = np.load(os.path.join(d, n + '.npy'), mmap_mode='r')
            parts = []
            for x0 in (0, W // 3, 2 * W // 3):
                (dy, dx), r = residual(np.asarray(a[c, 50:H - 50, x0:x0 + W // 3]), np.asarray(b[c, 50:H - 50, x0:x0 + W // 3]))
                parts.append({'dy_px': round(dy, 2), 'dx_px': round(dx, 2), 'r': round(r, 3)})
            e[n] = parts
        rep[seg] = e
        print(seg.split('-')[1][:4], {n: [(p['dy_px'], p['dx_px']) for p in v] for n, v in e.items()}, flush=True)
    json.dump(rep, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
