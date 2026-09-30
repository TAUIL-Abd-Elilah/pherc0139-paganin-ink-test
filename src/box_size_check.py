"""Is the 3-D box offset (src/align_check.py) stable against box size, and does it agree with registering the
rendered sheet (results/renders.json) at the same point? Run after src/ink_test.py.
usage: python src/box_size_check.py results/renders.json results/alignment_box_sizes.json [segment ...]"""
import json, sys

sys.path.insert(0, 'src')
import align_check as ac  # noqa: E402
from volumes import REC  # noqa: E402


def main():
    ren = json.load(open(sys.argv[1]))
    segs = sys.argv[3:] or sorted(ren)
    rep = {}
    for seg in segs:
        p = ren[seg]['centre_vertex_zyx']
        e = {'point_zyx': p, 'sheet_registration_dz_dy_dx': ren[seg]['offset_vs_standard']['pag50']['final_dz_dy_dx'], 'boxes': []}
        for bz, byx in ((16, 48), (32, 96), (48, 160), (64, 256)):
            ac.BZ, ac.BYX = bz, byx
            s, rp, rz = ac.offset(p, REC['pag50'])
            e['boxes'].append({'box_zyx': [2 * bz, 2 * byx, 2 * byx], 'dz_dy_dx': [float(x) for x in s], 'r_peak': rp})
        rep[seg] = e
        print(seg.split('-')[1][:4], e, flush=True)
    json.dump(rep, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
