"""The three reconstructions of PHerc0139's 9.362 um scan (same acquisition, 2025-07-20, BM18, 113 keV, 1.2 m),
as listed in each volume's metadata.json on the open-data bucket."""
import numpy as np
from scipy import ndimage

import zfetch as zf

REC = {'standard': 'PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr',          # delta/beta 1000, unsharp 4.0 / 1.2
       'pag50': 'PHerc0139/volumes/20251107135911-9.362um-1.2m-113keV-pag50-masked.zarr',       # delta/beta 50, unsharp 4.0 / 1.2
       'pag0': 'PHerc0139/volumes/20251107132835-9.362um-1.2m-113keV-pag0-masked.zarr'}         # delta/beta 0.0001, no unsharp
# pag[z - 1, y, x] = standard[z, y, x] (measured: src/align_check.py -> results/alignment.json)
ZOFF = {'standard': 0, 'pag50': -1, 'pag0': -1}
_centres = {}


def centre(vol, z):
    """Scroll centre (y, x) in level-0 voxels at height z: centroid of the level-4 mask slice."""
    key = (vol, z // 512)
    if key not in _centres:
        s = zf.meta(vol, 4)['shape']
        zz = min(s[0] - 1, z // 16)
        sl = zf.read_box(vol, 4, (zz, 0, 0), (zz + 1, s[1], s[2]))[0] > 0
        cy, cx = ndimage.center_of_mass(sl)
        _centres[key] = (cy * 16, cx * 16)
    return _centres[key]
