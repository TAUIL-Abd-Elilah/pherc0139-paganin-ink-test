"""Figures for the README, from results/{renders,fair,models,fair_cnn}.json and the renders and model maps in $PAG_OUT.
usage: python src/figures.py [segment for figure 1]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage

sys.path.insert(0, 'src')
from models import OUT, RECS, masks  # noqa: E402

NAME = {'standard': 'δ/β 1000 (standard)', 'pag50': 'δ/β 50', 'pag0': 'δ/β ≈ 0 (no Paganin)'}
COL = {'standard': '#1f5fa8', 'pag50': '#d08a1c', 'pag0': '#b8323a'}


def short(seg):
    return seg.split('-')[1][:4]


def fig_views(ren, mod, seg, out, h=400, w=1200):
    """Same patch in the three reconstructions: CT (mean of the 8 centre layers) with the labels outlined, and the
    ink_9um map (trained on the standard reconstruction of this very segment). The crop is the h x w window with
    the most labelled ink."""
    d = os.path.join(OUT, seg)
    s0 = np.load(os.path.join(d, 'standard.npy'), mmap_mode='r')
    Lf = masks(seg, ren[seg], s0.shape)
    dens = ndimage.uniform_filter(Lf.astype(np.float32), (h, w), mode='constant')[h // 2:-h // 2, w // 2:-w // 2]
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y0, x0 = int(cy), int(cx); y1, x1 = y0 + h, x0 + w
    L = Lf[y0:y1, x0:x1]
    edge = L ^ ndimage.binary_erosion(L, iterations=2)
    order = mod[seg]['ink9um_s42']['layer_order']
    fig, ax = plt.subplots(len(RECS), 2, figsize=(13, 2.35 * len(RECS) + 0.5), constrained_layout=True)
    for i, n in enumerate(RECS):
        s = np.load(os.path.join(d, n + '.npy'), mmap_mode='r')
        c = s.shape[0] // 2
        img = np.asarray(s[c - 4:c + 4, y0:y1, x0:x1], np.float32).mean(0)
        lo, hi = np.percentile(img, (1, 99))
        rgb = np.repeat(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)[..., None], 3, -1)
        rgb[edge] = (1.0, 0.8, 0.0)
        ax[i, 0].imshow(rgb, interpolation='none')
        ax[i, 0].set_title(f'{NAME[n]}: CT, mean of the 8 centre layers, labels outlined', fontsize=9)
        f = os.path.join(d, f'{n}_ink9um_s42' + ('_reverse' if order == 'reversed' else '') + '.tif')
        p = tifffile.imread(f).astype(np.float32)
        p = p / 255 if p.max() > 1.5 else p
        ax[i, 1].imshow(p[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1, interpolation='none')
        ax[i, 1].set_title(f"{NAME[n]}: ink_9um s42 (trained on δ/β 1000), AUC {mod[seg]['ink9um_s42'][n]:.3f} on the whole patch", fontsize=9)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'PHerc0139 {short(seg)}: the same {(x1 - x0) * 9.362e-3:.1f} x {(y1 - y0) * 9.362e-3:.1f} mm of papyrus in the three reconstructions', fontsize=10)
    fig.savefig(out, dpi=110); plt.close(fig)


def fig_contrast(fair, out, sig=2):
    """Top: each segment's ink-contrast depth profile in the three reconstructions (sigma = 2 px). Bottom right:
    the new volumes' profile as a fraction of the standard one (least-squares scale k) against smoothing."""
    segs = sorted(fair['segments'])
    fig = plt.figure(figsize=(16, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, len(segs))
    for j, seg in enumerate(segs):
        a = fig.add_subplot(gs[0, j])
        e = fair['segments'][seg]
        for n in RECS:
            y = np.array(e[n][f'd_sigma{sig}']['by_layer'])
            a.plot(np.arange(len(y)) - (len(y) - 1) / 2, y, color=COL[n], lw=2, label=NAME[n])
        a.axhline(0, color='0.6', lw=0.6)
        a.axvspan(-4, 4, color='0.9', zorder=0)
        c0 = e['pag0']['vs_standard'][f'sigma{sig}']['core_l2_ratio']; c5 = e['pag50']['vs_standard'][f'sigma{sig}']['core_l2_ratio']
        a.set_title(f'{short(seg)}: near surface {c0:.2f} (δ/β≈0), {c5:.2f} (δ/β 50)', fontsize=9)
        a.set_xlabel('layer offset (voxels; outward <- -> inward)', fontsize=8)
        if j == 0:
            a.set_ylabel(f"Cohen's d, ink vs papyrus\n(σ = {sig} px)")
            a.legend(fontsize=7, loc='upper right')
    a = fig.add_subplot(gs[1, len(segs) - 2:])
    sigs = (0, 1, 2, 4, 8)
    for n in RECS[1:]:
        for key, ls, lab in (('core_l2_ratio', '-', 'near the surface (±4 voxels)'), ('outer_l2_ratio', ':', 'far (≥ 7 voxels)')):
            v = np.array([[fair['segments'][g][n]['vs_standard'][f'sigma{t}'][key] for t in sigs] for g in segs])
            a.fill_between(sigs, v.min(0), v.max(0), color=COL[n], alpha=0.12)
            a.plot(sigs, np.median(v, 0), 'o' + ls, color=COL[n], label=f'{NAME[n]}, {lab}')
    a.axhline(1, color=COL['standard'], lw=1.5, ls='--', label='δ/β 1000 (reference)')
    a.set_ylim(0, 1.45); a.set_xlabel('in-plane smoothing σ (px; 1 px = 9.362 µm)'); a.set_ylabel('contrast / standard')
    a.set_title('New volumes / standard: median, band = range over 5 segments', fontsize=9)
    a.legend(fontsize=7, loc='lower right', ncol=1, framealpha=0.9)
    t = fig.add_subplot(gs[1, :len(segs) - 2]); t.axis('off')
    t.text(0, 0.95, "Same ink signature; at the surface about the same, away from it weaker.\n\n"
           "Each curve: Cohen's d between labelled ink and unlabelled papyrus within 2 mm,\n"
           "per layer of the 24-layer render along the mesh normal, for the same registered\n"
           "pixels in the three reconstructions. Grey band: within 4 voxels of the traced\n"
           "surface. Titles: contrast there (L2 over those layers) relative to the standard\n"
           "volume, at σ = 2 px. Away from the surface the standard volume keeps more\n"
           "contrast on every segment.", va='top', fontsize=10)
    fig.savefig(out, dpi=110); plt.close(fig)


def fig_auc(cnn, mod, out):
    segs = sorted(mod)
    panels = []
    for m, t in (('ink9um_s42', 'ink_9um s42 (trained on δ/β 1000,\nincluding these five segments)'),
                 ('ink9um_s43', 'ink_9um s43 (same)'),
                 ('reader_v2', 'Reader v2 (trained on δ/β 1000 data)'), ('v8in', 'v8-in (trained on δ/β 1000 data)')):
        if all(m in mod[s] for s in segs):
            panels.append((m, t))
    if cnn and 'summary' in cnn:
        panels.append(('cnn', 'small U-Net trained from scratch on each\nreconstruction, leave-one-segment-out'))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8), constrained_layout=True, sharey=True)
    w = 0.26
    for a, (m, t) in zip(np.atleast_1d(ax), panels):
        for i, n in enumerate(RECS):
            x = np.arange(len(segs)) + (i - 1) * w
            v = [cnn['summary'][n][s] if m == 'cnn' else mod[s][m][n] for s in segs]
            a.bar(x, v, w, color=COL[n], label=NAME[n])
        a.set_xticks(np.arange(len(segs))); a.set_xticklabels([short(s) for s in segs], fontsize=8)
        a.set_ylim(0.5, 1.0); a.set_title(t, fontsize=9)
    np.atleast_1d(ax)[0].set_ylabel('AUC, ink vs papyrus within 2 mm')
    np.atleast_1d(ax)[0].legend(fontsize=8, loc='lower left')
    fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ren = json.load(open('results/renders.json'))
    fair = json.load(open('results/fair.json'))
    mod = json.load(open('results/models.json')) if os.path.exists('results/models.json') else {}
    cnn = json.load(open('results/fair_cnn.json')) if os.path.exists('results/fair_cnn.json') else {}
    os.makedirs('figures', exist_ok=True)
    fig_contrast(fair, 'figures/02_contrast_by_depth.png')
    if mod:
        fig_auc(cnn, mod, 'figures/03_auc.png')
        seg = sys.argv[1] if len(sys.argv) > 1 else '20250831000000-w040_2025083102'
        fig_views(ren, mod, seg, 'figures/01_same_patch.png')


if __name__ == '__main__':
    main()
