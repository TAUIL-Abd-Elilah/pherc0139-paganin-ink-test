"""A model trained on each reconstruction itself: a small 2-D U-Net from scratch, leave-one-segment-out.

Same pixels, labels, architecture, seeds and schedule for all three reconstructions; nothing was trained on any of
them before. Input: the 24-layer render, each layer z-scored within its segment. Training pixels: labelled ink
(positive) and unlabelled papyrus within 2 mm of it (negative), the pixels AUC is scored on; everything else is
ignored. Crops of 128 x 128 px, half of them centred on ink, random flips. Scored on the whole held-out patch.
usage: python src/fair_cnn.py results/renders.json results/fair_cnn.json [--steps 1500] [--seeds 0 1]"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, 'src')
from models import OUT, RECS, auc, masks  # noqa: E402

CROP = 128


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=24, w=32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), block(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(F.max_pool2d(a, 2)); c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([self.u2(c), b], 1)); a = self.d1(torch.cat([self.u1(b), a], 1))
        return self.out(a)[:, 0]


def load(seg, r, n):
    s = np.load(os.path.join(OUT, seg, n + '.npy')).astype(np.float32)
    s0 = np.load(os.path.join(OUT, seg, 'standard.npy'), mmap_mode='r')
    c = s.shape[0] // 2
    valid = ndimage.binary_erosion(np.asarray(s0[c]) > 0, iterations=32)
    for m in RECS[1:]:
        valid &= np.asarray(np.load(os.path.join(OUT, seg, m + '.npy'), mmap_mode='r')[c]) > 0
    for k in range(s.shape[0]):
        s[k] = (s[k] - s[k][valid].mean()) / (s[k][valid].std() + 1e-6)
    s *= valid
    L = masks(seg, r, s.shape)
    pos = L & valid
    near = (ndimage.distance_transform_edt(~L) <= 2.0 / 9.362e-3) & ~L & valid
    y = np.full(L.shape, -1, np.int8); y[near] = 0; y[pos] = 1
    return s, y


def train(data, steps, seed, dev):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    inks = [np.argwhere(y == 1) for _, y in data]
    for it in range(steps):
        xb, yb = [], []
        for _ in range(16):
            i = rng.integers(len(data)); s, y = data[i]
            H, W = y.shape
            if rng.random() < 0.5:
                cy, cx = inks[i][rng.integers(len(inks[i]))]
            else:
                cy, cx = rng.integers(H), rng.integers(W)
            y0 = int(np.clip(cy - CROP // 2, 0, H - CROP)); x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
            xs, ys = s[:, y0:y0 + CROP, x0:x0 + CROP], y[y0:y0 + CROP, x0:x0 + CROP]
            if rng.random() < 0.5:
                xs, ys = xs[:, ::-1], ys[::-1]
            if rng.random() < 0.5:
                xs, ys = xs[:, :, ::-1], ys[:, ::-1]
            xb.append(np.ascontiguousarray(xs)); yb.append(np.ascontiguousarray(ys))
        x = torch.from_numpy(np.stack(xb)).to(dev); y = torch.from_numpy(np.stack(yb)).to(dev).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logit = net(x)
        m = y >= 0
        loss = F.binary_cross_entropy_with_logits(logit.float()[m], y[m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def predict(net, s, dev, tile=512, pad=32):
    net.eval()
    C, H, W = s.shape
    out = np.zeros((H, W), np.float32)
    for y0 in range(0, H, tile - 2 * pad):
        for x0 in range(0, W, tile - 2 * pad):
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            ya, xa = max(0, y1 - tile), max(0, x1 - tile)
            x = torch.from_numpy(np.ascontiguousarray(s[:, ya:y1, xa:x1]))[None].to(dev)
            hh, ww = x.shape[-2:]
            ph, pw = (-hh) % 4, (-ww) % 4
            p = torch.sigmoid(net(F.pad(x, (0, pw, 0, ph)))[0, :hh, :ww]).float().cpu().numpy()
            iy0 = ya + (pad if ya > 0 else 0); ix0 = xa + (pad if xa > 0 else 0)
            out[iy0:y1, ix0:x1] = p[iy0 - ya:, ix0 - xa:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('renders'); ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    a = ap.parse_args()
    ren = json.load(open(a.renders)); segs = sorted(ren)
    dev = torch.device('cuda')
    rep = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for n in RECS:
        data = {seg: load(seg, ren[seg], n) for seg in segs}
        for seg in segs:
            for seed in a.seeds:
                key = f'{n}|{seg}|{seed}'
                if key in rep:
                    continue
                net = train([data[t] for t in segs if t != seg], a.steps, seed, dev)
                s, y = data[seg]
                p = predict(net, s, dev)
                rep[key] = round(auc(p, y == 1, y == 0), 4)
                json.dump(rep, open(a.out, 'w'), indent=1)
                print(n, seg.split('-')[1][:4], seed, rep[key], flush=True)
        del data
    summ = {}
    for n in RECS:
        per = {seg: float(np.mean([rep[f'{n}|{seg}|{sd}'] for sd in a.seeds])) for seg in segs}
        summ[n] = {**{k: round(v, 4) for k, v in per.items()}, 'mean': round(float(np.mean(list(per.values()))), 4)}
    rep['summary'] = summ
    json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
