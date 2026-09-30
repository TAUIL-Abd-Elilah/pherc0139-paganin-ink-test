import urllib.request, os, json, sys, numpy as np, zarr, cv2
B='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
def get_label(seg, scan, level=0, name='inklabels'):
    root=f'PHerc0139/segments/{seg}/ink-labels/{scan}/20260918/{name}.zarr'
    d=os.path.join(os.environ.get('XFORM_OUT', 'out'), seg, scan, name); os.makedirs(os.path.join(d, str(level), 'c', '0'), exist_ok=True)
    for f in ('zarr.json', f'{level}/zarr.json', f'{level}/c/0/0'):
        p=os.path.join(d, f)
        if not os.path.exists(p): urllib.request.urlretrieve(f'{B}/{root}/{f}', p)
    return zarr.open_array(os.path.join(d, str(level)), mode='r')[:]
def ncc(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
if __name__=='__main__':
    seg=sys.argv[1]
    L24=get_label(seg,'2.399um-volume-20260102150214'); L93=get_label(seg,'9.362um-volume-20250728140407')
    print(seg.split('-')[1][:4], '2.4 labels', L24.shape, '9.362 labels', L93.shape, 'ratio %.4f %.4f'%(L24.shape[0]/L93.shape[0], L24.shape[1]/L93.shape[1]), 'ink frac %.3f %.3f'%((L24>0).mean(), (L93>0).mean()))
    R=cv2.resize((L24>0).astype(np.float32),(L93.shape[1],L93.shape[0]),interpolation=cv2.INTER_AREA); T=(L93>0).astype(np.float32)
    best=(-2,0,0)
    for dy in range(-24,25,2):
        for dx in range(-24,25,2):
            r=ncc(np.roll(np.roll(R,dy,0),dx,1)[30:-30,30:-30],T[30:-30,30:-30])
            if r>best[0]: best=(r,dy,dx)
    print('  resized 2.4 labels vs published 9.362 labels: r(0,0) %.3f; best %.3f at dy %d dx %d'%(ncc(R,T),*best))
