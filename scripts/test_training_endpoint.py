import requests, json

r = requests.get('http://localhost:5000/training', timeout=15)
d = r.json()

h  = d.get('history', [])
pc = d.get('per_class', {})
ck = d.get('checkpoints', {})

print(f"History   : {len(h)} epochs")
if h:
    last = h[-1]
    best = max(h, key=lambda x: x.get('vd', 0))
    print(f"Last epoch: ep={last['ep']}  vd={last['vd']}  tl={last['tl']}")
    print(f"Best epoch: ep={best['ep']}  vd={best['vd']}")

print(f"\nCheckpoints:")
for name, info in ck.items():
    print(f"  {name}: ep={info.get('epoch')} dice={info.get('best_dice')} base_ch={info.get('base_ch')} sz={info.get('img_size')}")

print(f"\nPer-class ({len(pc)} classes):")
for k, v in sorted(pc.items(), key=lambda x: -x[1])[:10]:
    bar = '█' * int(v * 20)
    print(f"  {k:20s}: {v:.4f}  {bar}")
