import torch
c = torch.load("outputs/classifier/feature_cache.pt",
               map_location="cpu", weights_only=False)
keys = list(c.keys())
print("Total:", len(c))
print("Keys sample:", keys[:5])
if keys:
    v = c[keys[0]]
    print("Value type:", type(v))
    if isinstance(v, dict):
        print("Value keys:", list(v.keys()))
