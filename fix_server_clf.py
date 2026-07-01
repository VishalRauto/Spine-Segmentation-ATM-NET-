"""Patch server.py SpineClassifier to use v2 residual architecture."""
path = "server.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# Find the class block start
marker_start = "                class _SpineClassifier(nn.Module):"
marker_end   = "                        }"   # end of forward return dict

idx_start = src.find(marker_start)
if idx_start == -1:
    print("ERROR: marker not found"); exit(1)

# Find the end of the class (the closing brace of forward return + 2 newlines)
idx_end = src.find(marker_end, idx_start)
if idx_end == -1:
    print("ERROR: end marker not found"); exit(1)
idx_end = idx_end + len(marker_end)

new_class = '''                class _SpineClassifier(nn.Module):
                    """SpineClassifierV2 architecture — residual MLP."""
                    def __init__(self, feat_dim=104, fusion_dim=128,
                                 num_disease=3, num_severity=3,
                                 num_levels=8, dropout=0.0):
                        super().__init__()
                        import torch.nn.functional as _F2
                        self.num_disease = num_disease
                        self.stem = nn.Sequential(
                            nn.Linear(feat_dim, fusion_dim),
                            nn.BatchNorm1d(fusion_dim), nn.GELU(),
                        )
                        self.res1 = nn.Sequential(
                            nn.Linear(fusion_dim, fusion_dim),
                            nn.BatchNorm1d(fusion_dim), nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(fusion_dim, fusion_dim),
                            nn.BatchNorm1d(fusion_dim),
                        )
                        self.res2 = nn.Sequential(
                            nn.Linear(fusion_dim, fusion_dim),
                            nn.BatchNorm1d(fusion_dim), nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(fusion_dim, fusion_dim),
                            nn.BatchNorm1d(fusion_dim),
                        )
                        self.disease_head  = nn.Sequential(
                            nn.Dropout(dropout*0.5),
                            nn.Linear(fusion_dim, num_disease))
                        self.severity_head = nn.Sequential(
                            nn.Dropout(dropout*0.5),
                            nn.Linear(fusion_dim, num_severity))
                        self.level_head    = nn.Sequential(
                            nn.Dropout(dropout*0.3),
                            nn.Linear(fusion_dim, num_levels))
                        self.pfi_head      = nn.Sequential(
                            nn.Dropout(dropout*0.2),
                            nn.Linear(fusion_dim, num_levels),
                            nn.Sigmoid())

                    def forward(self, x):
                        import torch.nn.functional as _F
                        h  = self.stem(x)
                        h  = _F.gelu(h + self.res1(h))
                        h  = _F.gelu(h + self.res2(h))
                        dl = self.disease_head(h)
                        return {
                            "disease_logits":  dl,
                            "disease_probs":   _F.softmax(dl, -1),
                            "severity_logits": self.severity_head(h),
                            "level_logits":    self.level_head(h),
                            "pfirrmann":       self.pfi_head(h)*4.0+1.0,
                            "mean_pfirrmann":  (self.pfi_head(h)*4.0+1.0).mean(-1),
                        }'''

src = src[:idx_start] + new_class + src[idx_end:]

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched server.py SpineClassifier to v2 architecture")
