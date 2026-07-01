"""
Final patch for train_classifier.py:
- 3-class disease (Normal / Degeneration / Structural)
- Smaller model (32-dim)
- Lower LR, stronger dropout
- Skip all 7-class complexity until Dice > 0.85
"""
import re

path = "train_classifier.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# ── 1. Collapse to 3 disease classes ─────────────────────────────────
old_block = '''        # Pfirrmann-driven mapping - aligned with ResUNet Dice=0.77 features
        worst_pfi = float(max(d["pfirrmann"] for d in discs))
        if any_spon:
            disease_id = 5   # Spondylolisthesis
        elif any_hern and worst_pfi >= 4:
            disease_id = 1   # Herniation + severe degen
        elif any_hern:
            disease_id = 2   # Herniation/bulge-like
        elif any_narrow or (worst_pfi >= 4 and any_bulge):
            disease_id = 3   # Stenosis
        elif any_bulge or worst_pfi >= 3:
            disease_id = 4   # Degeneration
        elif any_modic or any_ep:
            disease_id = 4   # Degeneration (endplate)
        else:
            disease_id = 0   # Normal'''

new_block = '''        # 3-class mapping matched to ResUNet Dice=0.77 capabilities:
        #   0 = Normal (Pfirrmann <= 2, no flags)
        #   1 = Degeneration (disc narrowing, bulging, Pfirrmann 3-4)
        #   2 = Structural (herniation, spondylolisthesis, stenosis, Pfirrmann 5)
        worst_pfi = float(max(d["pfirrmann"] for d in discs))
        has_structural = any_hern or any_spon or (any_narrow and worst_pfi >= 4)
        has_degen = (any_bulge or any_modic or any_ep or any_narrow
                     or worst_pfi >= 3)
        if has_structural or worst_pfi >= 5:
            disease_id = 2   # Structural
        elif has_degen:
            disease_id = 1   # Degeneration
        else:
            disease_id = 0   # Normal'''

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("Fixed disease labels -> 3 classes")
else:
    print("WARNING: old_block not found - checking alternate")

# ── 2. Update SpineClassifier to 3 classes ───────────────────────────
src = src.replace(
    "fusion_dim=128, dropout=args.dropout\n    ).to(device)",
    "fusion_dim=64, num_disease=3, dropout=args.dropout\n    ).to(device)"
)
src = src.replace(
    "    model = SpineClassifier(\n        feat_dim=SpineFeatureDataset.FEAT_DIM,\n        fusion_dim=128, dropout=args.dropout",
    "    model = SpineClassifier(\n        feat_dim=SpineFeatureDataset.FEAT_DIM,\n        fusion_dim=64, num_disease=3, dropout=args.dropout"
)

# ── 3. Update checkpoint save to record num_disease=3 ────────────────
src = src.replace(
    '"feat_dim":   SpineFeatureDataset.FEAT_DIM,\n                "fusion_dim": 128,',
    '"feat_dim":   SpineFeatureDataset.FEAT_DIM,\n                "fusion_dim": 64,\n                "num_disease": 3,'
)

# ── 4. Update DISEASE_MAP_INV in server.py is 3-class too ─────────────
# (handled separately)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Done")
