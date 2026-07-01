import sys, torch
sys.path.insert(0, r'c:\project\Spine Segmentation\ATM-Net++')

print("=" * 50)
print("Testing all model components")
print("=" * 50)

# 1. Rule-based text parser
try:
    from models.text_encoder.bio_clinical_bert import RuleBasedTextParser
    parser = RuleBasedTextParser()
    result = parser.parse("L4-L5 disc herniation with moderate stenosis at L3-L4")
    print(f"\n[1] RuleBasedParser: OK")
    print(f"    pathologies: {result['pathologies']}")
    print(f"    levels     : {result['levels']}")
    print(f"    severity   : {result['severity']}")
except Exception as e:
    print(f"\n[1] RuleBasedParser ERROR: {e}")

# 2. Disease classifier
try:
    from models.classification.disease_classifier import MultiTaskHead
    head = MultiTaskHead(input_dim=512)
    dummy = torch.randn(1, 512)
    out = head(dummy)
    dp = out["disease"]["pred"].item()
    sp = out["severity"]["pred"].item()
    pf = out["ivd_pathology"]["pfirrmann_score"].item()
    print(f"\n[2] MultiTaskHead: OK")
    print(f"    disease pred  : {dp}")
    print(f"    severity pred : {sp}")
    print(f"    pfirrmann     : {pf:.2f}")
    print(f"    output keys   : {list(out.keys())}")
except Exception as e:
    print(f"\n[2] MultiTaskHead ERROR: {e}")

# 3. Fusion module
try:
    from models.fusion.multimodal_fusion import MultimodalFusionModule
    fusion = MultimodalFusionModule()
    img_f = torch.randn(1, 768)
    txt_f = torch.randn(1, 768)
    demo  = torch.randn(1, 8)
    out = fusion(img_f, txt_f, demo)
    print(f"\n[3] Fusion Module: OK")
    print(f"    fused shape: {out['fused_features'].shape}")
    print(f"    output keys: {list(out.keys())}")
except Exception as e:
    print(f"\n[3] Fusion ERROR: {e}")

print("\n" + "=" * 50)
print("Component test complete")
print("=" * 50)
