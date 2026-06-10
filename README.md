# ATM-Net++: Anatomy-Aware Multimodal Lumbar Spine MRI Diagnostic System

> **Production-grade clinical AI platform** for automated lumbar spine MRI segmentation, disease classification, severity estimation, and explainable report generation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Data Setup](#data-setup)
6. [Training](#training)
7. [Inference](#inference)
8. [API Reference](#api-reference)
9. [Frontend](#frontend)
10. [Docker Deployment](#docker-deployment)
11. [Evaluation](#evaluation)
12. [Testing](#testing)
13. [Research Foundation](#research-foundation)

---

## Overview

ATM-Net++ accepts:
- **Lumbar spine MRI** (MHA, NIfTI, DICOM, PNG/JPG) — T1 and T2 modalities
- **Radiology report text** (processed via Bio-ClinicalBERT)
- **Patient demographics** (age, sex, BMI, symptoms)

And produces:
- **Pixel-level segmentation** of vertebrae (T10–S1), IVDs, spinal canal, spinal cord
- **Disease classification** — 7 categories with confidence scores
- **Severity estimation** — Mild / Moderate / Severe + Pfirrmann grade
- **Level localization** — which disc levels are affected
- **Grad-CAM explainability** heatmaps + attention maps
- **Clinical PDF report** — radiologist-style structured findings

**Target metric:** Dice Score > 0.90

---

## Architecture

```
Input (MRI + Report Text + Demographics)
        │
        ├─► Swin UNETR Backbone
        │     └─ Residual blocks + Attention gates + Deep supervision
        │     └─ Image features (768-dim)
        │
        ├─► Bio-ClinicalBERT Text Encoder
        │     └─ Frozen early layers + fine-tuned head
        │     └─ CLS embedding (768-dim) + token embeddings
        │
        └─► Demographic MLP Encoder (256-dim)
                │
                ▼
        ┌─────────────────────────┐
        │  Multimodal Fusion      │
        │  ┌──────────────────┐   │
        │  │ ATPG: Anatomy-   │   │
        │  │ Text Prompts     │   │
        │  └──────────────────┘   │
        │  ┌──────────────────┐   │
        │  │ HASF: Hierarchi- │   │
        │  │ cal Fusion       │   │
        │  └──────────────────┘   │
        │  ┌──────────────────┐   │
        │  │ Transformer      │   │
        │  │ Fusion Layers    │   │
        │  └──────────────────┘   │
        └─────────────────────────┘
                │
                ▼
        Multi-Task Heads
        ├─ Segmentation (CCAE-enhanced)
        ├─ Disease Classification (7 classes)
        ├─ Severity Estimation (3 classes + regression)
        ├─ Level Localization (8 IVD levels, multi-label)
        └─ Report Generation
```

---

## Project Structure

```
ATM-Net++/
├── configs/
│   └── base_config.yaml          # All hyperparameters
├── models/
│   ├── atmnet_plus_plus.py       # Main model class
│   ├── segmentation/
│   │   └── swin_unetr_backbone.py # Encoder + attention decoder
│   ├── text_encoder/
│   │   └── bio_clinical_bert.py  # Bio-ClinicalBERT wrapper
│   ├── fusion/
│   │   └── multimodal_fusion.py  # ATPG + HASF + CCAE + Transformer
│   ├── classification/
│   │   └── disease_classifier.py # Multi-task heads
│   ├── report_generator/
│   │   └── clinical_report.py    # Template + neural report
│   └── explainability/
│       └── grad_cam.py           # Grad-CAM + attention rollout
├── datasets/
│   ├── loaders/
│   │   └── spider_dataset.py     # SPIDER dataset PyTorch Dataset
│   ├── transforms/
│   │   └── augmentations.py      # Full augmentation pipeline
│   └── preprocessing/
│       ├── mha_reader.py         # Multi-format medical image reader
│       ├── normalizer.py         # Intensity normalization
│       └── label_mapper.py       # SPIDER → ATM-Net++ label remapping
├── training/
│   ├── train.py                  # Main training entry point
│   ├── trainer.py                # Training engine (AMP, checkpointing)
│   ├── losses/
│   │   └── combined_loss.py      # Dice + Focal + Boundary + Contrastive
│   └── metrics/
│       └── segmentation_metrics.py # Dice, IoU, HD95, ASD, F1
├── evaluation/
│   └── evaluator.py              # Full test-set evaluation
├── inference/
│   └── predictor.py              # Production inference engine + TTA
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── core/
│   │   ├── config.py             # Settings (pydantic-settings)
│   │   └── security.py           # JWT auth
│   ├── api/
│   │   ├── routes/               # auth, predict, patients, reports, analytics
│   │   ├── middleware/           # JWT dependency
│   │   └── schemas/              # Pydantic request/response models
│   ├── db/
│   │   ├── database.py           # Async SQLAlchemy
│   │   └── models/models.py      # ORM: Users, Patients, Studies, Predictions, Reports
│   └── services/
│       ├── model_service.py      # Singleton model loader
│       └── pdf_service.py        # ReportLab PDF generation
├── frontend/
│   ├── src/app/                  # Next.js 14 App Router pages
│   │   ├── auth/login/           # Login page
│   │   ├── dashboard/            # Analytics dashboard
│   │   ├── upload/               # MRI upload + results
│   │   └── layout.tsx            # Root layout + providers
│   ├── src/components/           # Reusable UI components
│   ├── src/lib/
│   │   ├── api.ts                # Full API client (TypeScript)
│   │   └── utils.ts              # Utilities
│   └── src/store/
│       └── authStore.ts          # Zustand auth state
├── tests/
│   ├── unit/                     # Losses, metrics, preprocessing, model
│   ├── integration/              # FastAPI endpoint tests
│   └── conftest.py               # Pytest fixtures
├── scripts/
│   ├── setup_data.py             # Link/copy SPIDER dataset
│   ├── run_inference.py          # Standalone inference CLI
│   ├── evaluate.py               # Full evaluation script
│   └── export_onnx.py            # ONNX export for deployment
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.frontend
│   │   └── init.sql
│   └── nginx/
│       └── nginx.conf
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── pytest.ini
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20+
- Docker + Docker Compose (for full stack)
- 8GB+ RAM (16GB+ recommended for training)
- GPU optional for inference, required for fast training

### 1. Clone and Install

```bash
cd "C:\project\Spine Segmentation\ATM-Net++"

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Set up environment

```bash
copy .env.example .env
# Edit .env with your SECRET_KEY and database settings
```

### 3. Set up data

```bash
python scripts/setup_data.py
# This links the SPIDER dataset from C:\project\Spine Segmentation\10159290
```

### 4. Start the full stack (Docker)

```bash
docker-compose up --build
```

Then visit:
- **Frontend:** http://localhost:3000
- **API docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

---

## Data Setup

The system uses the **SPIDER dataset** (Lumbar Spine MRI Segmentation):

| File | Description |
|------|-------------|
| `images/*.mha` | Sagittal T1/T2 MRI volumes |
| `masks/*.mha` | Segmentation labels |
| `overview.csv` | MRI acquisition metadata + sex/subset |
| `radiological_gradings.csv` | Per-IVD pathology grades |

**Label mapping:**
| SPIDER ID | Structure | ATM-Net++ ID |
|-----------|-----------|--------------|
| 20 | L1 | 4 |
| 21 | L2 | 5 |
| 22 | L3 | 6 |
| 23 | L4 | 7 |
| 24 | L5 | 8 |
| 25 | S1 | 9 |
| 122 | L4/L5 disc | 16 |
| 123 | L5/S1 disc | 17 |
| 201 | Spinal canal | 18 |

```bash
python scripts/setup_data.py --source "C:\project\Spine Segmentation\10159290"
```

---

## Training

```bash
# Basic training
python training/train.py --config configs/base_config.yaml

# With W&B logging
python training/train.py --config configs/base_config.yaml --experiment my_run

# Resume from checkpoint
python training/train.py --config configs/base_config.yaml \
    --resume checkpoints/atmnet_pp_epoch_50.pth

# Debug mode (2 epochs, small batch)
python training/train.py --config configs/base_config.yaml --debug
```

**Training features:**
- Mixed precision (FP16) via `torch.cuda.amp`
- Gradient accumulation (effective batch size = batch × accum_steps)
- Cosine LR schedule with linear warmup
- Early stopping (patience=30 on val Dice)
- TensorBoard + W&B logging
- Automatic best-model checkpointing

**Loss functions:**
- Segmentation: `1.0 × Dice + 0.5 × Focal + 0.2 × Boundary`
- Classification: Focal CE
- Feature alignment: NT-Xent contrastive
- Deep supervision: Dice on 3 intermediate decoder outputs

---

## Inference

### Command line

```bash
# Single MRI file
python scripts/run_inference.py \
    --image data/10159290/images/100_t2.mha \
    --checkpoint checkpoints/atmnet_pp_best.pth \
    --report "Posterior disc bulge at L4-L5" \
    --age 55 --sex F \
    --save-overlay outputs/overlay.png \
    --save-json outputs/results.json \
    --save-report outputs/report.pdf
```

### Python API

```python
from inference.predictor import SpinePredictor
import yaml

with open("configs/base_config.yaml") as f:
    config = yaml.safe_load(f)

predictor = SpinePredictor.from_checkpoint(
    "checkpoints/atmnet_pp_best.pth",
    config=config,
)

result = predictor.predict_from_file(
    image_path="data/10159290/images/100_t2.mha",
    report_text="Disc bulge at L4-L5",
    demographics={"age": 55, "sex": "F"},
)

print(result["classification"]["disease_name"])     # "Disc_Bulge"
print(result["severity"]["name"])                   # "Moderate"
print(result["levels"]["affected"])                 # ["L4/L5"]
print(result["report"]["findings"])                 # Full text
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/login` | Login, get JWT |
| GET  | `/api/v1/auth/me` | Current user |
| POST | `/api/v1/predict/upload-mri` | Upload MRI + predict |
| POST | `/api/v1/predict/segment` | Segmentation only |
| POST | `/api/v1/patients` | Create patient |
| GET  | `/api/v1/patients` | List patients |
| GET  | `/api/v1/reports/study/{id}` | Get report |
| GET  | `/api/v1/reports/download/{id}/pdf` | Download PDF |
| GET  | `/api/v1/analytics/summary` | Dashboard stats |
| GET  | `/health` | Health check |

Full interactive docs at **http://localhost:8000/docs**

---

## Frontend

Built with **Next.js 14**, **TypeScript**, **Tailwind CSS**.

**Pages:**
| Route | Description |
|-------|-------------|
| `/auth/login` | JWT login |
| `/dashboard` | Analytics overview, charts |
| `/upload` | Drag-drop MRI upload, inline results |
| `/patients` | Patient CRUD |

```bash
cd frontend
npm install
npm run dev      # Development: http://localhost:3000
npm run build    # Production build
```

---

## Docker Deployment

```bash
# Full stack (backend + frontend + db + redis + nginx)
docker-compose up --build -d

# Backend only
docker-compose up backend db redis -d

# View logs
docker-compose logs -f backend

# Stop
docker-compose down
```

**Services:**
| Service | Port | Description |
|---------|------|-------------|
| nginx | 80 | Reverse proxy |
| frontend | 3000 | Next.js |
| backend | 8000 | FastAPI |
| db | 5432 | PostgreSQL |
| redis | 6379 | Cache / task queue |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/atmnet_pp_best.pth \
    --compute-hd \
    --save-predictions
```

**Metrics computed:**
- Dice Score (per-class + mean)
- Jaccard/IoU
- HD95 (95th percentile Hausdorff Distance)
- ASD (Average Surface Distance)
- Precision, Recall, F1
- Disease classification accuracy + macro F1

---

## Testing

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running backend)
pytest tests/integration/ -v -m integration

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=. --cov-report=html
```

---

## Research Foundation

ATM-Net++ extends:
- **ATM-Net** (Anatomy-aware Text-guided segmentation)
- **Swin UNETR** (Liu et al., 2021) — Vision Transformer for medical image segmentation
- **Bio-ClinicalBERT** (Alsentzer et al., 2019)
- **Attention U-Net** (Oktay et al., 2018) — Attention gates on skip connections
- **Boundary Loss** (Kervadec et al., 2019)
- **Focal Loss** (Lin et al., 2017)
- **GradCAM** (Selvaraju et al., 2017)

Novel contributions:
1. **ATPG** — Anatomy-Text Prompt Generation guided by image features
2. **HASF** — Hierarchical Anatomy-aware Semantic Fusion with cross-modal attention
3. **CCAE** — Cross-modal Context-Aware Enhancement using FiLM conditioning
4. **Joint multimodal multi-task** training with contrastive image-text alignment

---

## License

Research use only. Not for clinical deployment without validation.

---

*ATM-Net++ v1.0.0 · Built for the SPIDER Lumbar Spine MRI Dataset*
