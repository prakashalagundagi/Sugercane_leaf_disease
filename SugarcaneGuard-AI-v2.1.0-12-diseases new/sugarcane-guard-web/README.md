# 🌾 SugarcaneGuard AI — Website (Final)

Upload a sugarcane leaf photo → get an AI disease diagnosis with confidence,
symptoms, treatment advice, and a printable report. Backend is wired to the model
trained in `SugarcaneGuard_AI_Jupyter_Updated_Path.ipynb`
(MobileNetV2, 224×224, 12 classes).

## Quick start (your laptop)

```bash
cd sugarcane-guard-web
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:8000**.

The site starts in **demo mode** (badge in the header) so you can explore the
full UI immediately. Demo predictions are deterministic but simulated and are
clearly labelled in the results.

## Getting REAL predictions — two ways

**Option A — Train inside the website (recommended).**
Open the **Train AI** page (`/train`), paste your dataset folder path
(e.g. `D:\OneDrive\Desktop\sugarcane leaf decise detection Final\archive`)
or upload the dataset as a zip, and press **Start training**. You'll see live
per-epoch progress, timing, ETA and final test accuracy. When it finishes, the
site switches to your real model automatically — no file copying.

**Option B — Train in the notebook.**
1. Run **all cells** of `SugarcaneGuard_AI_Jupyter_Updated_Path.ipynb`
   (through cell 11 — "Save final model and class-name mapping").
2. Copy from the notebook's `models/` folder into this project's `models/` folder:
   - `sugarcane_model.keras` (final model) — or `best_sugarcane_model.keras` (best checkpoint)
   - `class_names.json`
3. `pip install tensorflow` and restart the server (`python app.py`).

The header badge switches to **"live model"** and `/api/predict` uses your
trained weights with the notebook's exact preprocessing
(`mobilenet_v2.preprocess_input` on 224×224 RGB). Each prediction also reports
its analysis time (`inference_ms`).

## What's inside

- **Leaf scanner** — drag & drop, photo tips, 3 sample images, animated result
  dashboard (diagnosis card + confidence gauge + tabs for predictions, symptoms,
  treatment), printable diagnosis report
- **Disease guide** — search + severity filters, all 12 conditions with symptoms
  and management advice
- **Technology** — model pipeline diagram, spec tables, notebook mapping
- **FAQ** — accuracy, demo mode, photo tips, privacy

## Project layout

```
sugarcane-guard-web/
├── app.py                  # FastAPI backend: /api/predict, /api/health, /api/classes
├── static/
│   ├── index.html          # page structure
│   ├── css/styles.css      # professional theme (+ print styles)
│   ├── js/app.js           # scanner, results, guide, FAQ logic
│   ├── img/                # hero + field photography
│   ├── samples.json        # "try a sample" gallery manifest
│   └── samples/            # demo leaf photos
├── data/disease_info.json  # symptoms + management advice for the 12 classes
├── models/                 # drop sugarcane_model.keras + class_names.json here
└── requirements.txt
```

## API

- `GET /api/health` → `{status, version, mode: "demo"|"real", model_file, classes}`
- `GET /api/classes` → disease info for all 12 classes
- `POST /api/predict` (multipart `image`, ≤10 MB) →
  `{mode, predicted_class, confidence, top3:[{class, confidence, info}], model_file, note}`

## The 12 classes

Mosaic · Smut · Yellow Leaf · Sett Rot · Pokkah Boeng · Banded Chlorosis · Brown Spot · Brown Rust · Grassy Shoot · Viral Disease · Healthy Leaves · Dried Leaves
