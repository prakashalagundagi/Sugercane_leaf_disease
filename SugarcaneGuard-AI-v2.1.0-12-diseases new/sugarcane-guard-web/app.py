"""
SugarcaneGuard AI - production web backend (v2.1.0)
Serves the prediction API for the model trained in
SugarcaneGuard_AI_Jupyter_Updated_Path.ipynb.

Model contract (matches the notebook):
  - MobileNetV2 transfer-learning model, input 224x224x3, float32 RGB in [0, 255]
  - keras.applications.mobilenet_v2.preprocess_input is a layer INSIDE the
    saved model, so the API must feed raw [0, 255] images (never preprocess twice)
  - softmax over 12 classes (10 diseases + Healthy Leaves + Dried Leaves)
  - saved as models/sugarcane_model.keras (final) or
           models/best_sugarcane_model.keras (best checkpoint)
  - class names in models/class_names.json

Demo mode:
  If no trained model file is found, the API runs in clearly-labelled DEMO
  mode with a deterministic sample prediction so the full UI can be explored.
"""

import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import train as training

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "sugarcane_model.keras",
    BASE_DIR / "models" / "best_sugarcane_model.keras",
]
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
DISEASE_INFO_PATH = BASE_DIR / "data" / "disease_info.json"
STATIC_DIR = BASE_DIR / "static"
IMG_SIZE = 224
MAX_UPLOAD_MB = 10

# Order matches the notebook's class_folders dict (12 classes).
DEFAULT_CLASS_NAMES = [
    "Mosaic",
    "Smut",
    "Yellow Leaf",
    "Sett Rot",
    "Pokkah Boeng",
    "Banded Chlorosis",
    "Brown Spot",
    "Brown Rust",
    "Grassy Shoot",
    "Viral Disease",
    "Healthy Leaves",
    "Dried Leaves",
]

app = FastAPI(title="SugarcaneGuard AI", version="2.1.0")

class_names = list(DEFAULT_CLASS_NAMES)
disease_info: dict = {}
_model = None
_model_file: Path | None = None


def _resolve_model_file() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _load_metadata():
    global disease_info
    if CLASS_NAMES_PATH.is_file():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and names:
                globals()["class_names"] = names
        except (json.JSONDecodeError, OSError):
            pass
    if DISEASE_INFO_PATH.is_file():
        with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            disease_info = json.load(f)


def _load_real_model():
    """Import tensorflow and load the model only when a file exists."""
    global _model, _model_file
    if _model is not None:
        return _model
    model_file = _resolve_model_file()
    if model_file is None:
        raise RuntimeError("No trained model file found.")
    import tensorflow as tf  # lazy import: demo mode needs no tensorflow
    _model = tf.keras.models.load_model(str(model_file))
    _model_file = model_file
    return _model


def is_real_mode() -> bool:
    return _resolve_model_file() is not None


def invalidate_model_cache():
    """Called after a fresh training run so predictions use the new model."""
    global _model, _model_file
    _model = None
    _model_file = None


# ------------------------------------------------------------- prediction ----

def _prepare_image(data: bytes) -> np.ndarray:
    """
    Same preprocessing the notebook's predict_disease() uses: resize to
    224x224, float32 RGB in [0, 255]. Do NOT apply mobilenet_v2.preprocess_input
    here - it is already a layer inside the saved model, so applying it again
    would corrupt every prediction (double preprocessing bug).
    """
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def _predict_real(data: bytes) -> np.ndarray:
    model = _load_real_model()
    return model.predict(_prepare_image(data), verbose=0)[0].astype(float)


def _predict_demo(data: bytes) -> np.ndarray:
    """
    Deterministic demo prediction (NO model). The image hash picks the class so
    the same image always returns the same demo result. Clearly labelled.
    """
    digest = int(hashlib.md5(data).hexdigest(), 16)
    n = len(class_names)
    top_idx = digest % n
    rng = np.random.default_rng(digest % (2**32))
    winner_conf = float(rng.uniform(0.62, 0.94))
    rest = rng.dirichlet(np.ones(n - 1)) * (1.0 - winner_conf)
    probs = np.zeros(n)
    probs[top_idx] = winner_conf
    probs[[i for i in range(n) if i != top_idx]] = rest
    return probs


def _result_payload(probs: np.ndarray, mode: str) -> dict:
    order = np.argsort(probs)[::-1]
    top3 = [
        {
            "class": class_names[int(i)],
            "confidence": round(float(probs[int(i)]), 4),
            "info": disease_info.get(class_names[int(i)], {}),
        }
        for i in order[:3]
    ]
    best = top3[0]
    return {
        "mode": mode,
        "predicted_class": best["class"],
        "confidence": best["confidence"],
        "top3": top3,
        "model_file": _model_file.name if _model_file else None,
        "note": (
            None
            if mode == "real"
            else "Demo mode: no trained model found. Copy sugarcane_model.keras "
                 "(or best_sugarcane_model.keras) and class_names.json into the "
                 "models/ folder for real predictions."
        ),
    }


# ------------------------------------------------------------------ routes ----

@app.on_event("startup")
def startup():
    _load_metadata()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "mode": "real" if is_real_mode() else "demo",
        "model_file": (_resolve_model_file().name if is_real_mode() else None),
        "classes": class_names,
    }


@app.get("/api/classes")
def classes():
    # Guide content covers every known class (12), even while the loaded
    # model may still be the older 7-class one (until retrained).
    names = list(DEFAULT_CLASS_NAMES)
    for name in disease_info:
        if name not in names:
            names.append(name)
    return {name: disease_info.get(name, {}) for name in names}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"Image is larger than {MAX_UPLOAD_MB} MB."
        )
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file.")

    t0 = time.perf_counter()
    if is_real_mode():
        payload = _result_payload(_predict_real(data), mode="real")
    else:
        payload = _result_payload(_predict_demo(data), mode="demo")
    payload["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse(payload)


# ------------------------------------------------------- training API ----

@app.post("/api/train/start")
async def train_start(
    folder_path: str = Form(default=""),
    epochs: int = Form(default=20),
    dataset_zip: UploadFile = File(default=None),
):
    """
    Start a training job on the REAL dataset. Provide either:
      - folder_path: dataset folder on this machine (fastest, no copy), or
      - dataset_zip: a .zip upload of the dataset archive.
    """
    if not 1 <= epochs <= 100:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100.")
    has_folder = bool(folder_path and folder_path.strip())
    has_zip = dataset_zip is not None and bool(dataset_zip.filename)
    if has_folder == has_zip:
        raise HTTPException(
            status_code=400,
            detail="Provide either a dataset folder path OR a zip upload, not both.",
        )
    try:
        if has_zip:
            if not dataset_zip.filename.lower().endswith(".zip"):
                raise HTTPException(status_code=400, detail="Please upload a .zip file.")
            training.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            zip_path = training.UPLOAD_DIR / f"upload-{uuid.uuid4().hex[:8]}.zip"
            with open(zip_path, "wb") as f:
                while chunk := await dataset_zip.read(1024 * 1024):
                    f.write(chunk)
            source = {"type": "zip", "path": str(zip_path)}
        else:
            source = {"type": "folder", "path": folder_path.strip()}
        return training.start_job(source, epochs=epochs)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/train/status")
def train_status():
    """Live training progress: state, epoch, metrics, timing, ETA, log."""
    return training.get_status()


@app.get("/api/train/report")
def train_report():
    """Final training report (timing + metrics), if a run has completed."""
    if training.REPORT_PATH.is_file():
        with open(training.REPORT_PATH, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    raise HTTPException(status_code=404, detail="No training report yet.")


# ------------------------------------------------------- static frontend ----

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/download")
def download_page():
    """Professional download page for the complete project package."""
    return FileResponse(str(STATIC_DIR / "download.html"))


@app.get("/train")
def train_page():
    """Train AI page: train the model on the real dataset with live progress."""
    return FileResponse(str(STATIC_DIR / "train.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
