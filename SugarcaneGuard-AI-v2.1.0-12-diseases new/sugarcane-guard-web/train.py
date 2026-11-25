"""
SugarcaneGuard AI - training engine.

Trains the MobileNetV2 model exactly like
SugarcaneGuard_AI_Jupyter_Updated_Path.ipynb, but as a background job whose
progress (epoch, metrics, timing, ETA) the website streams live to the
"Train AI" page.

Two dataset sources are supported:
  1. folder_path: an existing folder on THIS machine, e.g.
     D:\\OneDrive\\Desktop\\sugarcane leaf decise detection Final\\archive
     (fastest - no copying needed)
  2. zip upload: a .zip of that same archive folder, extracted server-side.

Expected layout inside the dataset root (same as the notebook):
    Diseases/Diseases/{Mosaic, smut, Yellow Leaf, Sett Rot, Pokkah Boeng,
                        Banded Chlorosis, Brown Spot, BrownRust, Grassy shoot,
                        Viral Disease}/
    Healthy Leaves/Healthy Leaves/
    Dried Leaves/Dried Leaves/

On success it writes:
    models/sugarcane_model.keras   (final model)
    models/best_sugarcane_model.keras (best checkpoint)
    models/class_names.json
    models/training_report.json    (timing + metrics)
and the prediction API picks the new model up automatically.
"""

import json
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
REPORT_PATH = MODELS_DIR / "training_report.json"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

CLASS_SUBFOLDERS = {
    "Mosaic": ("Diseases", "Diseases", "Mosaic"),
    "Smut": ("Diseases", "Diseases", "smut"),
    "Yellow Leaf": ("Diseases", "Diseases", "Yellow Leaf"),
    "Sett Rot": ("Diseases", "Diseases", "Sett Rot"),
    "Pokkah Boeng": ("Diseases", "Diseases", "Pokkah Boeng"),
    "Banded Chlorosis": ("Diseases", "Diseases", "Banded Chlorosis"),
    "Brown Spot": ("Diseases", "Diseases", "Brown Spot"),
    "Brown Rust": ("Diseases", "Diseases", "BrownRust"),
    "Grassy Shoot": ("Diseases", "Diseases", "Grassy shoot"),
    "Viral Disease": ("Diseases", "Diseases", "Viral Disease"),
    "Healthy Leaves": ("Healthy Leaves", "Healthy Leaves"),
    "Dried Leaves": ("Dried Leaves", "Dried Leaves"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

_lock = threading.Lock()
_state = {
    "state": "idle",          # idle | preparing | training | evaluating | done | error
    "job_id": None,
    "source": None,
    "epochs_total": 0,
    "epoch": 0,
    "started_at": None,
    "elapsed_sec": 0.0,
    "epoch_times": [],        # seconds per finished epoch
    "eta_sec": None,
    "train_loss": None,
    "train_acc": None,
    "val_loss": None,
    "val_acc": None,
    "test_loss": None,
    "test_acc": None,
    "images_found": 0,
    "log": [],                # recent human-readable lines (capped)
    "error": None,
    "finished_at": None,
    "total_sec": None,
}


def get_status() -> dict:
    with _lock:
        return dict(_state)


def is_running() -> bool:
    with _lock:
        return _state["state"] in ("preparing", "training", "evaluating")


def _set(**kwargs):
    with _lock:
        _state.update(kwargs)


def _log(msg: str):
    with _lock:
        _state["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        _state["log"] = _state["log"][-120:]


def _fail(msg: str):
    _log("ERROR: " + msg)
    _set(state="error", error=msg, finished_at=datetime.now().isoformat())


def start_job(source: dict, epochs: int = 20) -> dict:
    """Start training in a background thread. source: {"type": "folder"|"zip", "path": str}."""
    with _lock:
        if _state["state"] in ("preparing", "training", "evaluating"):
            raise RuntimeError("A training job is already running.")
        job_id = f"train-{int(time.time())}"
        _state.update(
            {
                "state": "preparing",
                "job_id": job_id,
                "source": source,
                "epochs_total": epochs,
                "epoch": 0,
                "started_at": datetime.now().isoformat(),
                "elapsed_sec": 0.0,
                "epoch_times": [],
                "eta_sec": None,
                "train_loss": None, "train_acc": None,
                "val_loss": None, "val_acc": None,
                "test_loss": None, "test_acc": None,
                "images_found": 0,
                "log": [],
                "error": None,
                "finished_at": None,
                "total_sec": None,
            }
        )
    thread = threading.Thread(target=_run_training, args=(source, epochs, job_id), daemon=True)
    thread.start()
    return {"job_id": job_id, "state": "preparing"}


# ------------------------------------------------------------ the job ----

def _run_training(source: dict, epochs: int, job_id: str):
    t_start = time.perf_counter()
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        _fail(f"Required package missing: {exc}. Run: pip install tensorflow scikit-learn")
        return

    # ---- 1. locate / extract dataset ----
    if source["type"] == "zip":
        zip_path = Path(source["path"])
        if not zip_path.is_file():
            _fail(f"Uploaded zip not found: {zip_path.name}")
            return
        extract_dir = UPLOAD_DIR / f"dataset-{job_id}"
        _log(f"Extracting {zip_path.name} ...")
        _set(state="preparing")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            _fail(f"Could not extract zip: {exc}")
            return
        # handle a single top-level wrapper folder (e.g. archive/archive/...)
        dataset_root = _find_dataset_root(extract_dir)
        _log(f"Extracted to {dataset_root}")
    else:
        dataset_root = Path(source["path"])
        if not dataset_root.is_dir():
            _fail(f"Dataset folder not found: {dataset_root}")
            return
        _log(f"Using dataset folder: {dataset_root}")

    # ---- 2. collect images (notebook cells 2-4) ----
    _log("Scanning image folders ...")
    records = []
    for label, parts in CLASS_SUBFOLDERS.items():
        folder = dataset_root.joinpath(*parts)
        if not folder.is_dir():
            _fail(f"Folder for '{label}' not found: {folder}")
            return
        files = [p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            _fail(f"No images found in folder for '{label}': {folder}")
            return
        records += [(str(p), label) for p in files]
        _log(f"  {label}: {len(files)} images")
    _set(images_found=len(records))
    _log(f"Total images: {len(records)}")

    try:
        import pandas as pd
        df = pd.DataFrame(records, columns=["image", "label"])
        class_names = list(CLASS_SUBFOLDERS.keys())
        label_to_id = {name: i for i, name in enumerate(class_names)}
        df["label_id"] = df["label"].map(label_to_id)
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label"], random_state=SEED)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
        _log(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    except Exception as exc:
        _fail(f"Dataset split failed: {exc}")
        return

    # ---- 3. tf.data pipelines (notebook cell 5) ----
    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return tf.cast(image, tf.float32), tf.cast(label, tf.int32)

    def make_dataset(frame, training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (frame["image"].values, frame["label_id"].values))
        if training:
            ds = ds.shuffle(len(frame), seed=SEED)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    # ---- 4. model (notebook cell 6) ----
    _log("Building MobileNetV2 model ...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 5. train with live progress (notebook cell 7) ----
    epoch_start = [None]

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            _set(state="training")
            _log(f"Training started: {epochs} epochs, batch size {BATCH_SIZE}")

        def on_epoch_begin(self, epoch, logs=None):
            epoch_start[0] = time.perf_counter()
            _set(epoch=epoch + 1)
            _log(f"Epoch {epoch + 1}/{epochs} started ...")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            dur = time.perf_counter() - (epoch_start[0] or time.perf_counter())
            with _lock:
                _state["epoch_times"].append(dur)
                times = list(_state["epoch_times"])
            avg = sum(times) / len(times)
            remaining = epochs - (epoch + 1)
            eta = avg * remaining if remaining > 0 else 0
            elapsed = time.perf_counter() - t_start
            _set(
                epoch=epoch + 1,
                train_loss=round(float(logs.get("loss", 0)), 4),
                train_acc=round(float(logs.get("accuracy", 0)), 4),
                val_loss=round(float(logs.get("val_loss", 0)), 4),
                val_acc=round(float(logs.get("val_accuracy", 0)), 4),
                elapsed_sec=round(elapsed, 1),
                eta_sec=round(eta, 1),
            )
            _log(
                f"Epoch {epoch + 1}/{epochs} done in {dur:.1f}s - "
                f"loss: {logs.get('loss', 0):.4f} - acc: {logs.get('accuracy', 0):.4f} - "
                f"val_loss: {logs.get('val_loss', 0):.4f} - val_acc: {logs.get('val_accuracy', 0):.4f}"
            )

    callbacks = [
        ProgressCallback(),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "best_sugarcane_model.keras"),
            monitor="val_loss", save_best_only=True),
    ]

    _log("Training ... (this is the slow part; progress appears above)")
    try:
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  callbacks=callbacks, verbose=0)
    except Exception as exc:
        _fail(f"Training failed: {exc}\n{traceback.format_exc(limit=3)}")
        return

    # ---- 6. evaluate (notebook cell 9) ----
    _set(state="evaluating")
    _log("Evaluating on the held-out test set ...")
    try:
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        _set(test_loss=round(float(test_loss), 4), test_acc=round(float(test_acc), 4))
        _log(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")
    except Exception as exc:
        _log(f"Evaluation skipped: {exc}")

    # ---- 7. save (notebook cell 10) ----
    _log("Saving model ...")
    try:
        model.save(str(MODELS_DIR / "sugarcane_model.keras"))
        with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
        total_sec = time.perf_counter() - t_start
        report = {
            "job_id": job_id,
            "dataset": str(dataset_root),
            "images": len(records),
            "epochs_requested": epochs,
            "epochs_run": len(_state["epoch_times"]),
            "started_at": _state["started_at"],
            "finished_at": datetime.now().isoformat(),
            "total_seconds": round(total_sec, 1),
            "avg_epoch_seconds": round(
                sum(_state["epoch_times"]) / max(len(_state["epoch_times"]), 1), 1),
            "test_loss": _state["test_loss"],
            "test_accuracy": _state["test_acc"],
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        # invalidate any previously loaded model so /api/predict picks up the new one
        try:
            import app as app_module
            app_module.invalidate_model_cache()
        except Exception:
            pass
        _set(state="done", finished_at=report["finished_at"], total_sec=report["total_seconds"],
             elapsed_sec=round(total_sec, 1), eta_sec=0)
        _log(f"Done in {total_sec / 60:.1f} minutes. Model is now LIVE for predictions.")
    except Exception as exc:
        _fail(f"Saving model failed: {exc}")


def _find_dataset_root(extract_dir: Path) -> Path:
    """If the zip had one wrapper folder, descend into it."""
    current = extract_dir
    for _ in range(3):
        if (current / "Diseases").is_dir():
            return current
        subdirs = [p for p in current.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            current = subdirs[0]
        else:
            break
    return extract_dir
