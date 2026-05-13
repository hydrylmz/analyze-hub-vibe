# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# Lumen AI Backend — API layer only.
#
# Business logic lives in:
#   preprocessing.py  — skew detection, outlier removal, ordinal mapping
#   encoders.py       — feature encoding, predict-time transforms, SHAP
#   training.py       — Optuna HPO, model builders, run_training()
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from encoders import predict_batch_fast, predict_row_fast, _TARGET_ENC_AVAILABLE, _SHAP_AVAILABLE
from training import run_training

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lumen")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Lumen AI Backend v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe global state
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

STATE: dict[str, Any] = {
    # data
    "df": None,
    # model artifacts
    "model": None,
    "model_cols": None,
    "target": None,
    "task": None,
    "label_enc": None,
    "ord_enc": None,
    "target_enc": None,
    "high_card_cols": None,
    "low_card_cols": None,
    "cat_cols": None,
    "num_cols": None,
    "num_imputer": None,
    "metrics": None,
    "best_params": None,
    "cv_scores": None,
    # target transform (log1p for skewed regression targets)
    "target_transform": "none",
    "inverse_fn": None,
    "ordinal_mappings": None,
    # training state
    "train_status": "idle",
    "train_error": None,
    "train_started_at": None,
    "optuna_study": None,
    # prediction history (ring buffer)
    "history": deque(maxlen=500),
    "pred_counter": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "shap": _SHAP_AVAILABLE,
        "target_encoder": _TARGET_ENC_AVAILABLE,
    }


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    with _lock:
        STATE["df"] = df
        STATE["model"] = None
        STATE["train_status"] = "idle"

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    missing  = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]

    return {
        "rows":             len(df),
        "columns":          list(df.columns),
        "numeric_cols":     num_cols,
        "categorical_cols": cat_cols,
        "missing_pct":      round(missing.sum() / total_cells * 100, 2) if total_cells else 0,
        "missing_by_col":   {k: int(v) for k, v in missing.items() if v > 0},
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "preview":          df.head(6).fillna("").to_dict(orient="records"),
    }


@app.get("/columns")
def columns():
    with _lock:
        df = STATE["df"]
    if df is None:
        raise HTTPException(400, "Upload a CSV first.")
    return {
        "columns":          list(df.columns),
        "numeric_cols":     df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude="number").columns.tolist(),
    }


# ── Train (async) ─────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    target:     str
    exclude:    list[str] = []
    n_trials:   int = 50
    time_limit: int = 300


@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    with _lock:
        if STATE["df"] is None:
            raise HTTPException(400, "Upload a CSV first.")
        if STATE["train_status"] == "running":
            raise HTTPException(409, "Training already in progress. Cancel first.")
        STATE["train_status"] = "running"

    background_tasks.add_task(
        run_training,
        req.target, req.exclude, req.n_trials, req.time_limit,
        STATE, _lock,
    )
    return {
        "message":      "Training started in background.",
        "poll":         "/train/status",
        "n_trials":     req.n_trials,
        "time_limit_s": req.time_limit,
    }


@app.get("/train/status")
def train_status():
    with _lock:
        status      = STATE["train_status"]
        error       = STATE["train_error"]
        started     = STATE["train_started_at"]
        metrics     = STATE["metrics"]
        best_params = STATE["best_params"]
        study       = STATE["optuna_study"]

    n_trials_done = len(study.trials) if study else 0
    resp: dict = {
        "status":           status,
        "started_at":       started,
        "trials_completed": n_trials_done,
    }
    if status == "done":
        resp["metrics"]     = metrics
        resp["best_params"] = best_params
    if status == "error":
        resp["error"] = error
    return resp


@app.post("/train/cancel")
def train_cancel():
    with _lock:
        if STATE["train_status"] != "running":
            raise HTTPException(400, "No training is running.")
        STATE["train_status"] = "cancelled"
    return {"message": "Cancellation requested. Training will stop after current trial."}


# ── Predict (single) ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: dict):
    with _lock:
        model        = STATE["model"]
        task         = STATE["task"]
        le           = STATE["label_enc"]
        target       = STATE["target"]
        inverse_fn   = STATE["inverse_fn"]
        pred_counter = STATE["pred_counter"] + 1
        STATE["pred_counter"] = pred_counter
        feat_imp     = STATE.get("feat_imp_cache", {})

    if model is None:
        raise HTTPException(400, "Train a model first (/train).")

    try:
        X = predict_row_fast(data, STATE)
    except Exception as e:
        raise HTTPException(422, f"Feature prep error: {e}")

    raw = model.predict(X)[0]
    pred_id = f"p_{pred_counter:05d}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if task == "regression":
        value = float(raw)
        if inverse_fn is not None:
            value = float(inverse_fn(np.array([value]))[0])
        result = {
            "prediction":         round(value, 4),
            "confidence":         None,
            "feature_importance": feat_imp,
        }
    else:
        proba  = model.predict_proba(X)[0]
        conf   = round(float(proba.max()), 4)
        label  = le.inverse_transform([int(raw)])[0] if le else str(int(raw))
        result = {
            "prediction":         str(label),
            "confidence":         conf,
            "feature_importance": feat_imp,
        }

    with _lock:
        STATE["history"].appendleft({
            "id":         pred_id,
            "at":         ts,
            "target":     target,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "inputs":     data,
        })

    return result


# ── Predict (batch) ───────────────────────────────────────────────────────────
class BatchRequest(BaseModel):
    rows: list[dict]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.rows:
        raise HTTPException(422, "Empty rows list.")

    with _lock:
        model      = STATE["model"]
        task       = STATE["task"]
        le         = STATE["label_enc"]
        inverse_fn = STATE["inverse_fn"]

    if model is None:
        raise HTTPException(400, "Train a model first.")

    try:
        X = predict_batch_fast(req.rows, STATE)
    except Exception as e:
        raise HTTPException(422, f"Feature prep error: {e}")

    raws = model.predict(X)

    results = []
    if task == "regression":
        for r in raws:
            value = float(r)
            if inverse_fn is not None:
                value = float(inverse_fn(np.array([value]))[0])
            results.append({"prediction": round(value, 4), "confidence": None})
    else:
        probas = model.predict_proba(X)
        for r, p in zip(raws, probas):
            label = le.inverse_transform([int(r)])[0] if le else str(int(r))
            results.append({
                "prediction": str(label),
                "confidence": round(float(p.max()), 4),
            })

    return {"count": len(results), "predictions": results}


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    with _lock:
        if STATE["metrics"] is None:
            raise HTTPException(400, "No model trained yet.")
        return {
            "task":               STATE["task"],
            "target":             STATE["target"],
            "model_type":         type(STATE["model"]).__name__,
            "metrics":            STATE["metrics"],
            "best_params":        STATE["best_params"],
            "cv_scores":          STATE["cv_scores"],
            "feature_importance": STATE.get("feat_imp_cache", {}),
            "target_transform":   STATE["target_transform"],
        }


# ── Results / history ─────────────────────────────────────────────────────────
@app.get("/results")
def results():
    with _lock:
        history  = list(STATE["history"])
        metrics_ = STATE["metrics"]
        task     = STATE["task"]
        target   = STATE["target"]

    if not history and metrics_ is None:
        raise HTTPException(400, "No predictions yet.")

    confidences = [h["confidence"] for h in history if h["confidence"] is not None]
    buckets = [{"label": f"{i*20}–{(i+1)*20}%", "count": 0} for i in range(5)]
    for c in confidences:
        buckets[min(int(c * 5), 4)]["count"] += 1

    return {
        "total":              len(history),
        "history":            history,
        "metrics":            metrics_,
        "task":               task,
        "target":             target,
        "confidence_buckets": buckets,
    }


@app.delete("/results")
def clear_results():
    with _lock:
        STATE["history"].clear()
        STATE["pred_counter"] = 0
    return {"cleared": True}


# ── Model info ────────────────────────────────────────────────────────────────
@app.get("/model/info")
def model_info():
    with _lock:
        if STATE["model"] is None:
            raise HTTPException(400, "No model trained yet.")
        return {
            "model_type":           type(STATE["model"]).__name__,
            "task":                 STATE["task"],
            "target":               STATE["target"],
            "n_features":           len(STATE["model_cols"]),
            "feature_names":        STATE["model_cols"],
            "high_card_cols":       STATE["high_card_cols"],
            "low_card_cols":        STATE["low_card_cols"],
            "target_enc_active":    STATE["target_enc"] is not None,
            "best_params":          STATE["best_params"],
            "shap_available":       _SHAP_AVAILABLE,
            "target_enc_available": _TARGET_ENC_AVAILABLE,
            "target_transform":     STATE["target_transform"],
            "date_cols_parsed":     STATE.get("date_cols"),
            "frequent_categories":  STATE.get("frequent_cats"),
        }