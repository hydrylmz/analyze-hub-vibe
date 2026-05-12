# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json
import pickle
import os
from typing import Any

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

app = FastAPI(title="Lumen AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (lokal, tek session)
# ---------------------------------------------------------------------------
STATE: dict[str, Any] = {
    "df": None,            # tam dataframe
    "model": None,         # eğitilmiş model
    "model_cols": None,    # one-hot sonrası kolon listesi
    "target": None,        # hedef kolon adı
    "task": None,          # "regression" | "classification"
    "label_enc": None,     # classification için LabelEncoder
    "metrics": None,       # eğitim metrikleri
    "history": [],         # tahmin geçmişi
    "pred_counter": 0,     # tahmin sayacı
}


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def detect_task(series: pd.Series) -> str:
    """Hedef değişken sayısal ve unique sayısı fazlaysa regression, yoksa classification."""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() > 10:
            return "regression"
    return "classification"


def prepare_X(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X)
    return X, y


def feature_importance(model, cols: list[str], top_n: int = 5) -> dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}
    imp = model.feature_importances_
    pairs = sorted(zip(cols, imp), key=lambda x: x[1], reverse=True)[:top_n]
    return {k: round(float(v), 4) for k, v in pairs}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """CSV yükle, parse et, state'e kaydet."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse hatası: {e}")

    STATE["df"] = df
    STATE["model"] = None
    STATE["model_cols"] = None
    STATE["target"] = None
    STATE["task"] = None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    missing = df.isnull().sum().to_dict()
    total_cells = df.shape[0] * df.shape[1]
    total_missing = sum(missing.values())

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_pct": round(total_missing / total_cells * 100, 2) if total_cells else 0,
        "preview": df.head(6).fillna("").to_dict(orient="records"),
    }


@app.get("/columns")
def columns():
    """Yüklü CSV'nin kolonlarını döndür."""
    if STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Önce CSV yükle.")
    df: pd.DataFrame = STATE["df"]
    return {
        "columns": list(df.columns),
        "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(exclude="number").columns.tolist(),
    }


class TrainRequest(BaseModel):
    target: str
    exclude: list[str] = []


@app.post("/train")
def train(req: TrainRequest):
    """Hedef kolona göre model eğit."""
    if STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Önce CSV yükle.")

    df: pd.DataFrame = STATE["df"].copy()

    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"'{req.target}' kolonu bulunamadı.")

    # Kullanıcının exclude ettiği kolonları düşür
    if req.exclude:
        df = df.drop(columns=[c for c in req.exclude if c in df.columns], errors="ignore")

    # Eksik değerleri basitçe doldur
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

    task = detect_task(df[req.target])
    STATE["task"] = task
    STATE["target"] = req.target

    X, y = prepare_X(df, req.target)
    STATE["model_cols"] = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task == "regression":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
        }
        STATE["label_enc"] = None

    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        model = XGBClassifier(n_estimators=100, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="logloss",
                              random_state=42)
        model.fit(X_train, y_enc)
        preds_enc = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test_enc, preds_enc)), 4),
            "classes": list(le.classes_.astype(str)),
        }
        STATE["label_enc"] = le

    STATE["model"] = model
    STATE["metrics"] = metrics

    # Modeli diske kaydet
    os.makedirs("model_cache", exist_ok=True)
    with open("model_cache/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model_cache/meta.json", "w") as f:
        json.dump({
            "target": req.target,
            "task": task,
            "model_cols": list(X.columns),
        }, f)

    return {
        "task": task,
        "target": req.target,
        "metrics": metrics,
        "feature_importance": feature_importance(model, list(X.columns)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


@app.post("/predict")
def predict(data: dict):
    """Tek satır tahmin."""
    if STATE["model"] is None:
        raise HTTPException(status_code=400, detail="Önce modeli eğit (/train).")

    model = STATE["model"]
    model_cols: list[str] = STATE["model_cols"]
    task: str = STATE["task"]
    le: LabelEncoder | None = STATE["label_enc"]

    try:
        row = pd.DataFrame([data])
        row = pd.get_dummies(row)
        row = row.reindex(columns=model_cols, fill_value=0)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Veri hazırlama hatası: {e}")

    raw = model.predict(row)[0]

    from datetime import datetime
    STATE["pred_counter"] += 1
    pred_id = f"p_{STATE['pred_counter']:04d}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if task == "regression":
        result = {
            "prediction": round(float(raw), 4),
            "confidence": None,
            "feature_importance": feature_importance(model, model_cols),
        }
    else:
        proba = model.predict_proba(row)[0]
        confidence = round(float(proba.max()), 4)
        label = le.inverse_transform([int(raw)])[0] if le else str(raw)
        result = {
            "prediction": str(label),
            "confidence": confidence,
            "feature_importance": feature_importance(model, model_cols),
        }

    # Geçmişe ekle (son 100 tahmin)
    STATE["history"].insert(0, {
        "id": pred_id,
        "at": timestamp,
        "model": f"{task}-v{STATE['pred_counter']}",
        "target": STATE["target"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "inputs": data,
    })
    STATE["history"] = STATE["history"][:100]

    return result


@app.get("/metrics")
def metrics():
    """Son eğitim metriklerini döndür."""
    if STATE["metrics"] is None:
        raise HTTPException(status_code=400, detail="Henüz model eğitilmedi.")
    return {
        "task": STATE["task"],
        "target": STATE["target"],
        "metrics": STATE["metrics"],
        "feature_importance": feature_importance(STATE["model"], STATE["model_cols"]),
    }


@app.get("/results")
def results():
    """Tahmin geçmişi + model metrikleri."""
    if not STATE["history"] and STATE["metrics"] is None:
        raise HTTPException(status_code=400, detail="Henüz tahmin yapılmadı.")

    # Confidence dağılımı (5 bucket)
    confidences = [h["confidence"] for h in STATE["history"] if h["confidence"] is not None]
    buckets = [{"label": f"{i*20}–{(i+1)*20}%", "count": 0} for i in range(5)]
    for c in confidences:
        idx = min(int(c * 5), 4)
        buckets[idx]["count"] += 1

    return {
        "total": len(STATE["history"]),
        "history": STATE["history"],
        "metrics": STATE["metrics"],
        "task": STATE["task"],
        "target": STATE["target"],
        "confidence_buckets": buckets,
    }


@app.delete("/results")
def clear_results():
    """Tahmin geçmişini temizle."""
    STATE["history"] = []
    STATE["pred_counter"] = 0
    return {"cleared": True}