# backend/encoders.py
# ─────────────────────────────────────────────────────────────────────────────
# Feature encoding, prediction-time transforms, and SHAP importance.
# Extracted from main.py so training.py and main.py stay focused.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

from preprocessing import (
    _normalise,
    parse_dates,
    bucket_rare_categories,
)
# Optional category_encoders (target encoding)
try:
    from category_encoders import TargetEncoder
    _TARGET_ENC_AVAILABLE = True
except ImportError:
    _TARGET_ENC_AVAILABLE = False

# Optional SHAP (best-effort)
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(
    df: pd.DataFrame,
    target: str,
    y: np.ndarray | None = None,
    ord_enc: OrdinalEncoder | None = None,
    target_enc=None,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    high_card_cols: list[str] | None = None,
    low_card_cols: list[str] | None = None,
    num_imputer: IterativeImputer | None = None,
    fit: bool = True,
    high_card_threshold: int = 20,
) -> tuple:
    """
    Returns:
        (X_array, col_names, ord_enc, target_enc,
         cat_cols, num_cols, high_card_cols, low_card_cols, num_imputer)

    fit=True  → encoders are fitted and returned.
    fit=False → encoders are only transformed (predict path).

    Encoding strategy:
    - Numeric columns: MICE imputation (fit) or transform (predict).
    - High-cardinality categoricals (nunique > threshold): TargetEncoder
      (or OrdinalEncoder fallback if category_encoders not installed).
    - Low-cardinality categoricals: OrdinalEncoder with unknown handling.
    """
    X = df.drop(columns=[target])

    if fit:
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        high_card_cols = [c for c in cat_cols if X[c].nunique() > high_card_threshold]
        low_card_cols  = [c for c in cat_cols if c not in high_card_cols]

    # ── numeric imputation (MICE) ────────────────────────────────────────────
    X_num = X[num_cols].copy() if num_cols else pd.DataFrame(index=X.index)
    if num_cols:
        if fit:
            num_imputer = IterativeImputer(max_iter=5, random_state=42, skip_complete=True)
            X_num_arr = num_imputer.fit_transform(X_num)
        else:
            X_num_arr = num_imputer.transform(X_num)
    else:
        X_num_arr = np.empty((len(X), 0))

    # ── high-cardinality cats → TargetEncoder ────────────────────────────────
    X_hc_arr = np.empty((len(X), 0))
    if high_card_cols:
        X_hc = X[high_card_cols].copy().fillna("__missing__")
        if fit and _TARGET_ENC_AVAILABLE and y is not None:
            target_enc = TargetEncoder(cols=high_card_cols, smoothing=10)
            X_hc_arr = target_enc.fit_transform(X_hc, y).values.astype(np.float32)
        elif target_enc is not None and _TARGET_ENC_AVAILABLE:
            X_hc_arr = target_enc.transform(X_hc).values.astype(np.float32)
        else:
            tmp = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int16
            )
            if fit:
                X_hc_arr = tmp.fit_transform(X_hc).astype(np.float32)
                target_enc = tmp
            else:
                X_hc_arr = (target_enc or tmp).transform(X_hc).astype(np.float32)

    # ── low-cardinality cats → OrdinalEncoder ───────────────────────────────
    X_lc_arr = np.empty((len(X), 0))
    if low_card_cols:
        X_lc = X[low_card_cols].copy().fillna("__missing__")
        if fit:
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int16,
            )
            X_lc_arr = ord_enc.fit_transform(X_lc).astype(np.float32)
        else:
            X_lc_arr = ord_enc.transform(X_lc).astype(np.float32)

    X_arr = np.hstack([X_num_arr, X_hc_arr, X_lc_arr]).astype(np.float32)
    col_names = num_cols + (high_card_cols or []) + (low_card_cols or [])

    return (
        X_arr, col_names, ord_enc, target_enc,
        cat_cols, num_cols, high_card_cols, low_card_cols, num_imputer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction-time feature preparation (single row)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_ordinal_mappings_to_df(df: pd.DataFrame, ordinal_mappings: dict) -> pd.DataFrame:
    """Re-apply training-time ordinal mappings to a prediction DataFrame."""
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x, m=mapping: m.get(_normalise(x), np.nan))
    return df


def predict_row_fast(row_dict: dict, state: dict) -> np.ndarray:
    num_cols         = state["num_cols"]
    high_card_cols   = state["high_card_cols"]
    low_card_cols    = state["low_card_cols"]
    num_imputer      = state["num_imputer"]
    ord_enc          = state["ord_enc"]
    target_enc       = state["target_enc"]
    ordinal_mappings = state.get("ordinal_mappings") or {}
    date_cols        = state.get("date_cols")
    frequent_cats    = state.get("frequent_cats")

    row = pd.DataFrame([row_dict]).replace(["NA", ""], np.nan)
    
    if date_cols:
        row, _ = parse_dates(row, date_cols=date_cols)
        
    row = _apply_ordinal_mappings_to_df(row, ordinal_mappings)
    
    if frequent_cats:
        row, _ = bucket_rare_categories(row, exclude=[], frequent_cats=frequent_cats)

    num_part = row.reindex(columns=num_cols).astype(float)
    num_arr  = num_imputer.transform(num_part) if num_cols else np.empty((1, 0))

    if high_card_cols:
        hc_part = row.reindex(columns=high_card_cols).fillna("__missing__")
        hc_arr  = target_enc.transform(hc_part).values.astype(np.float32)
    else:
        hc_arr = np.empty((1, 0))

    if low_card_cols:
        lc_part = row.reindex(columns=low_card_cols).fillna("__missing__")
        lc_arr  = ord_enc.transform(lc_part).astype(np.float32)
    else:
        lc_arr = np.empty((1, 0))

    return np.hstack([num_arr, hc_arr, lc_arr]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction-time feature preparation (batch)
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch_fast(rows: list[dict], state: dict) -> np.ndarray:
    num_cols         = state["num_cols"]
    high_card_cols   = state["high_card_cols"]
    low_card_cols    = state["low_card_cols"]
    num_imputer      = state["num_imputer"]
    ord_enc          = state["ord_enc"]
    target_enc       = state["target_enc"]
    ordinal_mappings = state.get("ordinal_mappings") or {}
    date_cols        = state.get("date_cols")
    frequent_cats    = state.get("frequent_cats")

    df = pd.DataFrame(rows).replace(["NA", ""], np.nan)
    
    if date_cols:
        df, _ = parse_dates(df, date_cols=date_cols)
        
    df = _apply_ordinal_mappings_to_df(df, ordinal_mappings)
    
    if frequent_cats:
        df, _ = bucket_rare_categories(df, exclude=[], frequent_cats=frequent_cats)

    num_part = df.reindex(columns=num_cols).astype(float)
    num_arr  = num_imputer.transform(num_part) if num_cols else np.empty((len(df), 0))

    if high_card_cols:
        hc_part = df.reindex(columns=high_card_cols).fillna("__missing__")
        hc_arr  = target_enc.transform(hc_part).values.astype(np.float32)
    else:
        hc_arr = np.empty((len(df), 0))

    if low_card_cols:
        lc_part = df.reindex(columns=low_card_cols).fillna("__missing__")
        lc_arr  = ord_enc.transform(lc_part).astype(np.float32)
    else:
        lc_arr = np.empty((len(df), 0))

    return np.hstack([num_arr, hc_arr, lc_arr]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP / fallback feature importance
# ─────────────────────────────────────────────────────────────────────────────

def shap_importance(
    model, X_sample: np.ndarray, col_names: list[str], top_n: int = 10
) -> dict[str, float]:
    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample[:500])
            if isinstance(sv, list):
                sv = np.abs(np.array(sv)).mean(axis=0)
            imp = np.abs(sv).mean(axis=0)
            pairs = sorted(zip(col_names, imp), key=lambda x: x[1], reverse=True)[:top_n]
            return {k: round(float(v), 6) for k, v in pairs}
        except Exception:
            pass

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = sorted(zip(col_names, imp), key=lambda x: x[1], reverse=True)[:top_n]
        return {k: round(float(v), 6) for k, v in pairs}

    return {}