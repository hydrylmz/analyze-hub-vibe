# backend/training.py
# ─────────────────────────────────────────────────────────────────────────────
# Background training logic: Optuna HPO + final model fit.
# Imports preprocessing and encoders; updates shared STATE dict.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from encoders import encode_features, shap_importance
from preprocessing import (
    apply_ordinal_mappings,
    apply_target_transform,
    remove_outliers,
    parse_dates,
    bucket_rare_categories,
    transform_skewed_features,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("lumen.training")


# ─────────────────────────────────────────────────────────────────────────────
# Task detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_task(series: pd.Series, threshold: int = 20) -> str:
    """
    Regression if numeric dtype, more than `threshold` unique values,
    and unique-to-total ratio > 5%.  Otherwise classification.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    ratio = series.nunique() / max(len(series), 1)
    if series.nunique() > threshold and ratio > 0.05:
        return "regression"
    return "classification"


# ─────────────────────────────────────────────────────────────────────────────
# Model builders (LightGBM + XGBoost as Optuna candidates)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_for_trial(trial: optuna.Trial, task: str, n_classes: int):
    model_type = trial.suggest_categorical("model_type", ["lgbm", "xgb"])

    if model_type == "lgbm":
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "boosting_type":     trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "verbosity":         -1,
            "n_jobs":            -1,
            "random_state":      42,
        }
        if task == "regression":
            return lgb.LGBMRegressor(**params)
        obj = "binary" if n_classes == 2 else "multiclass"
        extra = {"num_class": n_classes} if n_classes > 2 else {}
        return lgb.LGBMClassifier(**params, objective=obj, **extra)

    else:  # xgb
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method":      "hist",
            "random_state":     42,
            "verbosity":        0,
            "n_jobs":           -1,
        }
        if task == "regression":
            return XGBRegressor(**params)
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        extra = {"num_class": n_classes} if n_classes > 2 else {}
        return XGBClassifier(
            **params, objective=obj, eval_metric="logloss",
            use_label_encoder=False, **extra,
        )


def make_hpo_pipeline(model, num_cols: list[str]) -> Pipeline:
    """
    Wraps model in a Pipeline with a fast SimpleImputer for HPO trials.
    Prevents MICE leakage across CV folds during Optuna search.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main training function (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    req_target: str,
    req_exclude: list[str],
    n_trials: int,
    time_limit: int,
    state: dict[str, Any],
    lock: threading.Lock,
) -> None:
    try:
        with lock:
            state["train_status"] = "running"
            state["train_error"]  = None
            state["train_started_at"] = datetime.now().isoformat()

        with lock:
            df: pd.DataFrame = state["df"].copy()

        df = df.drop(columns=[c for c in req_exclude if c in df.columns], errors="ignore")

        if req_target not in df.columns:
            raise ValueError(f"'{req_target}' column not found.")

        # ── 1. Parse Dates ───────────────────────────────────────────────────
        df, date_cols = parse_dates(df)

        # ── 2. Ordinal quality-scale mapping ─────────────────────────────────
        df, ordinal_mappings = apply_ordinal_mappings(df, exclude=[req_target])

        # ── 3. Rare Category Bucketing ───────────────────────────────────────
        df, frequent_cats = bucket_rare_categories(df, exclude=[req_target])

        # ── 4. Skew Transform ────────────────────────────────────────────────
        # Disabled: mathematically incorrect/harmful for Tree models (LGBM/XGBoost)
        # df, skewed_cols = transform_skewed_features(df, exclude=[req_target])

        # ── 5. Task detection ────────────────────────────────────────────────
        task = detect_task(df[req_target])

        # ── 6. Outlier removal (regression only; conservative IQR×3) ─────────
        if task == "regression":
            df = remove_outliers(df, req_target, iqr_multiplier=3.0, max_removal_pct=0.05)

        # ── 7. Target encoding ───────────────────────────────────────────────
        le: LabelEncoder | None = None
        target_transform_name = "none"
        inverse_fn = None

        if task == "classification":
            le = LabelEncoder()
            y = le.fit_transform(df[req_target].astype(str))
            n_classes = len(le.classes_)
        else:
            y_raw = df[req_target].values.astype(np.float64)
            y, inverse_fn, target_transform_name = apply_target_transform(y_raw)
            n_classes = 0

        # ── 8. Feature encoding (full pipeline; MICE for final model) ─────────
        (X_arr, col_names, ord_enc, target_enc,
         cat_cols, num_cols, high_card_cols, low_card_cols, num_imputer) = encode_features(
            df, req_target, y=y, fit=True
        )

        log.info(
            f"Training | task={task} rows={len(X_arr)} features={len(col_names)} "
            f"target_transform={target_transform_name} "
            f"high_card_cats={len(high_card_cols or [])} "
            f"low_card_cats={len(low_card_cols or [])}"
        )

        # ── 9. HPO data: raw numerics + already-encoded cats (no MICE leakage) ─
        X_raw_num = df.drop(columns=[req_target]).reindex(columns=num_cols).astype(float).values
        X_hpo = np.hstack([
            X_raw_num,
            X_arr[:, len(num_cols):]
        ]).astype(np.float32)

        # ── 9.5 Subsample for HPO if dataset is very large ────────────────────
        # For datasets > 100k rows, HPO trials are too slow. Subsample for HPO.
        # We still use the full dataset for the final model fit.
        if len(X_hpo) > 100_000:
            log.info(f"Subsampling HPO data from {len(X_hpo)} to 100,000 rows for speed.")
            indices = np.random.choice(len(X_hpo), 100_000, replace=False)
            X_hpo_sub = X_hpo[indices]
            y_hpo_sub = y[indices]
        else:
            X_hpo_sub = X_hpo
            y_hpo_sub = y

        # ── 10. Optuna study ──────────────────────────────────────────────────
        def objective(trial: optuna.Trial) -> float:
            with lock:
                if state["train_status"] == "cancelled":
                    raise optuna.exceptions.TrialPruned()

            model = build_model_for_trial(trial, task, n_classes)
            pipe  = make_hpo_pipeline(model, num_cols)

            if task == "regression":
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    pipe, X_hpo_sub, y_hpo_sub, cv=cv,
                    scoring="neg_mean_absolute_error", n_jobs=1,
                )
                return float(-scores.mean())
            else:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    pipe, X_hpo_sub, y_hpo_sub, cv=cv,
                    scoring="neg_log_loss", n_jobs=1,
                )
                return float(-scores.mean())

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        with lock:
            state["optuna_study"] = study

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=time_limit,
            catch=(Exception,),
            show_progress_bar=False,
        )

        with lock:
            if state["train_status"] == "cancelled":
                log.info("Training cancelled by user.")
                return

        best_params = study.best_params
        log.info(f"Best params: {best_params}  best_value={study.best_value:.6f}")

        # ── 11. Final model (early stopping on 10% holdout) ───────────────────
        best_trial = study.best_trial
        final_model = build_model_for_trial(best_trial, task, n_classes)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_arr, y, test_size=0.1, random_state=42,
            stratify=y if task == "classification" else None,
        )
        sw_tr = compute_sample_weight("balanced", y_tr) if task == "classification" else None

        if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            final_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
                **({"sample_weight": sw_tr} if sw_tr is not None else {}),
            )
        else:
            final_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **({"sample_weight": sw_tr} if sw_tr is not None else {}),
            )

        # ── 12. Evaluation Metrics ────────────────────────────────────────────
        # For large datasets, 5-fold CV on full data is too slow. Use holdout (X_val).
        if len(X_arr) > 50_000:
            log.info(f"Using holdout set (10%) for final metrics (dataset size={len(X_arr)})")
            y_pred = final_model.predict(X_val)
            
            if task == "regression":
                from sklearn.metrics import mean_absolute_error, r2_score
                mae_transformed = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                if inverse_fn is not None:
                    mae_display = round(float(np.expm1(mae_transformed)), 4)
                    mae_label   = "holdout_mae_orig_scale"
                else:
                    mae_display = round(mae_transformed, 4)
                    mae_label   = "holdout_mae"

                metrics = {
                    mae_label:      mae_display,
                    "holdout_r2":   round(float(r2), 4),
                    "target_transform": target_transform_name,
                }
                cv_scores_list = [mae_transformed]
            else:
                from sklearn.metrics import accuracy_score, log_loss
                y_proba = final_model.predict_proba(X_val)
                acc = accuracy_score(y_val, y_pred)
                ll = log_loss(y_val, y_proba)
                
                metrics = {
                    "holdout_accuracy": round(float(acc), 4),
                    "holdout_log_loss": round(float(ll), 4),
                    "classes":          list(le.classes_.astype(str)) if le else [],
                }
                cv_scores_list = [acc]
        else:
            # For small datasets, 5-fold CV is fine and more robust
            from sklearn.model_selection import cross_validate
            pipe_final = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model",   build_model_for_trial(best_trial, task, n_classes)),
            ])
            if task == "regression":
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_results = cross_validate(
                    pipe_final, X_hpo, y, cv=cv,
                    scoring=["neg_mean_absolute_error", "r2"],
                    n_jobs=1
                )
                cv_mae_scores = cv_results["test_neg_mean_absolute_error"]
                cv_r2_scores = cv_results["test_r2"]

                mae_transformed = float(-cv_mae_scores.mean())
                if inverse_fn is not None:
                    mae_display = round(float(np.expm1(mae_transformed)), 4)
                    mae_label   = "cv_mae_orig_scale"
                else:
                    mae_display = round(mae_transformed, 4)
                    mae_label   = "cv_mae"

                metrics = {
                    mae_label:      mae_display,
                    "cv_mae_std":   round(float(cv_mae_scores.std()), 4),
                    "cv_r2":        round(float(cv_r2_scores.mean()), 4),
                    "target_transform": target_transform_name,
                }
                cv_scores_list = (-cv_mae_scores).tolist()

            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_results = cross_validate(
                    pipe_final, X_hpo, y, cv=cv,
                    scoring=["accuracy", "neg_log_loss"],
                    n_jobs=1
                )
                cv_acc = cv_results["test_accuracy"]
                cv_ll  = cv_results["test_neg_log_loss"]
                
                metrics = {
                    "cv_accuracy":     round(float(cv_acc.mean()), 4),
                    "cv_accuracy_std": round(float(cv_acc.std()),  4),
                    "cv_log_loss":     round(float(-cv_ll.mean()), 4),
                    "classes":         list(le.classes_.astype(str)) if le else [],
                }
                cv_scores_list = cv_acc.tolist()


        # ── 13. SHAP importance ───────────────────────────────────────────────
        feat_imp = shap_importance(final_model, X_arr, col_names)

        # ── 14. Persist ───────────────────────────────────────────────────────
        os.makedirs("model_cache", exist_ok=True)
        with open("model_cache/model.pkl", "wb") as f:
            pickle.dump(final_model, f, protocol=5)
        meta = {
            "target":             req_target,
            "task":               task,
            "model_type":         type(final_model).__name__,
            "model_cols":         col_names,
            "best_params":        best_params,
            "metrics":            metrics,
            "target_transform":   target_transform_name,
            "date_cols":          date_cols,
            "frequent_cats":      frequent_cats,
        }
        with open("model_cache/meta.json", "w") as f:
            json.dump(meta, f)

        with lock:
            state.update({
                "model":              final_model,
                "model_cols":         col_names,
                "target":             req_target,
                "task":               task,
                "label_enc":          le,
                "ord_enc":            ord_enc,
                "target_enc":         target_enc,
                "cat_cols":           cat_cols,
                "num_cols":           num_cols,
                "high_card_cols":     high_card_cols,
                "low_card_cols":      low_card_cols,
                "num_imputer":        num_imputer,
                "metrics":            metrics,
                "best_params":        best_params,
                "cv_scores":          cv_scores_list,
                "train_status":       "done",
                "feat_imp_cache":     feat_imp,
                "target_transform":   target_transform_name,
                "inverse_fn":         inverse_fn,
                "ordinal_mappings":   ordinal_mappings,
                "date_cols":          date_cols,
                "frequent_cats":      frequent_cats,
            })

        log.info("Training complete.")

    except Exception as exc:
        log.exception("Training failed")
        with lock:
            state["train_status"] = "error"
            state["train_error"]  = str(exc)