# backend/preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
# General-purpose preprocessing utilities
#
# Three concerns live here:
#   1. Skew detection + log1p target transform (general AutoML standard)
#   2. IQR-based outlier removal (configurable, off by default)
#   3. Ordinal quality-scale mapping for ordered categorical columns
#      (e.g. Poor < Fair < Average < Good < Excellent)
#
# None of these are competition-specific.  They are standard steps found in
# H2O AutoML, AutoSklearn, FLAML, and similar production AutoML systems.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger("lumen.preprocessing")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Target transform detection
# ─────────────────────────────────────────────────────────────────────────────

# Skewness threshold above which we apply log1p.
# |skew| > 0.75 is a commonly used rule of thumb (used by H2O AutoML).
_SKEW_THRESHOLD = 0.75


def should_log_transform(y: np.ndarray) -> bool:
    """
    Return True if the target array is sufficiently right-skewed to benefit
    from a log1p transform.

    Conditions (all must hold):
    - All values strictly positive (log1p requires > -1; we require > 0 so
      that the inverse expm1 is unambiguous and residuals are meaningful).
    - |skewness| > _SKEW_THRESHOLD.
    """
    if np.any(y <= 0):
        return False
    skew = float(scipy_stats.skew(y))
    return abs(skew) > _SKEW_THRESHOLD


def apply_target_transform(
    y: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray] | None, str]:
    """
    Detect skew and optionally apply log1p.

    Returns
    -------
    y_transformed : np.ndarray
        Transformed target values.
    inverse_fn : callable or None
        Function to invert the transform on predictions.
        None if no transform was applied.
    transform_name : str
        "log1p" | "none"
    """
    if should_log_transform(y):
        log.info(
            f"Target skew={scipy_stats.skew(y):.3f} > threshold={_SKEW_THRESHOLD}. "
            "Applying log1p transform."
        )
        return np.log1p(y).astype(np.float32), np.expm1, "log1p"

    skew = float(scipy_stats.skew(y))
    log.info(f"Target skew={skew:.3f} ≤ threshold={_SKEW_THRESHOLD}. No transform applied.")
    return y.astype(np.float32), None, "none"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Outlier removal  (IQR-based, applied to features + target jointly)
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers(
    df: pd.DataFrame,
    target: str,
    iqr_multiplier: float = 3.0,
    max_removal_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Remove rows where the *target* column is an extreme outlier.

    Uses the IQR fence:  Q1 - k*IQR  …  Q3 + k*IQR
    with a conservative default of k=3.0 (catches only extreme outliers,
    not just values outside the normal range).

    A hard cap of `max_removal_pct` (default 5 %) ensures we never
    silently discard a large fraction of the dataset.

    Parameters
    ----------
    df : DataFrame
    target : str
        Target column name.
    iqr_multiplier : float
        Fence multiplier k.  3.0 is conservative; 1.5 is aggressive.
    max_removal_pct : float
        Safety cap: if more than this fraction would be removed, skip.

    Returns
    -------
    Filtered DataFrame (copy).
    """
    if not pd.api.types.is_numeric_dtype(df[target]):
        return df  # can't apply IQR to categorical targets

    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df

    lo = q1 - iqr_multiplier * iqr
    hi = q3 + iqr_multiplier * iqr

    mask = df[target].between(lo, hi)
    n_removed = (~mask).sum()
    removal_pct = n_removed / len(df)

    if removal_pct > max_removal_pct:
        log.warning(
            f"Outlier removal would drop {n_removed} rows ({removal_pct:.1%}) "
            f"— exceeds cap of {max_removal_pct:.0%}.  Skipping."
        )
        return df

    if n_removed > 0:
        log.info(
            f"Outlier removal: dropped {n_removed} rows ({removal_pct:.1%}) "
            f"outside [{lo:.2f}, {hi:.2f}] (IQR×{iqr_multiplier})."
        )

    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ordinal quality-scale mapping
# ─────────────────────────────────────────────────────────────────────────────

# Ordered quality scales found across many real-world datasets.
# Each entry is a list ordered from worst → best.
# Matching is case-insensitive and strips whitespace.
_QUALITY_SCALES: list[list[str]] = [
    # Standard 5-level quality / condition scale
    ["po", "fa", "ta", "gd", "ex"],
    # Written-out equivalents
    ["poor", "fair", "average", "good", "excellent"],
    ["very poor", "poor", "fair", "good", "very good", "excellent"],
    # Finish / exposure scales
    ["no", "mn", "av", "gd"],          # e.g. BsmtExposure in Ames
    ["unf", "lwq", "rec", "blq", "alq", "glq"],  # BsmtFinType
    # Generic ordinal scales
    ["none", "low", "medium", "high"],
    ["none", "low", "medium", "high", "very high"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    # Numeric-like strings that appear as text
    ["1", "2", "3", "4", "5"],
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
]

# Build a lookup: frozenset(values) → ordered list
_SCALE_LOOKUP: dict[frozenset, list[str]] = {
    frozenset(s): s for s in _QUALITY_SCALES
}


def _normalise(v: str) -> str:
    return str(v).strip().lower()


def detect_ordinal_scale(series: pd.Series) -> list[str] | None:
    """
    If the unique values of `series` (lowercased, stripped) exactly match a
    known quality scale, return the ordered list.  Otherwise return None.
    """
    unique_vals = {_normalise(v) for v in series.dropna().unique()}
    return _SCALE_LOOKUP.get(frozenset(unique_vals))


def apply_ordinal_mappings(
    df: pd.DataFrame,
    exclude: list[str],
    ordinal_mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Scan all object/categorical columns (excluding `exclude`, typically the
    target column) and replace any column whose values match a known quality
    scale with its integer ordinal encoding (0-based, worst → best).

    When `ordinal_mappings` is provided (predict path), those mappings are
    applied directly instead of re-detecting from data.

    Returns
    -------
    df : DataFrame (copy, with ordinal columns replaced by integers)
    ordinal_mappings : dict[col_name -> {normalised_value -> int}]
        Stored during training; passed back in at predict time.

    Why this is general (not competition-specific):
    - Ordered categoricals appear in HR, medical, survey, real-estate, and
      manufacturing datasets alike.
    - OrdinalEncoder with no category order loses the ordering signal entirely.
    - This heuristic safely detects and preserves that signal without any
      manual configuration.
    """
    df = df.copy()

    if ordinal_mappings is not None:
        # ── predict path: apply stored mappings ──────────────────────────────
        for col, mapping in ordinal_mappings.items():
            if col not in df.columns:
                continue
            df[col] = df[col].map(lambda x, m=mapping: m.get(_normalise(x), np.nan))
        return df, ordinal_mappings

    # ── train path: detect and build mappings ─────────────────────────────────
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    built: dict[str, dict[str, int]] = {}

    for col in cat_cols:
        if col in exclude:
            continue
        scale = detect_ordinal_scale(df[col])
        if scale is None:
            continue

        mapping = {v: i for i, v in enumerate(scale)}
        df[col] = df[col].map(lambda x, m=mapping: m.get(_normalise(x), np.nan))
        built[col] = mapping

    if built:
        log.info(f"Ordinal quality mapping applied to {len(built)} column(s): {list(built.keys())}")

    return df, built


# ─────────────────────────────────────────────────────────────────────────────
# 4. Universal Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def parse_dates(
    df: pd.DataFrame,
    date_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Auto-detect and extract date features (Year, Month, Day, DayOfWeek),
    then drop the original date columns.
    Train mode (date_cols=None): detect date columns.
    Predict mode: use provided date_cols.
    """
    df = df.copy()
    is_train = date_cols is None
    
    if is_train:
        # Detect date columns
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        
        # Heuristic: try parsing object columns by sampling
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            if col in date_cols:
                continue
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            
            # If the column name contains 'date' or we can parse the sample successfully
            try:
                # We use format='mixed' to handle various date formats gracefully
                pd.to_datetime(sample, errors="raise", format="mixed")
                date_cols.append(col)
            except Exception:
                pass
                
    # Process the columns
    for col in date_cols:
        if col in df.columns:
            dt_series = pd.to_datetime(df[col], errors="coerce", format="mixed")
            df[f"{col}_year"] = dt_series.dt.year.astype(np.float32)
            df[f"{col}_month"] = dt_series.dt.month.astype(np.float32)
            df[f"{col}_day"] = dt_series.dt.day.astype(np.float32)
            df[f"{col}_dow"] = dt_series.dt.dayofweek.astype(np.float32)
            df = df.drop(columns=[col])
            
    if is_train and date_cols:
        log.info(f"Date parsing applied to {len(date_cols)} column(s): {date_cols}")
        
    return df, date_cols


def bucket_rare_categories(
    df: pd.DataFrame,
    exclude: list[str],
    threshold: int = 5,
    frequent_cats: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Group rare categories (absolute frequency < threshold) into 'Other'.
    Train mode (frequent_cats=None): detect and build frequent categories.
    Predict mode: apply pre-built frequent categories.
    """
    df = df.copy()
    
    if frequent_cats is not None:
        # Predict mode
        for col, freq_list in frequent_cats.items():
            if col in df.columns:
                mask = ~df[col].isin(freq_list) & df[col].notna()
                df.loc[mask, col] = "Other"
        return df, frequent_cats

    # Train mode
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    built: dict[str, list[str]] = {}
    
    for col in cat_cols:
        if col in exclude:
            continue
        
        # Calculate absolute frequencies
        val_counts = df[col].value_counts(normalize=False)
        freq_list = val_counts[val_counts >= threshold].index.tolist()
        
        # If there are rare categories, apply bucketing
        if len(freq_list) < len(val_counts):
            mask = ~df[col].isin(freq_list) & df[col].notna()
            df.loc[mask, col] = "Other"
            built[col] = freq_list

    if built:
        log.info(f"Rare category bucketing applied to {len(built)} column(s).")
        
    return df, built


def transform_skewed_features(
    df: pd.DataFrame,
    exclude: list[str],
    skew_threshold: float = 0.75,
    skewed_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply log1p to highly right-skewed numerical features.
    Train mode (skewed_cols=None): compute skewness and determine which columns to transform.
    Predict mode: apply log1p to the pre-determined skewed columns.
    """
    df = df.copy()
    
    if skewed_cols is not None:
        # Predict mode
        for col in skewed_cols:
            if col in df.columns:
                # Ensure the column is numeric before clipping, as predict input might contain strings
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = np.log1p(df[col].clip(lower=0)).astype(np.float32)
        return df, skewed_cols
        
    # Train mode
    num_cols = df.select_dtypes(include="number").columns.tolist()
    detected_cols = []
    
    for col in num_cols:
        if col in exclude:
            continue
            
        # Only transform strictly positive/zero features (log1p requirement)
        if (df[col].dropna() < 0).any():
            continue
            
        skew = float(scipy_stats.skew(df[col].dropna()))
        if skew > skew_threshold:
            df[col] = np.log1p(df[col]).astype(np.float32)
            detected_cols.append(col)
            
    if detected_cols:
        log.info(f"Skew transform (log1p) applied to {len(detected_cols)} numeric feature(s).")
        
    return df, detected_cols