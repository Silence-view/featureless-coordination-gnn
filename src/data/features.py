"""Token feature extraction from MemeTrans feat_label.csv.

Keeps ~110 features from Groups 2-4 (holder concentration, market
activity, bundle metrics). Drops Group 1 temporal cols and anything
that leaks labels.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# stuff we definitely don't want as features
_DROP = {
    "Unnamed: 0", "mint_ts", "mint_address",
    "group1_price", "group1_migrate_year", "group1_migrate_month",
    "group1_migrate_day", "group1_migrate_hour", "group1_migrate_weekday",
    "token", "time",
    "label", "min_ratio", "pred_proba", "return_ratio",
}


def load_feat_label(path: str) -> pd.DataFrame:
    """Just reads the csv."""
    df = pd.read_csv(path)
    log.info("feat_label shape: %s", df.shape)
    return df


def extract_token_features(df: pd.DataFrame):
    """Pull out feature matrix X, binary labels y, and mint addresses.

    Returns (X: DataFrame, y: Series, mints: Series)
    """
    feat_cols = [c for c in df.columns if c not in _DROP]
    log.info("Using %d / %d columns as features", len(feat_cols), len(df.columns))

    X = df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.fillna(0)  # fallback for all-NaN cols

    y = (df["label"] == "high").astype(np.float32)
    mints = df["mint_address"]

    return X, y, mints


def zscore_normalise(train_X: pd.DataFrame, *others, clip=10.0):
    """Fit z-score on train, apply to everything.

    Clips to [-clip, clip] so outliers don't blow up gradients.
    """
    mu = train_X.mean()
    sigma = train_X.std().replace(0, 1).clip(lower=1e-6)

    def _norm(X):
        return ((X - mu) / sigma).clip(-clip, clip)

    results = [_norm(train_X)]
    for X in others:
        results.append(_norm(X))
    return tuple(results)
