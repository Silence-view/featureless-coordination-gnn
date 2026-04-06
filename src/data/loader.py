"""Load and concatenate parquet transaction files."""

import os
import glob
import logging

import pandas as pd

log = logging.getLogger(__name__)


def load_transactions(data_dir: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read all parquets in data_dir, return single DataFrame.

    Parameters
    ----------
    data_dir : path to folder with .parquet files
    columns  : optional subset of columns to load (saves RAM)
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    log.info("Loading %d parquets from %s", len(files), data_dir)

    dfs = []
    for i, f in enumerate(files):
        dfs.append(pd.read_parquet(f, columns=columns))
        if (i + 1) % 50 == 0:
            log.info("  %d / %d loaded", i + 1, len(files))

    tx = pd.concat(dfs, ignore_index=True)
    del dfs

    if "block_timestamp" in tx.columns:
        tx["block_timestamp"] = pd.to_datetime(tx["block_timestamp"])

    log.info("Loaded %s rows, %d wallets, %d tokens",
             f"{len(tx):,}", tx["wallet"].nunique(),
             tx["mint"].nunique() if "mint" in tx.columns else -1)
    return tx
