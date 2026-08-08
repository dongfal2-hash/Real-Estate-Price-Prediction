"""
stage2_data_engineering.py
====================================================================
Stage 2 — Data Engineering (one-hot encoding / column removal / target separation / EDA visualization)

Key question: Which variables should be included in the model, and in what
              format, to answer the question defined in Stage 1?

Processing sequence: one-hot encoding -> column removal -> target separation (X and y)
                     -> optional EDA visualization (histogram/heatmap)
                     -> optional CSV export
The train/test split is not performed here; it is handled in Stage 3: Modeling.

"""

import pandas as pd
import matplotlib.pyplot as plt

HOTCODING_COLS = ["property_type"]
TARGET_FEATURE = "price"
DROP_FEATURES = []

def one_hot_encode(df: pd.DataFrame, hotcoding: list = None) -> pd.DataFrame:
    """One-hot encode columns in hotcoding; return a copy unchanged if the list is empty."""
    hotcoding = hotcoding if hotcoding is not None else HOTCODING_COLS
    if not hotcoding:
        return df.copy()
    existing = [c for c in hotcoding if c in df.columns]
    if not existing:
        return df.copy()
    return pd.get_dummies(df, columns=existing, drop_first=False, dtype=int)


def drop_columns(df: pd.DataFrame, drop_feature: list = None) -> pd.DataFrame:
    """Remove columns listed in drop_feature, ignoring columns that do not exist."""
    drop_feature = drop_feature if drop_feature is not None else DROP_FEATURES
    if not drop_feature:
        return df.copy()
    existing = [c for c in drop_feature if c in df.columns]
    return df.drop(columns=existing)


def split_target(df: pd.DataFrame, target_feature: str = None):
    """Separate the DataFrame into X and y using target_feature."""
    target_feature = target_feature if target_feature is not None else TARGET_FEATURE
    y = df[target_feature]
    x = df.drop(columns=[target_feature])
    return x, y


# ---------------------------------------------------------------------------
# EDA visualization — Y histogram / correlation heatmap
# ---------------------------------------------------------------------------
def plot_target_histogram(
    y: pd.Series,
    visual_dir: str,
    save_fig_fn,
    filename: str = "EDA_target_histogram.png",
    bins: int = 30,
    log=None,
) -> str:
    """Save a histogram of the target (Y) variable's distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y, bins=bins, color="steelblue", edgecolor="white")
    ax.set_title(f"Distribution of Target: {y.name}")
    ax.set_xlabel(y.name)
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 2 — EDA: TARGET HISTOGRAM")
        log.log(f"Target: {y.name}")
        log.log(f"mean={y.mean():,.2f}  std={y.std():,.2f}  "
                 f"min={y.min():,.2f}  max={y.max():,.2f}")
        log.log(f"Saved: {path}")

    return path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_feature: str,
    visual_dir: str,
    save_fig_fn,
    filename: str = "EDA_heatmap.png",
    log=None,
) -> str:
    """Save a correlation heatmap of numeric variables, including the target."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n), max(5, 0.6 * n)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.columns)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 2 — EDA: CORRELATION HEATMAP")
        if target_feature in corr.columns:
            target_corr = corr[target_feature].drop(target_feature).sort_values(ascending=False)
            log.log(f"Correlation with target '{target_feature}':\n{target_corr.to_string()}")
        log.log(f"Saved: {path}")

    return path


def data_processing(
    df: pd.DataFrame,
    hotcoding: list = None,
    target_feature: str = None,
    drop_feature: list = None,
    save_dir: str = None,
    save_dataframe_fn=None,
    filename: str = "encoded_data.csv",
    visual_dir: str = None,
    save_fig_fn=None,
    heatmap_filename: str = "EDA_heatmap.png",
    histogram_filename: str = "EDA_target_histogram.png",
    log=None,
) -> dict:
    """
    input: df (raw DataFrame loaded from a .csv file)

    INPUT CONSTANT
        hotcoding      = [categorical columns to one-hot encode], e.g. ["property_type"]
        target feature = name of the target (Y) column, e.g. "price"
        drop feature   = columns to exclude from model inputs, e.g. ["year_sold"]

    Processing sequence: one-hot encoding -> column removal -> target separation
                         -> optional EDA visualization (histogram/heatmap)
                         -> optional CSV export

    Visualization options:
        When both visual_dir and save_fig_fn are provided, save and log:
          - EDA_heatmap.png            (feature-target correlations)
          - EDA_target_histogram.png   (target distribution)
        If either argument is missing, skip visualization.

    output: {"encoded_df", "x", "y", "feature_columns", "target_feature", "hotcoding"}
    """
    hotcoding = hotcoding if hotcoding is not None else HOTCODING_COLS
    target_feature = target_feature if target_feature is not None else TARGET_FEATURE
    drop_feature = drop_feature if drop_feature is not None else DROP_FEATURES

    encoded_df = one_hot_encode(df, hotcoding=hotcoding)
    encoded_df = drop_columns(encoded_df, drop_feature=drop_feature)
    x, y = split_target(encoded_df, target_feature=target_feature)
    feature_columns = list(x.columns)

    if log:
        log.section("STAGE 2 — DATA ENGINEERING")
        log.log(f"Hotcoding columns: {hotcoding}")
        log.log(f"Target feature: {target_feature}")
        log.log(f"Dropped features: {drop_feature}")
        log.log(f"Final feature columns ({len(feature_columns)}): {feature_columns}")

    heatmap_path = None
    histogram_path = None
    if visual_dir and save_fig_fn:
        heatmap_path = plot_correlation_heatmap(
            encoded_df, target_feature, visual_dir, save_fig_fn,
            filename=heatmap_filename, log=log,
        )
        histogram_path = plot_target_histogram(
            y, visual_dir, save_fig_fn,
            filename=histogram_filename, log=log,
        )

    if save_dir and save_dataframe_fn:
        save_dataframe_fn(encoded_df, save_dir, filename)

    return {
        "encoded_df": encoded_df, "x": x, "y": y,
        "feature_columns": feature_columns, "target_feature": target_feature,
        "hotcoding": hotcoding,  # Reused by build_feature_row
        "heatmap_path": heatmap_path, "histogram_path": histogram_path,
    }


def build_feature_row(feature_columns: list, inputs: dict, hotcoding: list = None) -> pd.DataFrame:
    """
    Convert user input (dict) into a one-row DataFrame matching the column order
    and structure used during training (for Streamlit predictions). This handles
    all columns listed in hotcoding generically.
    """
    hotcoding = hotcoding if hotcoding is not None else HOTCODING_COLS
    row = {col: 0 for col in feature_columns}

    for key, value in inputs.items():
        if key in hotcoding:
            continue  # One-hot encoded fields are handled separately below
        if key in row:
            row[key] = value

    for cat_col in hotcoding:
        if cat_col in inputs:
            onehot_col = f"{cat_col}_{inputs[cat_col]}"
            if onehot_col in row:
                row[onehot_col] = 1

    return pd.DataFrame([row])[feature_columns]
