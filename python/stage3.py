"""
stage3_modeling.py
====================================================================
Stage 3 — Statistics & Machine Learning (model preparation / training / evaluation)

Key questions: Which model performs best, and what evaluation criteria should be used?

Structure:
  - split_train_test()   : Common train/test split shared by all models
  - train_model_1/2/3()  : Fully independent function for each model. To replace a
                            model later, change only its function. Keep the return
                            dictionary keys (name/model/train_pred/test_pred/metrics)
                            so compare_models() and downstream code continue to work.
  - compare_models()     : Combine model results, build a comparison table, and select the best model
  - plot_model_comparison(): Save the comparison chart as a PNG
  - data_modeling()      : Entry point that runs all steps above in sequence
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_PREFIX = "property_type_Condo"

def split_train_test(x, y, test_size: float = None, random_state: int = None, stratify_prefix: str = None):
    """
    Create a common train/test split shared by all models for a fair comparison.
    If a column starts with stratify_prefix, use that column for stratification.
    """
    test_size = test_size if test_size is not None else TEST_SIZE
    random_state = random_state if random_state is not None else RANDOM_STATE
    stratify_prefix = stratify_prefix if stratify_prefix is not None else STRATIFY_PREFIX

    stratify_col = None
    if stratify_prefix:
        stratify_col = next((c for c in x.columns if c.startswith(stratify_prefix)), None)

    return train_test_split(
        x, y, test_size=test_size, random_state=random_state,
        stratify=x[stratify_col] if stratify_col else None,
    )


def _compute_metrics(name, y_train, train_pred, y_test, test_pred) -> dict:
    return {
        "model": name,
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_rmse": mean_squared_error(y_test, test_pred) ** 0.5,
        "test_r2": r2_score(y_test, test_pred),
    }


def _fit_and_score(name: str, estimator, x_train, y_train, x_test, y_test) -> dict:
    """
    Internal helper shared by model functions 1/2/3: fit -> predict -> calculate metrics.
    Using this helper is optional, but it keeps the return format consistent.
    """
    estimator.fit(x_train, y_train)
    train_pred = estimator.predict(x_train)
    test_pred = estimator.predict(x_test)
    return {
        "name": name,
        "model": estimator,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "metrics": _compute_metrics(name, y_train, train_pred, y_test, test_pred),
    }


# ---------------------------------------------------------------------------
# Model 1 — Linear Regression
# ---------------------------------------------------------------------------
MODEL1_NAME = "LinearRegression"
MODEL1_PARAMS = {}


def train_model_1(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """
    Model 1: LinearRegression.
    To use a different model later, replace only the estimator in this function.
    Keep the return dictionary format: name/model/train_pred/test_pred/metrics.
    """
    params = params if params is not None else MODEL1_PARAMS
    estimator = LinearRegression(**params)
    return _fit_and_score(MODEL1_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# Model 2 — Random Forest Regressor
# ---------------------------------------------------------------------------
MODEL2_NAME = "RandomForestRegressor"
MODEL2_PARAMS = {
    "n_estimators": 200,
    "criterion": "absolute_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def train_model_2(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """Model 2: RandomForestRegressor. Edit only this function to replace the model."""
    params = params if params is not None else MODEL2_PARAMS
    estimator = RandomForestRegressor(**params)
    return _fit_and_score(MODEL2_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# Model 3 — Gradient Boosting Regressor
# ---------------------------------------------------------------------------
MODEL3_NAME = "GradientBoostingRegressor"
MODEL3_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}


def train_model_3(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """Model 3: GradientBoostingRegressor. Edit only this function to replace the model."""
    params = params if params is not None else MODEL3_PARAMS
    estimator = GradientBoostingRegressor(**params)
    return _fit_and_score(MODEL3_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# Comparison / visualization / entry point
# ---------------------------------------------------------------------------
def compare_models(model_results: list, log=None) -> dict:
    """
    Build a comparison table from train_model_1/2/3() results and select the best model.
    Each item in model_results must contain name/model/train_pred/test_pred/metrics.
    """
    metrics_df = pd.DataFrame([r["metrics"] for r in model_results])
    metrics_df = metrics_df.sort_values("test_mae").reset_index(drop=True)
    best_name = metrics_df.iloc[0]["model"]

    trained_models = {r["name"]: r["model"] for r in model_results}
    predictions = {r["name"]: {"train_pred": r["train_pred"], "test_pred": r["test_pred"]} for r in model_results}

    if log:
        log.section("STAGE 3 — STATISTICS & MACHINE LEARNING")
        log.log(f"Models compared: {[r['name'] for r in model_results]}")
        for r in model_results:
            row = r["metrics"]
            log.log(f"\n[{row['model']}]")
            log.log(f"  Train MAE: {row['train_mae']:,.2f}")
            log.log(f"  Test MAE:  {row['test_mae']:,.2f}")
            log.log(f"  Test RMSE: {row['test_rmse']:,.2f}")
            log.log(f"  Test R2:   {row['test_r2']:.4f}")
        log.log(f"\nBest model (lowest test MAE): {best_name}")

    return {
        "metrics": metrics_df,
        "trained_models": trained_models,
        "predictions": predictions,
        "best_model_name": best_name,
        "best_model": trained_models[best_name],
    }


def plot_model_comparison(metrics_df: pd.DataFrame, visual_dir: str, save_fig_fn,
                            filename: str = "model_comparison.png", log=None) -> str:
    """Save test MAE/R2 comparison bars by model, highlighting the best model in green."""
    df_sorted = metrics_df.sort_values("test_mae")
    best_model = df_sorted.iloc[0]["model"]
    colors = ["seagreen" if m == best_model else "lightgray" for m in df_sorted["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df_sorted["model"], df_sorted["test_mae"], color=colors)
    axes[0].set_title("Test MAE by Model (lower = better)")
    axes[0].set_ylabel("MAE ($)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df_sorted["model"], df_sorted["test_r2"], color=colors)
    axes[1].set_title("Test R2 by Model (higher = better)")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Model Comparison — Best: {best_model}")
    plt.tight_layout()
    path = save_fig_fn(fig, visual_dir, filename)

    if log:
        log.section("STAGE 3 — MODEL COMPARISON GRAPH")
        log.log(f"Saved: {path}")
        log.log(f"Best model highlighted: {best_model}")

    return path


def data_modeling(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None,
    stratify_prefix: str = None,
    model1_params: dict = None,
    model2_params: dict = None,
    model3_params: dict = None,
    visual_dir: str = None,
    save_fig_fn=None,
    comparison_filename: str = "model_comparison.png",
    log=None,
) -> dict:
    """
    input: x (feature DataFrame produced by Stage 2), y (target Series)

    CONSTANTS (all can be overridden when calling; otherwise, model defaults are used)
        test_size / random_state / stratify_prefix = train/test split settings
        model1_params = LinearRegression hyperparameters
        model2_params = RandomForestRegressor hyperparameters
        model3_params = GradientBoostingRegressor hyperparameters

    Processing sequence:
      1) split_train_test()          — Create one common train/test split shared by all three models
      2) train_model_1/2/3()         — Train each model independently using its own parameters
      3) compare_models()            — Build a comparison table and select the best model
      4) plot_model_comparison()     — Optionally save a comparison PNG and write to the log

    output: {"x_train","x_test","y_train","y_test","metrics","trained_models",
             "predictions","best_model_name","best_model","comparison_plot_path"}
    """
    x_train, x_test, y_train, y_test = split_train_test(
        x, y, test_size=test_size, random_state=random_state, stratify_prefix=stratify_prefix,
    )

    if log:
        log.section("STAGE 3 — HYPERPARAMETERS USED")
        log.log(f"test_size={test_size if test_size is not None else TEST_SIZE}, "
                f"random_state={random_state if random_state is not None else RANDOM_STATE}, "
                f"stratify_prefix={stratify_prefix if stratify_prefix is not None else STRATIFY_PREFIX}")
        log.log(f"Model1 (LinearRegression) params: {model1_params if model1_params is not None else MODEL1_PARAMS}")
        log.log(f"Model2 (RandomForestRegressor) params: {model2_params if model2_params is not None else MODEL2_PARAMS}")
        log.log(f"Model3 (GradientBoostingRegressor) params: {model3_params if model3_params is not None else MODEL3_PARAMS}")

    result_1 = train_model_1(x_train, y_train, x_test, y_test, params=model1_params)
    result_2 = train_model_2(x_train, y_train, x_test, y_test, params=model2_params)
    result_3 = train_model_3(x_train, y_train, x_test, y_test, params=model3_params)

    compared = compare_models([result_1, result_2, result_3], log=log)

    comparison_path = None
    if visual_dir and save_fig_fn:
        comparison_path = plot_model_comparison(
            compared["metrics"], visual_dir, save_fig_fn, filename=comparison_filename, log=log,
        )

    return {
        "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test,
        **compared,
        "comparison_plot_path": comparison_path,
    }
