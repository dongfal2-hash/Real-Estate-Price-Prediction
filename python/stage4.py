"""
stage4_model_deployment.py
====================================================================
Stage 4 — Model Deployment (result storage / visualization / insights)

Key questions: What insights does this model provide, and what actions should be taken?

This stage directly consumes the output of Stage 3 data_modeling(), including the
trained_models and predictions dictionaries. It works with any number of models
(2, 3, 5, etc.) without code changes.

Responsibilities of this stage:
  1) Save test predictions as CSV
  2) Create visualizations (Actual vs Predicted by model, best-model feature importance, and MAE comparison)
  3) Save all trained models as pickle files
  4) Generate actionable insights (business conclusions)
Training logic is outside this stage; it only consumes Stage 3 output.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_predictions(model_output: dict, csv_dir: str, save_dataframe_fn,
                      filename: str = "test_predictions.csv") -> str:
    """Save test-set actuals and each model's predictions to CSV, regardless of model count."""
    y_test = model_output["y_test"]
    pred_df = pd.DataFrame({"actual": y_test.values})
    for name, preds in model_output["predictions"].items():
        pred_df[f"{name}_pred"] = preds["test_pred"]
    return save_dataframe_fn(pred_df, csv_dir, filename)


def plot_actual_vs_predicted(model_output: dict, visual_dir: str, save_fig_fn,
                              filename: str = "actual_vs_predicted.png") -> str:
    """Plot Actual vs Predicted by model, automatically adding subplots as needed."""
    y_test = model_output["y_test"]
    predictions = model_output["predictions"]
    names = list(predictions.keys())
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    palette = ["steelblue", "seagreen", "salmon", "orange", "purple", "gray"]
    for i, name in enumerate(names):
        test_pred = predictions[name]["test_pred"]
        ax = axes[i]
        ax.scatter(y_test, test_pred, alpha=0.4, color=palette[i % len(palette)])
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")

    plt.tight_layout()
    return save_fig_fn(fig, visual_dir, filename)


def plot_feature_importance(model_output: dict, visual_dir: str, save_fig_fn,
                             filename: str = "feature_importance.png", log=None):
    """
    Visualize feature_importances_ for tree-based models or coef_ for linear models,
    using best_model. If neither attribute exists, skip quietly and return None.
    """
    best_model = model_output["best_model"]
    best_name = model_output["best_model_name"]
    feature_columns = model_output.get("feature_columns") or list(model_output["x_train"].columns)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values()
        title = f"{best_name} Feature Importance"
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(best_model.coef_, index=feature_columns).sort_values()
        title = f"{best_name} Coefficients"
    else:
        if log:
            log.log(f"[WARN] {best_name} has neither feature_importances_ nor coef_. Skipped importance plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="teal")
    ax.set_title(title)
    plt.tight_layout()
    return save_fig_fn(fig, visual_dir, filename)


def plot_mae_comparison(model_output: dict, visual_dir: str, save_fig_fn,
                         filename: str = "mae_comparison.png") -> str:
    """Plot train/test MAE comparison bars for any number of models."""
    metrics_df = model_output["metrics"]
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(metrics_df)), 5))
    metrics_df.plot(x="model", y=["train_mae", "test_mae"], kind="bar", ax=ax,
                     color=["skyblue", "salmon"])
    ax.set_title("Train vs Test MAE by Model")
    ax.set_ylabel("MAE ($)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return save_fig_fn(fig, visual_dir, filename)


def save_models(model_output: dict, root_dir: str, save_pickle_fn) -> dict:
    """Save every model in trained_models as a pickle file, regardless of model count."""
    paths = {}
    for name, model in model_output["trained_models"].items():
        path = save_pickle_fn(model, os.path.join(root_dir, f"RE_{name}_Model.pkl"))
        paths[f"{name}_path"] = path
    return paths


def generate_insight(model_output: dict, log=None) -> str:
    """Generate business insights from best_model using feature_importances_ or coef_."""
    metrics_df = model_output["metrics"]
    best = metrics_df.loc[metrics_df["test_mae"].idxmin()]
    best_model = model_output["best_model"]
    feature_columns = model_output.get("feature_columns") or list(model_output["x_train"].columns)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_columns)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(abs(pd.Series(best_model.coef_)), index=feature_columns)
    else:
        importances = pd.Series(dtype=float)

    top_features = importances.sort_values(ascending=False).head(3)
    top_str = ", ".join(top_features.index) if len(top_features) else "N/A"

    insight = (
        f"[Model to deploy] {best['model']} "
        f"(Test MAE ${best['test_mae']:,.0f}, R2 {best['test_r2']:.3f}).\n"
        f"[Top drivers] The features affecting price most are {top_str}, in descending order.\n"
        f"[Action] Adjust pricing guidelines based on these leading factors, and flag "
        f"listings with large gaps between actual and predicted prices for separate review."
    )

    if log:
        log.section("STAGE 4 — MODEL DEPLOYMENT / INSIGHT")
        log.log(insight)

    return insight


def model_deployment(
    model_output: dict,
    csv_dir: str = None,
    save_dataframe_fn=None,
    visual_dir: str = None,
    save_fig_fn=None,
    root_dir: str = None,
    save_pickle_fn=None,
    log=None,
) -> dict:
    """
    input: model_output — return value from Stage 3 data_modeling()
           (x_train/x_test/y_train/y_test, trained_models, predictions,
            including metrics, best_model, and best_model_name)

    Processing sequence:
      1) Save test predictions as CSV              (requires csv_dir + save_dataframe_fn)
      2) Save Actual vs Predicted by model         (requires visual_dir + save_fig_fn)
      3) Save best-model feature importance        (same requirements as above)
      4) Save the train/test MAE comparison chart  (same requirements as above)
      5) Save all trained models as pickle files   (requires root_dir + save_pickle_fn)
      6) Generate and log business insights

    Each step is quietly skipped if its required arguments are missing, allowing partial execution.

    output: {"prediction_csv_path","actual_vs_predicted_path","importance_path",
             "mae_comparison_path","model_paths","insight"}
    """
    result = {}

    if csv_dir and save_dataframe_fn:
        result["prediction_csv_path"] = save_predictions(model_output, csv_dir, save_dataframe_fn)

    if visual_dir and save_fig_fn:
        result["actual_vs_predicted_path"] = plot_actual_vs_predicted(model_output, visual_dir, save_fig_fn)
        result["importance_path"] = plot_feature_importance(model_output, visual_dir, save_fig_fn, log=log)
        result["mae_comparison_path"] = plot_mae_comparison(model_output, visual_dir, save_fig_fn)

    if root_dir and save_pickle_fn:
        result["model_paths"] = save_models(model_output, root_dir, save_pickle_fn)

    result["insight"] = generate_insight(model_output, log=log)

    return result
