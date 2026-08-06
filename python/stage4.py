"""
stage4_model_deployment.py
====================================================================
Stage 4 — Model Deployment (결과 저장 / 시각화 / 인사이트)

핵심 질문: 이 모델을 통해 얻을 수 있는 인사이트는 무엇이며,
           결국 무엇을 해야 하는가 (what to do)?

Stage 3(data_modeling)의 산출물(trained_models / predictions 딕셔너리 구조)을
그대로 입력받아 동작한다. 모델 개수가 몇 개든(2개, 3개, 5개...) 코드 수정 없이 동작.

이 단계의 책임:
  1) test 예측 결과 CSV 저장
  2) 시각화 (모델별 Actual vs Predicted, best_model Feature Importance, MAE 비교)
  3) 모든 학습된 모델 pickle 저장
  4) 실행 가능한 인사이트(비즈니스 결론) 생성
학습 로직 자체는 다루지 않는다 (Stage 3 산출물을 입력으로만 사용).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_predictions(model_output: dict, csv_dir: str, save_dataframe_fn,
                      filename: str = "test_predictions.csv") -> str:
    """test set 실제값 + 모델별 예측값을 CSV로 저장. 모델 개수와 무관하게 동작."""
    y_test = model_output["y_test"]
    pred_df = pd.DataFrame({"actual": y_test.values})
    for name, preds in model_output["predictions"].items():
        pred_df[f"{name}_pred"] = preds["test_pred"]
    return save_dataframe_fn(pred_df, csv_dir, filename)


def plot_actual_vs_predicted(model_output: dict, visual_dir: str, save_fig_fn,
                              filename: str = "actual_vs_predicted.png") -> str:
    """모델별 Actual vs Predicted 산점도. 모델 개수에 따라 subplot 개수가 자동으로 늘어남."""
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
    best_model 기준으로 feature_importances_(트리 계열) 또는 coef_(선형 계열)를 시각화.
    둘 다 없는 모델이면 스킵하고 None 반환 (에러 없이 조용히 넘어감).
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
    """모델별 train/test MAE 비교 막대그래프. 모델 개수와 무관하게 동작."""
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
    """trained_models 안의 모든 모델을 pickle로 저장. 모델 개수와 무관하게 동작."""
    paths = {}
    for name, model in model_output["trained_models"].items():
        path = save_pickle_fn(model, os.path.join(root_dir, f"RE_{name}_Model.pkl"))
        paths[f"{name}_path"] = path
    return paths


def generate_insight(model_output: dict, log=None) -> str:
    """best_model 기준 비즈니스 인사이트 생성 (feature_importances_ 또는 coef_ 활용)."""
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
        f"[Top drivers] {top_str} 순으로 가격에 영향을 미침.\n"
        f"[Action] 위 상위 요인을 기준으로 가격 책정 가이드라인을 조정하고, "
        f"실거래가와 예측가 차이가 큰 매물은 별도 검토 대상으로 플래그 처리할 것."
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
    input: model_output — Stage 3 data_modeling()의 반환값
           (x_train/x_test/y_train/y_test, trained_models, predictions,
            metrics, best_model, best_model_name 을 포함)

    처리 순서:
      1) test 예측 결과 CSV 저장            (csv_dir + save_dataframe_fn 필요)
      2) 모델별 Actual vs Predicted 저장     (visual_dir + save_fig_fn 필요)
      3) best_model Feature Importance 저장  (위와 동일)
      4) train/test MAE 비교 그래프 저장     (위와 동일)
      5) 모든 학습된 모델 pickle 저장         (root_dir + save_pickle_fn 필요)
      6) 비즈니스 인사이트 생성 + log

    각 단계는 필요한 인자가 없으면 조용히 스킵된다 (부분 실행 가능).

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
