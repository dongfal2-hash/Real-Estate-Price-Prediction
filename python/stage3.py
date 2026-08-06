"""
stage3_modeling.py
====================================================================
Stage 3 — Statistics & Machine Learning (모델 준비 / 학습 / 평가)

핵심 질문: 어떤 모델이 가장 좋은가? 그 평가 기준은 무엇인가?

구조:
  - split_train_test()  : 공통 train/test 분할 (모든 모델이 공유)
  - train_model_1/2/3()  : 모델별로 완전히 독립된 함수.
                            나중에 모델을 바꾸고 싶으면 해당 함수 내부만 교체하면 됨.
                            단, 반환 dict의 key 형태(name/model/train_pred/test_pred/metrics)는
                            유지해야 compare_models() 이하가 그대로 동작한다.
  - compare_models()     : 여러 모델 결과를 모아 비교표/최적모델 선정
  - plot_model_comparison(): 비교 그래프 png 저장
  - data_modeling()      : 위 전체를 순서대로 실행하는 진입점
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 공통 입력 상수
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_PREFIX = "property_type_Condo"   # None이면 stratify 미적용


def split_train_test(x, y, test_size: float = None, random_state: int = None, stratify_prefix: str = None):
    """
    공통 train/test 분할. 모든 모델이 동일한 split을 공유한다 (공정한 비교를 위해).
    stratify_prefix로 시작하는 컬럼이 있으면 그 컬럼 기준 stratify 적용.
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
    모델 1/2/3 함수들이 공통으로 쓰는 내부 헬퍼: 학습 -> 예측 -> 평가지표 계산.
    이 헬퍼 자체를 안 써도 되지만, 세 함수의 반환 형태를 통일하기 위해 사용한다.
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
# 모델 1 — Linear Regression
# ---------------------------------------------------------------------------
MODEL1_NAME = "LinearRegression"
MODEL1_PARAMS = {}


def train_model_1(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """
    모델 1: LinearRegression.
    나중에 다른 모델로 바꾸려면 이 함수 안의 estimator만 교체하면 됨.
    (반환 dict 형태: name/model/train_pred/test_pred/metrics 는 유지할 것)
    """
    params = params if params is not None else MODEL1_PARAMS
    estimator = LinearRegression(**params)
    return _fit_and_score(MODEL1_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# 모델 2 — Random Forest Regressor
# ---------------------------------------------------------------------------
MODEL2_NAME = "RandomForestRegressor"
MODEL2_PARAMS = {
    "n_estimators": 200,
    "criterion": "absolute_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def train_model_2(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """모델 2: RandomForestRegressor. 다른 모델로 교체 시 이 함수만 수정."""
    params = params if params is not None else MODEL2_PARAMS
    estimator = RandomForestRegressor(**params)
    return _fit_and_score(MODEL2_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# 모델 3 — Gradient Boosting Regressor
# ---------------------------------------------------------------------------
MODEL3_NAME = "GradientBoostingRegressor"
MODEL3_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}


def train_model_3(x_train, y_train, x_test, y_test, params: dict = None) -> dict:
    """모델 3: GradientBoostingRegressor. 다른 모델로 교체 시 이 함수만 수정."""
    params = params if params is not None else MODEL3_PARAMS
    estimator = GradientBoostingRegressor(**params)
    return _fit_and_score(MODEL3_NAME, estimator, x_train, y_train, x_test, y_test)


# ---------------------------------------------------------------------------
# 비교 / 시각화 / 진입점
# ---------------------------------------------------------------------------
def compare_models(model_results: list, log=None) -> dict:
    """
    train_model_1/2/3()의 결과 리스트를 받아 비교표를 만들고 최적 모델을 고른다.
    model_results 의 각 원소는 {"name","model","train_pred","test_pred","metrics"} 형태여야 한다.
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
    """모델별 test_mae / test_r2 비교 막대그래프 저장. 최적 모델은 초록색으로 강조."""
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
    input: x (Stage 2 산출 feature DataFrame), y (target Series)

    CONSTANT (전부 호출 시 override 가능, 안 넘기면 각 모델 함수의 기본값 사용)
        test_size / random_state / stratify_prefix = train/test 분할 설정
        model1_params = LinearRegression 하이퍼파라미터
        model2_params = RandomForestRegressor 하이퍼파라미터
        model3_params = GradientBoostingRegressor 하이퍼파라미터

    처리 순서:
      1) split_train_test()          — 공통 train/test 분할 (한 번만 수행, 3모델이 공유)
      2) train_model_1/2/3()         — 모델별 독립 학습, 각자의 params 사용
      3) compare_models()            — 비교표 작성, 최적 모델 선정
      4) plot_model_comparison()     — (옵션) 비교 그래프 png 저장 + log 기록

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
