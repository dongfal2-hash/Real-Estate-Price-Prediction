######################################
# Prediction of Price of Real-estate #
######################################

"""
main.py — Real Estate Price Prediction 파이프라인 오케스트레이터
====================================================================
구조:
  stage0.py  : 공통 유틸(로그/저장 함수) 모듈 — prepare_result_dirs, ReportLogger,
               save_fig, save_dataframe, save_pickle 등 (예전 ml_utils.py 역할)
  stage2.py  : data_processing — 원-핫 인코딩 / drop / target 분리 / EDA 시각화
  stage3.py  : data_modeling   — train/test split + 3개 모델(LR/RF/GBR) 학습·비교
  stage4.py  : model_deployment — 예측 저장 / 시각화 / 모델 pickle 저장 / 인사이트
  data_set() : 이 파일(main.py)에서 직접 정의 — 데이터 로드 + EDA 로그 기록

전부 같은 폴더에 있는 flat import 구조.

★ 이 파일 상단의 INPUT CONSTANT 섹션이 파이프라인 전체의 유일한 설정 지점이다.
  값을 바꾸고 싶으면 각 stage 파일이 아니라 여기만 고치면 된다.
"""

import os
import pandas as pd

from stage0 import prepare_result_dirs, ReportLogger, save_fig, save_dataframe, save_pickle
from stage2 import data_processing
from stage3 import data_modeling
from stage4 import model_deployment

# ---------------------------------------------------------------------------
# Constant path
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_df.csv")  # input name of csv file
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Stage 2: Data Engineering: INPUT CONSTANT 
# ---------------------------------------------------------------------------
HOTCODING_COLS = ["property_type"]   # 원-핫 인코딩할 컬럼. 비우면([]) 그냥 패스
TARGET_FEATURE = "price"              # Y
DROP_FEATURES = []                    # 모델 입력에서 제외할 컬럼

# ---------------------------------------------------------------------------
# Stage 3: train/test split: INPUT CONSTANT 
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_PREFIX = "property_type_Condo"   # None이면 stratify 미적용

# ---------------------------------------------------------------------------
# Stage 3: Hyperparameter: INPUT CONSTANT 
# ---------------------------------------------------------------------------
MODEL1_PARAMS = {}   # LinearRegression (하이퍼파라미터 없음)
MODEL2_PARAMS = {    # RandomForestRegressor
    "n_estimators": 200,
    "criterion": "absolute_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
MODEL3_PARAMS = {    # GradientBoostingRegressor
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

paths = prepare_result_dirs(RESULT_DIR)
log = ReportLogger(os.path.join(paths["txt"], "real_estate_report.txt"))


def data_set(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    input: data_path (cleaned_df.csv 경로)
    output: df

    공통 log(전역)에 데이터 개요(shape, dtype/null 결합 테이블, head/tail)를 기록한다.
    """
    df = pd.read_csv(data_path)

    log.section("Title: REAL ESTATE PRICE PREDICTION")
    log.log(f"Data shape: {df.shape}")

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "null_pct(%)": (df.isnull().mean() * 100).round(2),
    })
    info.index.name = "column"
    log.log(f"\nColumn info:\n{info.to_string()}")
    log.log(f"\nColumn head:\n{df.head(3)}")
    log.log(f"\nColumn tail:\n{df.tail(3)}")

    return df


def main():
    # ---------------- Stage 0: 데이터 로드 + EDA ----------------
    df = data_set(DATA_PATH)

    # ---------------- Stage 2: 데이터 엔지니어링 (인코딩/drop/target 분리) + EDA 시각화 ----------------
    stage2_out = data_processing(
        df,
        hotcoding=HOTCODING_COLS,
        target_feature=TARGET_FEATURE,
        drop_feature=DROP_FEATURES,
        save_dir=paths["csv"], save_dataframe_fn=save_dataframe,
        visual_dir=paths["visual"], save_fig_fn=save_fig,
        log=log,
    )

    # ---------------- Stage 3: 3개 모델(LR/RF/GBR) 학습 및 비교 ----------------
    stage3_out = data_modeling(
        stage2_out["x"], stage2_out["y"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify_prefix=STRATIFY_PREFIX,
        model1_params=MODEL1_PARAMS,
        model2_params=MODEL2_PARAMS,
        model3_params=MODEL3_PARAMS,
        visual_dir=paths["visual"], save_fig_fn=save_fig,
        log=log,
    )

    # ---------------- Stage 4: 결과 저장 / 시각화 / 모델 pickle 저장 / 인사이트 ----------------
    stage4_out = model_deployment(
        stage3_out,
        csv_dir=paths["csv"], save_dataframe_fn=save_dataframe,
        visual_dir=paths["visual"], save_fig_fn=save_fig,
        root_dir=paths["root"], save_pickle_fn=save_pickle,
        log=log,
    )

    log.save()
    print(f"\n[DONE] Results saved under: {RESULT_DIR}")
    print(f"Best model: {stage3_out['best_model_name']}")

    return {"df": df, "stage2": stage2_out, "stage3": stage3_out, "stage4": stage4_out}


if __name__ == "__main__":
    main()