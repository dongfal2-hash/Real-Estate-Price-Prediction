"""
ml_utils.py
공통 유틸리티 모듈 - 4개 프로젝트(Real Estate, Loan Eligibility, Clustering, Neural Network)에서
공통으로 사용하는 함수 모음.

기능:
- 결과 폴더 준비 (txt / csv / visual)
- 텍스트 리포트 저장
- 데이터프레임 CSV 저장
- matplotlib figure 저장
- 콘솔 + 파일 동시 출력 로거
- Real Estate 모델 학습 (파일 I/O 없는 순수 함수 -> real_estate_analysis.py, streamlit_app.py 공용)
"""

import os
import sys
import json
import pickle
from contextlib import contextmanager

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 저장 가능하도록
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_result_dirs(base_dir: str) -> dict:
    """base_dir 아래에 txt / csv / visual 하위 폴더를 만들고 경로 dict 반환
    making sub folder below base_dir: txt / csv / visual """

    paths = {
        "root": base_dir,
        "txt": os.path.join(base_dir, "txt"),
        "csv": os.path.join(base_dir, "csv"),
        "visual": os.path.join(base_dir, "visual"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


class ReportLogger:
    """print()한 내용을 콘솔에 출력하면서 동시에 텍스트 파일에도 기록
        record contents on text file """


    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self._buffer = []

    def log(self, *args, sep=" ", end="\n"):
        line = sep.join(str(a) for a in args) + end
        sys.stdout.write(line)
        self._buffer.append(line)

    def section(self, title: str):
        bar = "=" * 70
        self.log("\n" + bar)
        self.log(title)
        self.log(bar)

    def save(self):
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.writelines(self._buffer)


def save_fig(fig_or_plt, visual_dir: str, filename: str, dpi: int = 120):
    """현재 figure 혹은 전달된 fig 객체를 visual 폴더에 저장
        store figure into the visual folder    """

    path = os.path.join(visual_dir, filename)
    if hasattr(fig_or_plt, "savefig"):
        fig_or_plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return path


def save_dataframe(df, csv_dir: str, filename: str):
    path = os.path.join(csv_dir, filename)
    df.to_csv(path, index=False)
    return path


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


# --------------------------------------------------------------------------
# Real Estate: 공용 학습 함수 (파일 I/O 없음, 순수 계산만 수행)
# real_estate_analysis.py 는 이 함수 결과를 파일로 저장하고,
# streamlit_app.py 는 이 함수 결과를 화면에 표시한다.
# --------------------------------------------------------------------------
def train_real_estate_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    cleaned_df.csv 형태의 원본 데이터프레임을 받아
    - property_type 원-핫 인코딩
    - train/test split (property_type_Condo 기준 stratify)
    - LinearRegression, RandomForestRegressor 학습
    - 학습/평가 지표 계산
    을 수행하고 결과를 dict로 반환한다. 디스크에 아무것도 쓰지 않는다.
    """
    encoded_df = pd.get_dummies(df, columns=["property_type"], drop_first=False, dtype=int)

    x = encoded_df.drop("price", axis=1)
    y = encoded_df["price"]

    stratify_col = None
    for col in x.columns:
        if col.startswith("property_type_Condo"):
            stratify_col = col
            break

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state,
        stratify=x[stratify_col] if stratify_col else None,
    )

    lr_model = LinearRegression().fit(x_train, y_train)
    rf_model = RandomForestRegressor(
        n_estimators=200, criterion="absolute_error", random_state=random_state, n_jobs=-1
    ).fit(x_train, y_train)

    lr_train_pred = lr_model.predict(x_train)
    lr_test_pred = lr_model.predict(x_test)
    rf_train_pred = rf_model.predict(x_train)
    rf_test_pred = rf_model.predict(x_test)

    metrics_df = pd.DataFrame([
        {
            "model": "LinearRegression",
            "train_mae": mean_absolute_error(y_train, lr_train_pred),
            "test_mae": mean_absolute_error(y_test, lr_test_pred),
            "test_rmse": mean_squared_error(y_test, lr_test_pred) ** 0.5,
            "test_r2": r2_score(y_test, lr_test_pred),
        },
        {
            "model": "RandomForestRegressor",
            "train_mae": mean_absolute_error(y_train, rf_train_pred),
            "test_mae": mean_absolute_error(y_test, rf_test_pred),
            "test_rmse": mean_squared_error(y_test, rf_test_pred) ** 0.5,
            "test_r2": r2_score(y_test, rf_test_pred),
        },
    ])

    return {
        "encoded_df": encoded_df,
        "feature_columns": list(x.columns),
        "lr_model": lr_model,
        "rf_model": rf_model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "lr_train_pred": lr_train_pred,
        "lr_test_pred": lr_test_pred,
        "rf_train_pred": rf_train_pred,
        "rf_test_pred": rf_test_pred,
        "metrics": metrics_df,
    }


def build_feature_row(feature_columns: list, inputs: dict) -> pd.DataFrame:
    """사용자 입력(dict)을 학습 때 사용한 컬럼 순서/구성에 맞는 1행 데이터프레임으로 변환"""
    row = {col: 0 for col in feature_columns}
    for key, value in inputs.items():
        if key in row:
            row[key] = value

    property_type_col = f"property_type_{inputs.get('property_type', '')}"
    if property_type_col in row:
        row[property_type_col] = 1

    return pd.DataFrame([row])[feature_columns]
