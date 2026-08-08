######################################
# Prediction of Price of Real-estate #
######################################

"""
main.py — Real Estate Price Prediction 

Pipeline orchestration
====================================================================
Structure
  stage0.py  : prepare_result_dirs, ReportLogger,
               save_fig, save_dataframe, save_pickle 
  stage2.py  : data_processing — one-hot encoding , drop, select and drop target feature,  EDA visualization
  stage3.py  : data_modeling   — train/test split - three machine-learning
  stage4.py  : model_deployment — model storage on pickle, storeage of expecation and visualization 
  data_set() : 


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
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "cleaned_df.csv")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
# ---------------------------------------------------------------------------
# Stage 1:  into main() 
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 2: Data Engineering: INPUT CONSTANT 
# ---------------------------------------------------------------------------
HOTCODING_COLS = ["property_type"]   # for one-hotcoding column [list] 
TARGET_FEATURE = "price"              # Y
DROP_FEATURES = []                    # drop features [list]

# ---------------------------------------------------------------------------
# Stage 3: train/test split: INPUT CONSTANT 
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2                         # split train/test size 
RANDOM_STATE = 42                       # random or repeat (42)
STRATIFY_PREFIX = "property_type_Condo"   # column stratify 

# ---------------------------------------------------------------------------
# Stage 3: Hyperparameter: INPUT CONSTANT 
# ---------------------------------------------------------------------------
MODEL1_PARAMS = {}                      # LinearRegression ()
MODEL2_PARAMS = {                       # RandomForestRegressor
    "n_estimators": 200,
    "criterion": "absolute_error",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
MODEL3_PARAMS = {                       # GradientBoostingRegressor
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

paths = prepare_result_dirs(RESULT_DIR)
log = ReportLogger(os.path.join(paths["txt"], "real_estate_report.txt"))


def data_set(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    input: data_path (cleaned_df.csv )
    output: df

    Record a data overview in the shared (global) log, including the shape, 
    a combined table of data types and null values, and the first and last few rows.
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
    # ---------------- Stage 0: load data+ EDA ----------------
    df = data_set(DATA_PATH)

    # ---------------- Stage 2: Data engineering ----------------
    stage2_out = data_processing(
        df,
        hotcoding=HOTCODING_COLS,
        target_feature=TARGET_FEATURE,
        drop_feature=DROP_FEATURES,
        save_dir=paths["csv"], save_dataframe_fn=save_dataframe,
        visual_dir=paths["visual"], save_fig_fn=save_fig,
        log=log,
    )

    # ---------------- Stage 3: modeling _ ML-learning ----------------
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

    # ---------------- Stage 4: store pickle for ML deployment, visualization ----------------
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