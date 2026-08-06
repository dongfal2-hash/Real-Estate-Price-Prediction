"""
streamlit_app.py
====================================================================
Real Estate Price Prediction — 4-tab Streamlit app (unified dark theme)

Principle: read only what already exists in results/ (pkl/csv/png).
           No retraining, no regenerating plots — cache the loads,
           compute only lightweight metrics on the fly.

Tab 1  Prediction        : answer the business question using the
                            production model (Random Forest)
Tab 2  Model Selection    : why Random Forest was chosen among 3 models
Tab 3  EDA                 : data exploration results (heatmap/histogram)
Tab 4  Summary              : 4-step executive summary

Theme: black background / white text, with one accent color per section
       (see SECTION_COLORS below). Colors and dark theme are also set
       app-wide in .streamlit/config.toml.
"""

import os
import pickle
import sys

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(BASE_DIR, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from stage2 import build_feature_row, HOTCODING_COLS
from stage3 import MODEL2_NAME  # "RandomForestRegressor"

DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_df.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR = os.path.join(RESULT_DIR, "csv")
VISUAL_DIR = os.path.join(RESULT_DIR, "visual")
SLIDE_DIR = os.path.join(BASE_DIR, "slide")
SUMMARY_IMG_PATH = os.path.join(SLIDE_DIR, "summary.png")
RF_MODEL_PATH = os.path.join(RESULT_DIR, f"RE_{MODEL2_NAME}_Model.pkl")

st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Theme constants — shared across every tab
# ---------------------------------------------------------------------------
# Section accent colors (reused from the Summary step colors, mapped by meaning):
#   Business needs (blue)       -> Prediction tab (answers the business question)
#   Data engineering (purple)   -> EDA tab (data exploration)
#   Statistics and ML/DL (green)-> Model Selection tab
#   BI intelligence (yellow)    -> Summary tab
COLOR_BLUE = "#8EC9F0"
COLOR_PURPLE = "#C9A6E8"
COLOR_GREEN = "#A8E6A3"
COLOR_YELLOW = "#FFF09E"

STEP_COLORS = {1: COLOR_BLUE, 2: COLOR_PURPLE, 3: COLOR_GREEN, 4: COLOR_YELLOW}
_NUMBER_LABEL = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣"}

BG_COLOR = "#000000"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT_COLOR = "#B3B3B3"
FONT_FAMILY = "'Inter', sans-serif"

_CARD_TEMPLATE = """
<div style="background:{bg}; color:{text}; border:2px solid {color};
            border-radius:8px; padding:16px; height:100%; font-family:{font};">
  <div style="font-size:26px; color:{color}; font-weight:bold;">{number}</div>
  <div style="font-weight:bold; font-size:18px; margin:6px 0 10px 0; color:{text};">{title}</div>
  <ul style="margin:0; padding-left:18px; font-size:14px; line-height:1.5; color:{text};">{bullets}</ul>
</div>
"""

_SECTION_HEADER_TEMPLATE = """
<div style="border-left:4px solid {color}; padding:2px 0 2px 14px; margin-bottom:16px;">
  <div style="font-size:24px; font-weight:bold; color:{text}; font-family:{font};">{title}</div>
  {subtitle_html}
</div>
"""


def _bullets_html(items: list) -> str:
    return "".join(f"<li>{i}</li>" for i in items)


def render_section_header(color: str, title: str, subtitle: str = None):
    """Consistent header style used at the top of every tab: colored left bar, no icons."""
    subtitle_html = (
        f'<div style="font-size:14px; color:{MUTED_TEXT_COLOR}; margin-top:4px;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(_SECTION_HEADER_TEMPLATE.format(
        color=color, text=TEXT_COLOR, font=FONT_FAMILY, title=title, subtitle_html=subtitle_html,
    ), unsafe_allow_html=True)


def _render_step_card(step: int, title: str, bullets: list):
    st.markdown(_CARD_TEMPLATE.format(
        bg=BG_COLOR, text=TEXT_COLOR, font=FONT_FAMILY,
        color=STEP_COLORS[step], number=_NUMBER_LABEL[step],
        title=title, bullets=_bullets_html(bullets),
    ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders — everything here only READS existing results (no recompute)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_encoded_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, "encoded_data.csv"))


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(os.path.join(CSV_DIR, "test_predictions.csv"))


@st.cache_resource
def load_rf_model():
    with open(RF_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def compute_model_metrics() -> pd.DataFrame:
    """Compute test MAE/RMSE/R2 per model from test_predictions.csv (lightweight, no retraining)."""
    pred_df = load_test_predictions()
    model_cols = [c for c in pred_df.columns if c.endswith("_pred")]
    rows = []
    for col in model_cols:
        name = col.replace("_pred", "")
        rows.append({
            "model": name,
            "test_mae": mean_absolute_error(pred_df["actual"], pred_df[col]),
            "test_rmse": mean_squared_error(pred_df["actual"], pred_df[col]) ** 0.5,
            "test_r2": r2_score(pred_df["actual"], pred_df[col]),
        })
    return pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)


def feature_columns_from_encoded(encoded_df: pd.DataFrame, target: str = "price") -> list:
    return [c for c in encoded_df.columns if c != target]


# ---------------------------------------------------------------------------
# Tab 1 — Prediction (answers the core question with the production model)
# ---------------------------------------------------------------------------
def render_prediction_tab():
    render_section_header(
        COLOR_BLUE,
        "Can predictive modeling outperform simple valuation heuristics for home prices?",
        subtitle=f"Production model: {MODEL2_NAME}",
    )

    raw_df = load_raw_data()
    encoded_df = load_encoded_data()
    feature_columns = feature_columns_from_encoded(encoded_df)
    rf_model = load_rf_model()

    col1, col2, col3 = st.columns(3)
    with col1:
        property_type = st.selectbox("Property type", sorted(raw_df["property_type"].unique()))
        beds = st.slider("Beds", int(raw_df["beds"].min()), int(raw_df["beds"].max()), int(raw_df["beds"].median()))
        baths = st.slider("Baths", int(raw_df["baths"].min()), int(raw_df["baths"].max()), int(raw_df["baths"].median()))
    with col2:
        sqft = st.slider("Sqft", int(raw_df["sqft"].min()), int(raw_df["sqft"].max()),
                          int(raw_df["sqft"].median()), step=50)
        year_built = st.slider("Year built", int(raw_df["year_built"].min()), int(raw_df["year_built"].max()),
                                int(raw_df["year_built"].median()))
        lot_size = st.slider("Lot size", int(raw_df["lot_size"].min()), int(raw_df["lot_size"].max()),
                              int(raw_df["lot_size"].median()), step=100)
    with col3:
        basement = st.radio("Has basement?", ["Yes", "No"], horizontal=True)
        property_tax = st.slider("Property tax ($)", int(raw_df["property_tax"].min()),
                                  int(raw_df["property_tax"].max()), int(raw_df["property_tax"].median()))
        insurance = st.slider("Insurance ($)", int(raw_df["insurance"].min()), int(raw_df["insurance"].max()),
                               int(raw_df["insurance"].median()))
        year_sold = st.slider("Year sold", int(raw_df["year_sold"].min()), int(raw_df["year_sold"].max()),
                               int(raw_df["year_sold"].max()))

    inputs = {
        "property_type": property_type,
        "beds": beds, "baths": baths, "sqft": sqft,
        "year_built": year_built, "lot_size": lot_size,
        "basement": 1 if basement == "Yes" else 0,
        "property_tax": property_tax, "insurance": insurance, "year_sold": year_sold,
    }

    if st.button("Predict", type="primary"):
        feature_row = build_feature_row(feature_columns, inputs, hotcoding=HOTCODING_COLS)
        rf_pred = rf_model.predict(feature_row)[0]

        # Heuristic A: average $/sqft x input sqft
        price_per_sqft = (raw_df["price"] / raw_df["sqft"]).mean()
        heuristic_a = price_per_sqft * sqft

        # Heuristic B: average sale price by property_type
        heuristic_b = raw_df.loc[raw_df["property_type"] == property_type, "price"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{MODEL2_NAME} (model)", f"${rf_pred:,.0f}")
        c2.metric("Heuristic A: avg $/sqft", f"${heuristic_a:,.0f}", f"{rf_pred - heuristic_a:+,.0f} vs model")
        c3.metric(f"Heuristic B: {property_type} average", f"${heuristic_b:,.0f}",
                   f"{rf_pred - heuristic_b:+,.0f} vs model")

        st.caption(
            "A larger gap from the heuristics means the model is capturing price factors "
            "(quality, location, timing) that simple rules of thumb miss."
        )


# ---------------------------------------------------------------------------
# Tab 2 — Model Selection (why Random Forest)
# ---------------------------------------------------------------------------
def render_model_selection_tab():
    render_section_header(COLOR_GREEN, "Why Random Forest was selected",
                           subtitle="Comparison across 3 models on the held-out test set")

    metrics_df = compute_model_metrics()
    best_name = metrics_df.iloc[0]["model"]

    def highlight_best(row):
        return [f"background-color: {COLOR_GREEN}; color: #000000" if row["model"] == best_name else ""
                for _ in row]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format({
            "test_mae": "${:,.0f}", "test_rmse": "${:,.0f}", "test_r2": "{:.3f}",
        }),
        use_container_width=True,
    )
    st.success(f"Selected model: {best_name} (lowest test MAE)")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(VISUAL_DIR, "model_comparison.png"), caption="Test MAE / R2 by model")
    with col2:
        st.image(os.path.join(VISUAL_DIR, "mae_comparison.png"), caption="Train vs test MAE by model")


# ---------------------------------------------------------------------------
# Tab 3 — EDA
# ---------------------------------------------------------------------------
def render_eda_tab():
    render_section_header(COLOR_PURPLE, "EDA results", subtitle="What drives price in the data")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(VISUAL_DIR, "EDA_heatmap.png"), caption="Feature-target correlation")
    with col2:
        st.image(os.path.join(VISUAL_DIR, "EDA_target_histogram.png"), caption="Price distribution")
    st.caption("Features with a larger absolute correlation carry more explanatory power for price.")


# ---------------------------------------------------------------------------
# Tab 4 — Summary (4-step executive summary)
# ---------------------------------------------------------------------------
def render_summary_tab():
    render_section_header(COLOR_YELLOW, "Summary")

    if os.path.exists(SUMMARY_IMG_PATH):
        st.image(SUMMARY_IMG_PATH, use_container_width=True)
    else:
        st.error(
            f"Summary image not found at:\n\n`{SUMMARY_IMG_PATH}`\n\n"
            "Place the summary slide PNG at that path (results/slide/summary.png "
            "relative to this app), or update SUMMARY_IMG_PATH at the top of the file."
        )

    st.caption("Tools: Python | pandas | scikit-learn | matplotlib | Streamlit")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        f"<h1 style='color:{TEXT_COLOR}; font-family:{FONT_FAMILY};'>Real Estate Price Prediction</h1>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Selection", "EDA", "Summary"])
    with tab1:
        render_prediction_tab()
    with tab2:
        render_model_selection_tab()
    with tab3:
        render_eda_tab()
    with tab4:
        render_summary_tab()


if __name__ == "__main__":
    main()
