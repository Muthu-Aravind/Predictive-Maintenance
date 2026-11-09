"""Streamlit interface for numeric engine-condition prediction."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "engine_data.csv"

st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)
st.title("Engine Health - Predictive Maintenance")
st.caption("Sensor-based screening using the registered machine-learning model")

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    st.error("The registered model is not available. Run the GitHub Actions training workflow first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Unable to load the registered model: {exc}")
    st.stop()

features = metadata["features"]
if DATA_PATH.is_file():
    reference = pd.read_csv(DATA_PATH)
    if not set(features).issubset(reference.columns):
        st.error("The registered dataset does not contain the model's expected input columns.")
        st.stop()
else:
    st.error("The registered source dataset is missing; it is required to establish safe input ranges.")
    st.stop()

st.info(
    "The supplied dataset documentation does not define whether class 0 or class 1 represents engine failure. "
    "This application reports numeric class predictions only. Do not use it as an automatic maintenance or safety decision."
)

st.subheader("Enter engine sensor readings")
st.caption("Input ranges are bounded by the observed training-source dataset range. Confirm units and operating limits with an engineer.")

columns = st.columns(2)
defaults = {}
for index, feature in enumerate(features):
    series = reference[feature].astype(float)
    defaults[feature] = {
        "min": float(series.min()),
        "max": float(series.max()),
        "value": float(series.median()),
    }

with st.form("engine_sensor_form"):
    values = {}
    for index, feature in enumerate(features):
        limits = defaults[feature]
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                min_value=limits["min"],
                max_value=limits["max"],
                value=limits["value"],
                format="%.3f",
                key=f"sensor_{index}",
            )
    submitted = st.form_submit_button(
        "Predict engine condition",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_frame = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    classes = [int(value) for value in model.classes_]

    st.subheader("Prediction result")
    st.metric("Predicted recorded class", f"Class {prediction}")
    probability_frame = pd.DataFrame({
        "Numeric class": [f"Class {label}" for label in classes],
        "Model probability": probabilities,
    })
    st.dataframe(probability_frame, use_container_width=True, hide_index=True)
    st.bar_chart(probability_frame.set_index("Numeric class"))
    st.warning(
        "Confirm the target-class meaning, sensor units, calibration and equipment-specific operating limits "
        "with qualified engineers before using this result in a maintenance workflow."
    )

with st.expander("Model details and limitations"):
    st.write("Model:", metadata["model_name"])
    st.write("Selection criterion:", metadata["selection_metric"])
    st.write("Training rows:", metadata["training_rows"])
    st.write("Hold-out rows:", metadata["test_rows"])
    st.json(metadata["test_metrics"])
