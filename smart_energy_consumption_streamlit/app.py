import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st



st.set_page_config(page_title="Smart Home Energy Consumption", layout="wide")

# NOTE: This Streamlit app supports manual single-row predictions (no CSV upload).



st.title("Smart Home Energy Consumption Predictor")
st.write(
    "Manual Smart Home Energy Consumption Predictor (no CSV upload required)."
)



@st.cache_resource
def load_artifact():
    # Artifact is stored inside this Streamlit app folder.
    artifact_path = Path(__file__).resolve().parent / "energy_model.joblib"
    if not artifact_path.exists():
        st.error(
            "Model artifact not found: smart_energy_consumption_streamlit/energy_model.joblib. "
            "Train the model first by running train_energy_model.py."
        )
        st.stop()
    return joblib.load(artifact_path)



artifact = load_artifact()
pipe = artifact["pipeline"]
target_col = artifact["target_col"]


# Manual input (no CSV upload)
# Expected features are taken from the trained artifact.
feature_columns = artifact.get("feature_columns")

if feature_columns is None:
    # Fallback to feature_columns.json for robustness.
    feature_json_path = Path(__file__).resolve().parent / "feature_columns.json"
    if not feature_json_path.exists():
        st.error(
            "Trained artifact is missing feature_columns and feature_columns.json was not found. "
            "Re-train the model with train_energy_model.py."
        )
        st.stop()

    import json

    try:
        feature_columns = json.loads(feature_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not read feature_columns.json: {e}")
        st.stop()


# Build UI inputs for each feature
with st.form("manual_input_form"):
    st.subheader("Enter smart home features")

    input_values = {}

    # Heuristic typing based on training data column dtypes.
    # We treat numeric columns as numbers, and the rest as categories.
    # This keeps the UI simple while matching the model pipeline.
    #
    # Note: if you later add more advanced preprocessing, update this section.
    for col in feature_columns:
        # numeric if mostly numbers in the dataset is unknown here.
        # We'll try float input and fall back to text if parsing fails.
        if col.lower() in ["home id", "household size"]:
            # known numeric-ish
            input_values[col] = st.number_input(col, value=0.0)
        elif col.lower() in ["outdoor temperature (°c)", "outdoor temperature (°c) ", "outdoor temperature (°c)"]:
            input_values[col] = st.number_input(col, value=0.0)
        elif col.lower() in ["time"]:
            input_values[col] = st.text_input(col, value="12:00")
        elif col.lower() in ["date"]:
            input_values[col] = st.date_input(col, value=None)
            # Convert date_input to string if provided
            if input_values[col] is None:
                input_values[col] = "2023-01-01"
            else:
                input_values[col] = str(input_values[col])
        elif col.lower() in ["season"]:
            input_values[col] = st.selectbox(col, options=["Spring", "Summer", "Fall", "Winter"], index=0)
        else:
            # For everything else (e.g., Appliance Type, any other categorical)
            input_values[col] = st.text_input(col, value="Fridge")

    submitted = st.form_submit_button("Predict energy consumption")

if submitted:
    progress = st.progress(0)
    try:
        progress.progress(15, text="Preparing input row...")
        X = pd.DataFrame([input_values], columns=feature_columns)

        # Ensure no accidental target column exists
        if target_col in X.columns:
            X = X.drop(columns=[target_col])

        progress.progress(35, text="Running model pipeline...")
        preds = pipe.predict(X)

        out = X.copy()
        out[target_col] = preds

        progress.progress(100, text="Done")
        st.success("Prediction ready.")
        st.dataframe(out, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction row CSV",
            data=csv_bytes,
            file_name="prediction.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        progress.empty()



st.sidebar.header("Notes")
st.sidebar.markdown(
    "- This app uses a preprocessing + model pipeline saved as `energy_model.joblib`.\n"
    "- If you change the dataset schema, retrain the model."
)

