import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st



st.set_page_config(page_title="Smart Home Energy Consumption", layout="wide")


st.title("Smart Home Energy Consumption Predictor")
st.write(
    "Upload a CSV of smart home features. The app will predict **Energy Consumption (kWh)** using a trained ML model."
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


uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Loading CSV..."):
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Drop target column if user accidentally included it
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    st.subheader("Predict")

    predict_btn = st.button("Predict energy consumption")
    if predict_btn:
        progress = st.progress(0)
        try:
            progress.progress(20, text="Running model pipeline...")
            preds = pipe.predict(X)
            progress.progress(70, text="Preparing results...")

            out = df.copy()
            out[target_col] = preds

            progress.progress(100, text="Done")
            st.success("Predictions ready.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            progress.empty()
else:
    st.info("Upload your dataset CSV to get predictions.")


st.sidebar.header("Notes")
st.sidebar.markdown(
    "- This app uses a preprocessing + model pipeline saved as `energy_model.joblib`.\n"
    "- If you change the dataset schema, retrain the model."
)

