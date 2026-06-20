import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



HERE = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("DATA_PATH", HERE / "smart_home_energy_consumption_large.csv"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", HERE))


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at: {path}. "
            "Place the CSV in this repo or set DATA_PATH env var."
        )



TARGET_COL_CANDIDATES = [
    "Energy Consumption (kWh)",
    "Energy Consumption(kWh)",
    "Energy Consumption",
    "energy_consumption",
]


def find_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find target column. Columns available: {list(df.columns)}")


def main():
    require_file(DATA_PATH, "Training dataset CSV")
    df = pd.read_csv(DATA_PATH)
    target_col = find_target_col(df)


    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = list(X.columns)


    # Identify numeric/categorical columns
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # A strong baseline for tabular regression
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features=None,
    )


    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== Evaluation ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "energy_model.joblib"
    payload = {
        "pipeline": pipe,
        "target_col": target_col,
        "feature_columns": feature_columns,
    }
    joblib.dump(payload, artifact_path)

    # Also write feature columns to JSON for transparency/debugging
    feature_json_path = ARTIFACT_DIR / "feature_columns.json"
    try:
        import json

        feature_json_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal: app can still rely on the joblib artifact
        pass

    print(f"Saved artifact: {artifact_path}")




if __name__ == "__main__":
    main()

