"""
Train and serve a model for Food Delivery Delay prediction using real CSV data.

How to use:
1) Train and save model/scaler:
   python train_model.py --train
2) Start Flask API:
   python train_model.py --serve
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "orders_autumn_2020.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "item_count",
    "user_lat",
    "user_long",
    "venue_lat",
    "venue_long",
    "estimated_delivery_minutes",
    "cloud_coverage",
    "temperature",
    "wind_speed",
    "precipitation",
    "order_hour",
    "order_dayofweek",
    "distance_km",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"], errors="coerce")

    out["order_hour"] = out["TIMESTAMP"].dt.hour
    out["order_dayofweek"] = out["TIMESTAMP"].dt.dayofweek

    lat_delta = out["USER_LAT"] - out["VENUE_LAT"]
    long_delta = out["USER_LONG"] - out["VENUE_LONG"]
    out["distance_km"] = np.sqrt((lat_delta * 111.0) ** 2 + (long_delta * 55.8) ** 2)

    out = out.rename(
        columns={
            "ITEM_COUNT": "item_count",
            "USER_LAT": "user_lat",
            "USER_LONG": "user_long",
            "VENUE_LAT": "venue_lat",
            "VENUE_LONG": "venue_long",
            "ESTIMATED_DELIVERY_MINUTES": "estimated_delivery_minutes",
            "CLOUD_COVERAGE": "cloud_coverage",
            "TEMPERATURE": "temperature",
            "WIND_SPEED": "wind_speed",
            "PRECIPITATION": "precipitation",
            "ACTUAL_DELIVERY_MINUTES - ESTIMATED_DELIVERY_MINUTES": "delay_minutes",
            "ACTUAL_DELIVERY_MINUTES": "actual_delivery_minutes",
        }
    )

    return out


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = add_engineered_features(df)

    required_cols = FEATURE_COLUMNS + ["delay_minutes", "actual_delivery_minutes"]
    df = df.dropna(subset=required_cols)
    return df


def train_and_save_model() -> None:
    df = load_training_data()
    X = df[FEATURE_COLUMNS]
    y = df["delay_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model trained with {len(df)} rows")
    print(f"Test MAE (minutes): {mae:.3f}")
    print(f"Test R2 score: {r2:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")


app = Flask(__name__)
model = None
scaler = None


def load_artifacts():
    global model, scaler
    if model is None or scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("model.pkl/scaler.pkl not found. Run: python train_model.py --train")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)


def normalize_payload(data: dict):
    required = [
        "timestamp",
        "item_count",
        "user_lat",
        "user_long",
        "venue_lat",
        "venue_long",
        "estimated_delivery_minutes",
        "cloud_coverage",
        "temperature",
        "wind_speed",
        "precipitation",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing field: {key}")

    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    if pd.isna(ts):
        raise ValueError("timestamp must be a valid datetime string")

    vals = {k: float(data[k]) for k in required if k != "timestamp"}
    lat_delta = vals["user_lat"] - vals["venue_lat"]
    long_delta = vals["user_long"] - vals["venue_long"]
    distance_km = float(np.sqrt((lat_delta * 111.0) ** 2 + (long_delta * 55.8) ** 2))

    return [
        vals["item_count"],
        vals["user_lat"],
        vals["user_long"],
        vals["venue_lat"],
        vals["venue_long"],
        vals["estimated_delivery_minutes"],
        vals["cloud_coverage"],
        vals["temperature"],
        vals["wind_speed"],
        vals["precipitation"],
        float(ts.hour),
        float(ts.dayofweek),
        distance_km,
    ]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_artifacts()
        payload = request.get_json(force=True)
        features = normalize_payload(payload)
        X_scaled = scaler.transform(np.array([features]))

        predicted_delay = float(model.predict(X_scaled)[0])
        estimated = float(payload["estimated_delivery_minutes"])
        predicted_actual = estimated + predicted_delay

        return jsonify(
            {
                "predicted_delay_minutes": round(predicted_delay, 2),
                "predicted_actual_delivery_minutes": round(predicted_actual, 2),
                "status": "Delayed" if predicted_delay > 0 else "Earlier than estimate",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
    elif args.serve:
        app.run(host="0.0.0.0", port=5001, debug=True)
    else:
        parser.print_help()
