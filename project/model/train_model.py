"""
Beginner-friendly ML training + Flask API for:
Food Delivery Time Delay Predictor

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a simple synthetic dataset for delay prediction."""
    rng = np.random.default_rng(seed)

    distance = rng.uniform(1, 20, n_samples)  # km
    prep_time = rng.uniform(5, 45, n_samples)  # minutes
    traffic = rng.choice(["Low", "Medium", "High"], n_samples, p=[0.35, 0.4, 0.25])
    weather = rng.choice(["Clear", "Rainy"], n_samples, p=[0.75, 0.25])
    peak_hour = rng.choice(["No", "Yes"], n_samples, p=[0.6, 0.4])

    # Convert categories to numeric influence values for label generation
    traffic_score = np.select([traffic == "Low", traffic == "Medium", traffic == "High"], [0, 1, 2])
    weather_score = np.where(weather == "Rainy", 1, 0)
    peak_score = np.where(peak_hour == "Yes", 1, 0)

    # A simple rule + noise to form a realistic-ish target
    delay_signal = (
        0.30 * distance
        + 0.45 * prep_time
        + 8.0 * traffic_score
        + 6.0 * weather_score
        + 5.0 * peak_score
        + rng.normal(0, 4, n_samples)
    )

    threshold = 30
    delayed = (delay_signal > threshold).astype(int)

    df = pd.DataFrame(
        {
            "distance_km": distance,
            "prep_time_min": prep_time,
            "traffic_level": traffic,
            "weather": weather,
            "peak_hour": peak_hour,
            "delayed": delayed,
        }
    )
    return df


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features in a very simple way."""
    out = df.copy()
    out["traffic_level"] = out["traffic_level"].map({"Low": 0, "Medium": 1, "High": 2})
    out["weather"] = out["weather"].map({"Clear": 0, "Rainy": 1})
    out["peak_hour"] = out["peak_hour"].map({"No": 0, "Yes": 1})
    return out


# ------------------------------------------------------------
# Train and save model
# ------------------------------------------------------------
def train_and_save_model() -> None:
    df = generate_synthetic_data()
    df = preprocess_features(df)

    X = df[["distance_km", "prep_time_min", "traffic_level", "weather", "peak_hour"]]
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model trained. Test accuracy: {accuracy:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")


# ------------------------------------------------------------
# Flask API
# ------------------------------------------------------------
app = Flask(__name__)
model = None
scaler = None


def load_artifacts():
    global model, scaler
    if model is None or scaler is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError(
                "model.pkl/scaler.pkl not found. Run: python train_model.py --train"
            )
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)


def normalize_payload(data: dict):
    """Validate and normalize input payload from Node backend."""
    required = ["distance", "prep_time", "traffic", "weather", "peak_hour"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing field: {key}")

    try:
        distance = float(data["distance"])
        prep_time = float(data["prep_time"])
    except Exception as e:
        raise ValueError("distance and prep_time must be numeric") from e

    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    weather_map = {"Clear": 0, "Rainy": 1}
    peak_map = {"No": 0, "Yes": 1}

    traffic = data["traffic"]
    weather = data["weather"]
    peak_hour = data["peak_hour"]

    if traffic not in traffic_map:
        raise ValueError("traffic must be one of: Low, Medium, High")
    if weather not in weather_map:
        raise ValueError("weather must be one of: Clear, Rainy")
    if peak_hour not in peak_map:
        raise ValueError("peak_hour must be one of: Yes, No")

    return [distance, prep_time, traffic_map[traffic], weather_map[weather], peak_map[peak_hour]]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_artifacts()
        payload = request.get_json(force=True)
        features = normalize_payload(payload)
        X = np.array([features])
        X_scaled = scaler.transform(X)

        pred = int(model.predict(X_scaled)[0])
        proba_delayed = float(model.predict_proba(X_scaled)[0][1])

        return jsonify(
            {
                "prediction": pred,
                "label": "Delayed" if pred == 1 else "On Time",
                "probability_delayed": round(proba_delayed, 4),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model and save artifacts")
    parser.add_argument("--serve", action="store_true", help="Start Flask API")
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
    elif args.serve:
        # Host 0.0.0.0 so it's easy to access in local/dev containers.
        app.run(host="0.0.0.0", port=5001, debug=True)
    else:
        parser.print_help()
