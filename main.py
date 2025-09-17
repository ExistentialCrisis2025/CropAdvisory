from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

DATA_FILE = "Crop_recommendation.csv"
MODEL_FILE = "crop_model.pkl"
WEATHER_API_KEY = "5a780c43787742e794f210032251609"

app = Flask(__name__)

def train_and_save(df):
    # Encode categorical features
    le_city = LabelEncoder()
    le_soil = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    if "city" in df.columns and "soil_type" in df.columns and "season" in df.columns:
        df["city_enc"] = le_city.fit_transform(df["city"])
        df["soil_enc"] = le_soil.fit_transform(df["soil_type"])
        df["season_enc"] = le_season.fit_transform(df["season"])
    else:
        # if dataset has only Kaggle fields
        df["city_enc"] = 0
        df["soil_enc"] = 0
        df["season_enc"] = 0
    df["crop_enc"] = le_crop.fit_transform(df["label"])

    feature_cols = [col for col in df.columns if col not in ["label", "crop_enc"]]
    if "crop_enc" in df.columns:
        feature_cols.remove("crop_enc")

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall",
            "city_enc", "soil_enc", "season_enc", "month"]] \
        if "month" in df.columns else df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

    y = df["crop_enc"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump({
        "model": model,
        "le_city": le_city,
        "le_soil": le_soil,
        "le_season": le_season,
        "le_crop": le_crop
    }, MODEL_FILE)

def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
    r = requests.get(url)
    data = r.json()
    temp = data["current"]["temp_c"]
    humidity = data["current"]["humidity"]
    rainfall = data["current"].get("precip_mm", 0.0)
    return temp, humidity, rainfall

@app.route("/train", methods=["POST"])
def train():
    df = pd.read_csv(DATA_FILE)
    train_and_save(df)
    return jsonify({"message": "Model trained and saved successfully."})

@app.route("/predict", methods=["POST"])
def predict():
    data = joblib.load(MODEL_FILE)
    model = data["model"]
    le_city = data["le_city"]
    le_soil = data["le_soil"]
    le_season = data["le_season"]
    le_crop = data["le_crop"]

    req = request.get_json()
    city = req.get("city", "").lower()
    soil_type = req.get("soil_type", "").lower()
    month = int(req.get("month", 6))

    # Encode city & soil safely
    try:
        city_enc = le_city.transform([city])[0]
    except:
        city_enc = 0
    try:
        soil_enc = le_soil.transform([soil_type])[0]
    except:
        soil_enc = 0

    # Determine season
    season = "Kharif" if month in [6,7,8,9] else \
             "Rabi" if month in [11,12,1,2] else \
             "Zaid" if month in [3,4,5] else "Annual"
    try:
        season_enc = le_season.transform([season])[0]
    except:
        season_enc = 0

    # Weather API
    temp, humidity, rainfall = get_weather(city)

    # Approx soil nutrients
    N, P, K, ph = 80, 40, 40, 6.5

    features = np.array([[N, P, K, temp, humidity, ph, rainfall,
                          city_enc, soil_enc, season_enc, month]])

    pred_enc = model.predict(features)[0]
    prediction = le_crop.inverse_transform([pred_enc])[0]

    return jsonify({
        "city": city,
        "soil_type": soil_type,
        "month": month,
        "season": season,
        "weather": {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall
        },
        "recommended_crop": prediction
    })

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        df = pd.read_csv(DATA_FILE)
        train_and_save(df)
    app.run(host="0.0.0.0", port=5000)
