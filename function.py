import requests
from flask import jsonify
import pandas as pd
import numpy as np
import joblib

#models

todayPredictModel = joblib.load("today_rain_predict.pkl")
tomorrowPredictModel=joblib.load("tomorrow_rain_predict.pkl")
rainFallModel=joblib.load("rainFall_predict.pkl")

# Load scaler if used
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None


def deg_to_index(deg):
    return int((deg + 11.25) // 22.5) % 16

def weatherApiCall(lan,lon):
    weatherApi=f"https://api.openweathermap.org/data/2.5/weather?lat={lan}&lon={lon}&appid=dd5dbf7b2cfed5c0a6c17be4d15aba12&units=metric"
    windApi=f"https://api.open-meteo.com/v1/forecast?latitude={lan}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,cloud_cover,precipitation"
    weatherRes=requests.get(weatherApi)
    windRes=requests.get(windApi)
    weatherJsonData=weatherRes.json()
    windJsonData=windRes.json()
    # return jsonify(windRes.json())
    
    result= ({
        'pressure':weatherJsonData['main']['pressure'],
        'humidity':weatherJsonData['main']['humidity'],
        'temp':weatherJsonData['main']['temp'],
        'temp_min':weatherJsonData['main']['temp_min'],
        'temp_max':weatherJsonData['main']['temp_max'],
        'windDir':deg_to_index(weatherJsonData['wind']['deg']),
        'windSpeed':windJsonData["current"]["wind_speed_10m"],
        })
    
    return result


def predict_tomorrow(data, model=tomorrowPredictModel, scaler=scaler):
    # Extract features list
    if isinstance(data, dict) and "features" in data:
        features = data["features"]
    elif isinstance(data, list):
        features = data
    else:
        raise ValueError("Input must be a dict with 'features' key or a list of features")

    # Convert to NumPy array and reshape
    X = np.array(features).reshape(1, -1)

    # Apply scaling if available
    if scaler:
        X = scaler.transform(X)

    # Check feature count
    if X.shape[1] != model.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but model expects {model.n_features_in_}")

    # Make prediction
    pred = model.predict(X)[0]
    result = "Rain" if pred == 1 else "No Rain"
    return {"prediction": int(pred), "result": result}

def predict_today(data, model=todayPredictModel, scaler=scaler):
    # Extract features list
    if isinstance(data, dict) and "features" in data:
        features = data["features"]
    elif isinstance(data, list):
        features = data
    else:
        raise ValueError("Input must be a dict with 'features' key or a list of features")

    # Convert to NumPy array and reshape
    X = np.array(features).reshape(1, -1)

    # Apply scaling if available
    if scaler:
        X = scaler.transform(X)

    # Check feature count
    if X.shape[1] != model.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but model expects {model.n_features_in_}")

    # Make prediction
    pred = model.predict(X)[0]
    result = "Rain" if pred == 1 else "No Rain"
    return {"prediction": int(pred), "result": result}


def predict_rainfall(data, model=rainFallModel, scaler=scaler):
    # Extract features list
    if isinstance(data, dict) and "features" in data:
        features = data["features"]
    elif isinstance(data, list):
        features = data
    else:
        raise ValueError("Input must be a dict with 'features' key or a list of features")

    # Convert to NumPy array and reshape
    X = np.array(features).reshape(1, -1)

    # Apply scaling if available
    if scaler:
        X = scaler.transform(X)

    # Check feature count
    if X.shape[1] != model.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but model expects {model.n_features_in_}")

    # Make prediction
    pred = model.predict(X)[0]
    return pred