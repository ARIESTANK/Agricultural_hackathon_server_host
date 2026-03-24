from flask import Flask, request, jsonify
import joblib
import numpy as np
import function
import requests

# print(f"Degree is - {function.deg_to_index(90)}")

app = Flask(__name__, static_folder="../frontend/agrimyanmar/build")

# Load scaler if used
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

@app.route("/")
def home():
    return "Rain Prediction API is running!"

@app.route("/rainTest", methods=['POST'])
def raintest():

    data = request.get_json()

    lan = data.get("lat")
    lon = data.get("lon")


    apiData=function.weatherApiCall(lan,lon) #function call to get api weather data
    
    data =  [
            apiData['temp_min'],
            apiData['temp_max'],
            apiData['windDir'],
            apiData['temp'],
            apiData['humidity'],
            apiData['windSpeed'],
            apiData['pressure']
        
    ]
    testRainData = [
        24.0,   # MinTemp (warm night)
        28.0,   # MaxTemp (not too hot → cloudy)
        8,      # WindDir (S → often moist air)
        26.0,   # temp (current)
        85.0,   # humidity (VERY high → rain likely)
        30.0,   # windspeed (km/h → active weather)
        1002.0  # pressure (LOW → storm/rain)
    ]

    # Call today/tomorrow prediction functions
    today_result = function.predict_today(data)
    tomorrow_result = function.predict_tomorrow(data)

    # Return both results together
    return jsonify({
        "status": "success",
        "weatherData":{
            'temp_max':apiData['temp_max'],
            'temp_min':apiData['temp_min'],
            'windDir':apiData['windDir'],
            'temp':apiData['temp'],
            'humidity':apiData['humidity'],
            'windSpeed':apiData['windSpeed'],
            'pressure':apiData['pressure']
            },
        "today": today_result,
        "tomorrow": tomorrow_result
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)