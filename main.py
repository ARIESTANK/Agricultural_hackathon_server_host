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


# @app.route("/today_predict", methods=["POST"])
# def today_predict():
#     try:
#         data = request.json
#         if "features" not in data:
#             return jsonify({"status": "error", "message": "Missing 'features' key"}), 400

#         features = np.array(data["features"]).reshape(1, -1)

#         # Apply scaling if exists
#         if scaler:
#             features = scaler.transform(features)

#         # Check if feature count matches model
#         if features.shape[1] != todayPredictModel.n_features_in_:
#             return jsonify({
#                 "status": "error",
#                 "message": f"X has {features.shape[1]} features, but model expects {todayPredictModel.n_features_in_}."
#             }), 400

#         # Make prediction
#         prediction = todayPredictModel.predict(features)[0]
#         result = "Rain" if prediction == 1 else "No Rain"

#         # Return as JSON (React can parse this easily)
#         return jsonify({
#             "status": "success",
#             "prediction": int(prediction),
#             "result": result
#         }), 200

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500
    
# # @app.route("/tomorrow_predict", methods=["POST"])

@app.route("/rainTest", methods=['GET'])
def raintest():
    # have to get the location via maps

    lan=15.299396346225851 
    lon=90.30703691717518
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
        "today": today_result,
        "tomorrow": tomorrow_result
    }), 200

if __name__ == "__main__":
    app.run(debug=True)