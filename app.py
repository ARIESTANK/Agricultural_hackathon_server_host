from flask import Flask, request, jsonify
import joblib
import function
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# 🔐 Use environment variable (IMPORTANT)
SECRET_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=SECRET_KEY)

# Load scaler if used
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

@app.route("/")
def home():
    return "Rain + AI Chat API running!"

# 🌧️ Rain Prediction
@app.route("/rainTest", methods=['POST'])
def raintest():
    data = request.get_json()

    lat = data.get("lat")
    lon = data.get("lon")

    apiData = function.weatherApiCall(lat, lon)

    input_data = [
        apiData['temp_min'],
        apiData['temp_max'],
        apiData['windDir'],
        apiData['temp'],
        apiData['humidity'],
        apiData['windSpeed'],
        apiData['pressure']
    ]

    today_result = function.predict_today(input_data)
    tomorrow_result = function.predict_tomorrow(input_data)
    rainData = function.predict_rainfall(input_data)

    return jsonify({
        "status": "success",
        "weatherData": apiData,
        "today": today_result,
        "rainFall": rainData,
        "tomorrow": tomorrow_result
    }), 200


# 🤖 Chatbot (Groq)
@app.route("/chatBot", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a climate-smart farming assistant for Myanmar farmers. Please generate the response for short and perfect"
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        return jsonify({
            "reply": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)