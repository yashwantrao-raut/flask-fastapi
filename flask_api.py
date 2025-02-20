# flask_api.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('rainfall_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['temperature'], data['humidity'], data['wind_speed']]
    prediction = model.predict([features])[0]
    return jsonify({'predicted_rainfall': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)