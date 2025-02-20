# fastapi_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="My Awesome API", description="A simple example API with Swagger", version="1.0"
)

# Load the model
model = joblib.load('rainfall_model.joblib')

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float

@app.post('/predict')
def predict(data: WeatherData):
    features = [data.temperature, data.humidity, data.wind_speed]
    prediction = model.predict([features])[0]
    return {'predicted_rainfall': prediction,'unit': 'mm'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5500)