from fastapi import FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import joblib
import numpy as np

app = FastAPI(
    title="Rainfall Prediction API",
    description="This API predicts rainfall based on weather data.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "predictions",
            "description": "Operations with rainfall predictions",
        },
    ]
)

# Load the model
model = joblib.load('rainfall_model.joblib')

# Define scaling factors
TEMP_SCALE = 40
HUMIDITY_SCALE = 100
WIND_SCALE = 20
RAINFALL_SCALE = 10

# In-memory storage for demonstration purposes
predictions = {}


class WeatherData(BaseModel):
    temperature: float = Field(..., gt=-50, lt=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, description="Wind speed in km/h")

    class Config:
        schema_extra = {
            "example": {
                "temperature": 25.5,
                "humidity": 60,
                "wind_speed": 10.3
            }
        }


class PredictionUpdate(BaseModel):
    rainfall: float = Field(..., ge=0, description="Updated rainfall prediction in mm")


class Prediction(BaseModel):
    id: int
    rainfall: float
    unit: str


class Unit(str, Enum):
    mm = "mm"
    inches = "inches"


@app.post("/predict", response_model=Prediction, status_code=201, tags=["predictions"])
async def predict(data: WeatherData):
    """
    Create a new rainfall prediction based on weather data.
    - **temperature**: Temperature in Celsius
    - **humidity**: Humidity percentage
    - **wind_speed**: Wind speed in km/h
    """
    # Preprocess inputs
    scaled_temp = data.temperature / TEMP_SCALE
    scaled_humidity = data.humidity / HUMIDITY_SCALE
    scaled_wind = data.wind_speed / WIND_SCALE
    features = [scaled_temp, scaled_humidity, scaled_wind]

    # Get prediction
    scaled_prediction = model.predict([features])[0]
    prediction_mm = scaled_prediction * RAINFALL_SCALE

    # Store prediction
    prediction_id = len(predictions) + 1
    prediction = Prediction(
        id=prediction_id,
        rainfall=round(prediction_mm, 2),
        unit='mm'
    )
    predictions[prediction_id] = prediction
    return prediction


@app.get("/predict/{prediction_id}", response_model=Prediction, tags=["predictions"])
async def get_prediction(
        prediction_id: int = Path(..., title="The ID of the prediction to retrieve"),
        unit: Optional[Unit] = Query(None, description="Unit for rainfall measurement")
):
    """
    Retrieve a specific rainfall prediction by its ID.
    - **prediction_id**: The ID of the prediction to retrieve
    - **unit**: Optional unit for rainfall measurement (mm or inches)
    """
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    prediction = predictions[prediction_id]
    if unit == Unit.inches:
        prediction.rainfall = round(prediction.rainfall / 25.4, 2)
        prediction.unit = "inches"
    return prediction


@app.put("/predict/{prediction_id}", response_model=Prediction, tags=["predictions"])
async def update_prediction(
        prediction_id: int = Path(..., title="The ID of the prediction to update"),
        update: PredictionUpdate = Body(..., description="Updated rainfall value")
):
    """
    Update an existing rainfall prediction.
    - **prediction_id**: The ID of the prediction to update
    - **rainfall**: The new rainfall value in mm
    """
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    predictions[prediction_id].rainfall = update.rainfall
    return predictions[prediction_id]


@app.delete("/predict/{prediction_id}", status_code=204, tags=["predictions"])
async def delete_prediction(
        prediction_id: int = Path(..., title="The ID of the prediction to delete")
):
    """
    Delete a specific rainfall prediction.
    - **prediction_id**: The ID of the prediction to delete
    """
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    del predictions[prediction_id]


@app.get("/predictions", response_model=List[Prediction], tags=["predictions"])
async def list_predictions(
        skip: int = Query(0, ge=0, description="Number of predictions to skip"),
        limit: int = Query(10, ge=1, le=100, description="Maximum number of predictions to return")
):
    """
    List all rainfall predictions with pagination.
    - **skip**: Number of predictions to skip (for pagination)
    - **limit**: Maximum number of predictions to return (for pagination)
    """
    return list(predictions.values())[skip: skip + limit]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5500)