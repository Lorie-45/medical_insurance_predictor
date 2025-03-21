from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()

model = joblib.load("insurance_rf_model.joblib")


class InsuranceData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# Preprocessing function
def preprocess(data: InsuranceData):
    # Binary encoding
    sex = 1 if data.sex.lower() == "male" else 0
    smoker = 1 if data.smoker.lower() == "yes" else 0

    # One-hot encoding for region
    region_northwest = 1 if data.region.lower() == "northwest" else 0
    region_northeast = 1 if data.region.lower() == "northeast" else 0
    region_southeast = 1 if data.region.lower() == "southeast" else 0
    region_southwest = 1 if data.region.lower() == "southwest" else 0

    # Construct the input array
    input_data = np.array(
        [
            [
                data.age,
                sex,
                data.bmi,
                data.children,
                smoker,
                region_northwest,
                region_northeast,
                region_southeast,
                region_southwest,
            ]
        ]
    )
    return input_data



# Prediction endpoint
@app.post("/predict")
def predict(data: InsuranceData):
    # Preprocess the input data
    input_data = preprocess(data)
    # Make prediction
    prediction = model.predict(input_data)
    # Return the prediction
    return {"predicted_charges": prediction[0]}