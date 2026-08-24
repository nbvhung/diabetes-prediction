from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Diabetes Prediction API - Chapter 12")

@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'web.html'))

# Load model và scaler đã huấn luyện ở các bước trước
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'diabetes_model.sav'), 'rb'))
scaler = pickle.load(open(os.path.join(os.path.dirname(__file__), 'scaler.sav'), 'rb'))

class PatientInput(BaseModel):
    pregnancies: float = 6.0
    glucose: float = 148.0
    blood_pressure: float = 72.0
    skin_thickness: float = 35.0
    insulin: float = 0.0
    bmi: float = 33.6
    diabetes_pedigree: float = 0.627
    age: float = 50.0

@app.post("/predict")
def predict_diabetes(data: PatientInput):
    # Tạo vector đầu vào 8 đặc trưng theo đúng thứ tự dataset
    features = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]])

    # Scale dữ liệu với scaler đã lưu
    features_scaled = scaler.transform(features)

    # Dự đoán lớp (0 hoặc 1) và xác suất
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]

    label_text = "Dương tính (Có nguy cơ Tiểu đường)" if prediction == 1 else "Âm tính (Bình thường)"
    confidence = round(float(probabilities[prediction]) * 100, 2)

    return {
        "prediction": prediction,
        "label": label_text,
        "confidence_score": confidence,
        "message": f"Chẩn đoán: {label_text} (Xác suất: {confidence}%)"
    }
