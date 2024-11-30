from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

# Инициализация FastAPI

app = FastAPI()

# Загрузка модели, скейлера и признаков из файла
with open("lasso_model_with_scaler.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]        # Lasso-модель
scaler = model_data["scaler"]      # Стандартизатор
FEATURES = model_data["features"]  # Список признаков

# Класс для описания одного объекта
class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float

# Класс для коллекции объектов
class Items(BaseModel):
    objects: List[Item]

# Эндпоинт для предсказания по одному объекту
@app.post("/predict_item")

def predict_item(item: Item) -> dict:
    try:
        # Преобразование объекта в DataFrame
        input_data = pd.DataFrame([item.dict()])

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(input_data.columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Пропущены признаки. Ожидаются: {FEATURES}"
            )

        # Применение стандартизации
        input_scaled = scaler.transform(input_data[FEATURES])

        # Предсказание
        prediction = model.predict(input_scaled)[0]
        return {"predicted_price": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для предсказания по коллекции объектов
@app.post("/predict_items")

def predict_items(items: Items) -> dict:
    try:
        # Преобразование коллекции объектов в DataFrame
        input_data = pd.DataFrame([item.dict() for item in items.objects])

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(input_data.columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Пропущены признаки. Ожидаются: {FEATURES}"
            )

        # Применение стандартизации
        input_scaled = scaler.transform(input_data[FEATURES])

        # Предсказания
        predictions = model.predict(input_scaled)
        return {"predicted_prices": [round(float(pred), 2) for pred in predictions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для обработки файла CSV
@app.post("/predict_csv")

def predict_csv(file: UploadFile):
    try:
        # Чтение данных из CSV-файла
        df = pd.read_csv(file.file)

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(df.columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Некорректный файл. Ожидаются признаки: {FEATURES}"
            )

        # Применение стандартизации
        input_scaled = scaler.transform(df[FEATURES])

        # Предсказания
        predictions = model.predict(input_scaled)

        # Добавление предсказаний в DataFrame
        df["predicted_price"] = [round(float(pred), 2) for pred in predictions]

        # Сохранение результата в новый файл
        output_file = "predictions.csv"
        df.to_csv(output_file, index=False)

        return {"message": "Файл обработан успешно. Скачайте результат.", "output_file": output_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))