from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Carregando o arquivo
model = joblib.load('/content/drive/  sample_data.pkl')

app = FastAPI()

class Item(BaseModel):
    # Defina a estrutura dos dados de entrada na API
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(item: Item):
    # Converta os dados de entrada para um DataFrame
    input_data = pd.DataFrame([item.dict()])

    # Faça a previsão usando o modelo
    prediction = model.predict(input_data)

    # Converta o resultado de volta para uma string (ou outro formato desejado)
    result = prediction[0]

    return {"prediction": result}
