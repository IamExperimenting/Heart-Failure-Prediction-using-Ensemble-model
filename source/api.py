import uvicorn,joblib,os,pandas as pd 
from fastapi import FastAPI,Header, HTTPException
from variables import Features
from fastapi.testclient import TestClient
from configparser import ConfigParser

parser = ConfigParser()
parser.read('config.ini')

model_output_path = parser.get('output_path','pickle_file')
ensemble_model = joblib.load(model_output_path)

app = FastAPI()

@app.get('/')
async def index():
    return {"msg": "Assignment Page"}

@app.post('/predict')
async def predict(data:Features):
    raw_data = data.dict()
    predict_input = pd.DataFrame([raw_data])
    prediction = ensemble_model.predict(predict_input)[0]
    return {
        "Prediction" : str(prediction)
    }

if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)