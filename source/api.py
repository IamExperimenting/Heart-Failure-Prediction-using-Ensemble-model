import uvicorn,joblib,os,pandas as pd 
from fastapi import FastAPI,Header, HTTPException
from variables import Features
from fastapi.testclient import TestClient
from configparser import ConfigParser
from pydantic import BaseModel

parser = ConfigParser()
parser.read('config.ini')

secret_token = "fastapitest"

dummy_json = {
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000.00,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
}

model_output_path = parser.get('output_path','pickle_file')
ensemble_model = joblib.load(model_output_path)

app = FastAPI()

class Features(BaseModel):
    age : float
    anaemia : int
    creatinine_phosphokinase : int
    diabetes : int
    ejection_fraction : int
    high_blood_pressure : int
    platelets : float
    serum_creatinine : float
    serum_sodium : int
    sex : int
    smoking : int
    time : int

@app.get("/test/{item_id}", response_model=Features)
async def read_main(item_id: str, x_token: str = Header(...)):
    if x_token != secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item_id not in dummy_json:
        raise HTTPException(status_code=404, detail="Item not found")
    return dummy_json[item_id]


@app.post("/test/", response_model=Features)
async def create_item(item: Features, x_token: str = Header(...)):
    if x_token != secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item.age in dummy_json:
        raise HTTPException(status_code=400, detail="Item already exists")
    dummy_json[item.age] = item
    return item

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
