# Heart Failure Prediction


### Introduction
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Requirement 

- OS - Ubuntu 18.04
- Miniconda for linux


### Instructions
####Download miniconda using the following command:\
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
sh Miniconda3-latest-Linux-x86_64.sh \
source .bashrc 
  
####To create conda environment to support this use-case \
conda env create -f environment.yml
 
####To activate the conda environment \
conda activate Assignment 
  
####change directory to source directory \
'cd source/' 
 
####To train an ensemble model run, open command prompt and please run the below command, \
python main.py 
  
####To start the server on localhost, open commad prompt and please run the below command, \
uvicorn api:app --reload 
  
####open internet browser and get into localhost server 'http://127.0.0.1:8000/docs' 

Please click 'POST' menu 

Please press 'Try it out' button on the right side of the bar \
please paste the below sample json input in the request body 

{
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

Finally press Execute bar. 

You ca find the prediction result in Response Body as \
  {
    "Prediction": "1"
  }

#### Curl command
To predict using curl command, open commad prompt and please run the below command, 

  curl -X 'POST' 
  'http://127.0.0.1:8000/predict' 
  -H 'accept: application/json' 
  -H 'Content-Type: application/json' 
  -d '{
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
  }'


