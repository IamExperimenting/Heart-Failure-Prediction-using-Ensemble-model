## importing required python modules
import pandas as pd,warnings
from configparser import ConfigParser
warnings.filterwarnings('ignore')
from model import Modelling

## reading config.ini file
parser = ConfigParser()
parser.read('config.ini')
training_data = parser.get('input_data','training_data')
data = pd.read_csv(training_data)
## importing the Modelling module
ensemble_start = Modelling(data)
## data split
ensemble_start = ensemble_start.data_split(data)
## generating accuracy for base classifier
ensemble_start.base_classifier()
## generating featue importance plot
ensemble_start.feature_importance()
## training the ensemble model
ensemble_start.ensemble_model()