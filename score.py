# Databricks notebook source


# COMMAND ----------

pip install azureml.core

# COMMAND ----------

import pickle
import pandas as pd
from mlflow.tracking import MlflowClient
import json
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import azureml
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
import utils
#from utils import mylib

# COMMAND ----------

client = MlflowClient()

tmp_path = client.download_artifacts(run_id="4b29842fc6024642aa3e30e66c55eea1", path='model/model.pkl')

f = open(tmp_path,'rb')

model = pickle.load(f)


# COMMAND ----------

# Use the Pickle file 
accuracy_pkl = pickle_model.score(X_tfidf_test, y_tfidf_test)
accuracy_model = model.predict(X_tfidf_test, y_tfidf_test)
print(accuracy_pkl == accuracy_model)
output = True

# COMMAND ----------

def init():
    global model
    client = MlflowClient()
    tmp_path = client.download_artifacts(run_id="4b29842fc6024642aa3e30e66c55eea1", path='model/model.pkl')
    with open(temp_path, 'rb') as file:
        model = pickle.load()
    
    # For demonstration purposes only
    print(mylib.get_alphas())

input_sample = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
output_sample = np.array([3726.995])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
