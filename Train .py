# Databricks notebook source
dbutils.fs.ls("/FileStore/df/")


# COMMAND ----------

df = spark.read.csv("/FileStore/df/df3.csv")

# COMMAND ----------

display(df)

# COMMAND ----------

df_pandas=df.toPandas()

# COMMAND ----------

df_pandas.head(5)

# COMMAND ----------

import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

# COMMAND ----------

def train_claim_insurance(data):
  
  # Evaluate metrics
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

  np.random.seed(40)

  # Split the data into training and test sets. (0.75, 0.25) split.
  train, test = train_test_split(data)

  # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
  train_x = train.drop(["_c8"], axis=1)
  test_x = test.drop(["_c8"], axis=1)
  train_y = train[["_c8"]]
  test_y = test[["_c8"]]

    
  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run():
    lr = LinearRegression(fit_intercept=True)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log mlflow attributes for mlflow UI
    #mlflow.log_param("alpha", alpha)
    #mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
    modelpath = "/dbfs/mlflow/test_insurance/model-%f-%f" % (rmse,r2)
    mlflow.sklearn.save_model(lr, modelpath)
    

# COMMAND ----------

train_claim_insurance(df_pandas)

# COMMAND ----------


