# Databricks notebook source
# Unmount a mount point
dbutils.fs.unmount("/mnt/firstml2553918991")

# COMMAND ----------

##url = "wasbs://" + ContainerName + "@" + storageAccountName + ".blob.core.windows.net/"
dbutils.fs.mount(
source = "wasbs://input@firstml2553918991.blob.core.windows.net/",
mount_point = "/mnt/firstml2553918991/",
#config = "fs.azure.sas." + ContainerName + "." + storageAccountName + ".blob.core.windows.net" + ":" + sas
extra_configs = {"fs.azure.sas.input.firstml2553918991.blob.core.windows.net":"?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2022-05-18T02:03:38Z&st=2022-05-17T18:03:38Z&spr=https&sig=xVUfkMG%2FEDP2c%2BVqY0mw5%2FxRnYhKoqB3KDMMlkfc1Vw%3D"})

# COMMAND ----------

display(dbutils.fs.ls("mnt/firstml2553918991/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Imputer

# COMMAND ----------

csvFile = "dbfs:/mnt/firstml2553918991/healthinsurance.csv"

csvSchema = StructType([
StructField("age",IntegerType(),True),
StructField("sex",StringType(),True),
StructField("weight",FloatType(),True),
StructField("bmi",FloatType(),True),
StructField("hereditary_diseases",StringType(),True),
StructField("no_of_dependents",IntegerType(),True),
StructField("smoker",IntegerType(),True),
StructField("city",StringType(),True),
StructField("bloodpressure",IntegerType(),True),
StructField("diabetes",IntegerType(),True),
StructField("regular_ex",IntegerType(),True),
StructField("job_title",StringType(),True),
StructField("claim",FloatType(),True)
])

csvDF = (spark.read.option('header','true')
.schema(csvSchema)
.csv(csvFile))

# COMMAND ----------

display(csvDF)

# COMMAND ----------

def preprocessing(x):
    imputer = Imputer(
    inputCols=x['age','bmi'].columns,outputCols=["{}_imputed".format(c) for c in x['age','bmi'].columns]).setStrategy("median")

# Add imputation cols to df
    df = imputer.fit(x).transform(x)  # Imputing missing values with median in age and bmi column
    df1=df.drop('age','bmi') # drop original age and bmi (as imputed ones got created)
    df2 = df1.filter(df1.bloodpressure != 0) # drop the rows having bp as 0
    from pyspark.sql.functions import when
    df3 = df2.withColumn("sex", when(df2.sex == "male",0)\
      .when(df2.sex == "female",1) \
      .otherwise(df2.sex))
    df3=df3.withColumn("sex",df3.sex.cast('integer'))  # make them integer
    df4 = df3.drop('city','job_title') # Feature selection
    df5=df4
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Alzheimer','1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Arthritis', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Cancer', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Diabetes', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Epilepsy', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'EyeDisease', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'HeartDisease', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'High BP', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'Obesity', '1'))
    df5 = df5.withColumn('hereditary_diseases', regexp_replace('hereditary_diseases', 'NoDisease', '0'))
    df5 = df5.withColumn("hereditary_diseases",df5.hereditary_diseases.cast('integer'))
    df5 = df5.withColumn("bloodpressure",df5.bloodpressure.cast('double'))
    #df5_pandas=df5.toPandas()
    return df5

# COMMAND ----------

processed=preprocessing(csvDF)

# COMMAND ----------

display(processed)

# COMMAND ----------

processed.head(5)

# COMMAND ----------

processed.coalesce(1).write.format("com.databricks.spark.csv").option("header", "False").save("dbfs:/FileStore/df/df3.csv")

# COMMAND ----------


