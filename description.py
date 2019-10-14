from pyspark.sql import SparkSession
import pandas as pd
from io import StringIO
import boto3

BUCKET = "s3a://us-east-1-prod-svcs-ml-cluster-emr-files"
BUCKET_BOTO = "us-east-1-prod-svcs-ml-cluster-emr-files"

spark = SparkSession.builder.master("yarn") \
    .config("hive.metastore.client.factory.class",
            "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory") \
    .enableHiveSupport().getOrCreate()

spark.catalog.setCurrentDatabase("datalake-processed")

# Load train x and y product ids.
# Call concat_x_y.py first
train_x_y = spark\
    .read.csv(f"{BUCKET}/fake_items_model/datasets/x_y_train.csv")\
    .selectExpr("_c0 AS product_id", "_c1 AS label")

products = spark.sql("SELECT product_id, description FROM product_create WHERE year='2019'")\
    .repartition(200).cache()

# Join train data with product_create based on product_id to get descriptions
train_x_y_desc = train_x_y\
    .join(products, on="product_id", how="left")\
    .dropDuplicates(subset=["product_id"])

# Flush to pandas data frame and write to local csv
df_pd = train_x_y_desc.toPandas()
df_pd.to_csv('X_train_desc.csv')

# Upload csv to S3
fs = s3fs.S3FileSystem(anon=True)
fs.put('X_train_desc.csv', BUCKET_BOTO + '/fake_items_model/datasets/X_traindesc.csv')
