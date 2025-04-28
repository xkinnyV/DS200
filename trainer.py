import pyspark
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from transforms import Transforms
import numpy as np

class SparkConfig:
    appName = "MNIST"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 2

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split: str, 
                 spark_config: SparkConfig, 
                 transforms: Transforms) -> None:
        self.model = model
        self.split = split
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf, self.transforms)

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)
        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])
            df = self.sqlContext.createDataFrame(rdd, schema)
            
            # Chuyển đổi dữ liệu thành numpy array để huấn luyện CNN
            X = np.array(df.select("image").collect()).reshape(-1, 28, 28, 1)
            y = np.array(df.select("label").collect()).reshape(-1)
            
            # Huấn luyện mô hình CNN
            history = self.model.train(X, y)
            
            print("="*10)
            print(f"Loss: {history.history['loss'][-1]}")
            print(f"Accuracy: {history.history['accuracy'][-1]}")
            print("="*10)
        
        print("Total Batch Size of RDD Received:", rdd.count())
        print("+"*20)