from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext, SparkConf
import pandas as pd
import pickle
import os

# conf = SparkConf().setAppName("HandSignRecognition").setMaster("spark://houyudeMacBook-Pro.local:7077")
conf = SparkConf().setAppName("HandSignRecognition").setMaster("spark://houyudeMBP.lan:7077")
sc = SparkContext(conf=conf)

# spark = SparkSession.builder.master("local[*]").getOrCreate()
spark = SparkSession.builder.getOrCreate()
os.environ['PYSPARK_PYTHON'] = "/opt/anaconda3/bin/python"

# Load training data
dataset_train = pickle.load(open('/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl.pkl', 'rb'))
# dataset_train = sc.parallelize(dataset_train, numSlices=1000)

dataset_test =  pickle.load(open('/Users/yuneshou/Desktop/2022winter/CS219/sign-language-recognition/ik_annotated/kaggle_asl_test.pkl', 'rb'))
# dataset_train = pickle.load(open('/Users/zuerwang/Desktop/CS219/sign-language-recognition/ik_annotated/kaggle_asl.pkl', 'rb'))[:10000]
# dataset_test =  pickle.load(open('/Users/zuerwang/Desktop/CS219/sign-language-recognition/ik_annotated/kaggle_asl_test.pkl', 'rb'))
dataset_train['Label'] = dataset_train['Label'].apply(lambda x: ord(x) - 65)
dataset_test['Label'] = dataset_test['Label'].apply(lambda x: ord(x) - 65)

dataset_train = spark.createDataFrame(dataset_train)
assembler = VectorAssembler(
    inputCols=['LeftHand',
                   'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
                   'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
                   'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
                   'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
                   'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
                   'LeftHandIndex1_rx', 'LeftHandMiddle1_rx', 'LeftHandRing1_rx', 'LeftHandPinky1_rx',
                   'LeftHandThumb1_rx', 'LeftHandIndex2_rx', 'LeftHandMiddle2_rx', 'LeftHandRing2_rx', 'LeftHandPinky2_rx',
                   'LeftHandThumb2_rx', 'LeftHandIndex3_rx', 'LeftHandMiddle3_rx', 'LeftHandRing3_rx', 'LeftHandPinky3_rx',
                   'LeftHandThumb3_rx',
                   'LeftHandIndex1_ry', 'LeftHandMiddle1_ry', 'LeftHandRing1_ry', 'LeftHandPinky1_ry',
                   'LeftHandThumb1_ry', 'LeftHandIndex2_ry', 'LeftHandMiddle2_ry', 'LeftHandRing2_ry', 'LeftHandPinky2_ry',
                   'LeftHandThumb2_ry', 'LeftHandIndex3_ry', 'LeftHandMiddle3_ry', 'LeftHandRing3_ry', 'LeftHandPinky3_ry',
                   'LeftHandThumb3_ry', 'LeftHandIndex1_rz', 'LeftHandMiddle1_rz', 'LeftHandRing1_rz', 'LeftHandPinky1_rz',
                   'LeftHandThumb1_rz', 'LeftHandIndex2_rz', 'LeftHandMiddle2_rz', 'LeftHandRing2_rz', 'LeftHandPinky2_rz',
                   'LeftHandThumb2_rz', 'LeftHandIndex3_rz', 'LeftHandMiddle3_rz', 'LeftHandRing3_rz', 'LeftHandPinky3_rz',
                   'LeftHandThumb3_rz', 'IndexMiddle', 'ThumbNeighbor'],
    outputCol="features")
dataset_train = assembler.transform(dataset_train)
dataset_train = dataset_train.select("features", "Label")
dataset_train = dataset_train.withColumnRenamed("Label", "label")
# indexer = StringIndexer(inputCol="label_str", outputCol="label")
# dataset_train = indexer.fit(dataset_train).transform(dataset_train)
# dataset_train = dataset_train.withColumn("label", dataset_train["label"].cast(ord()))
# dataset_train.show(1)

# print(dataset_test)
dataset_test = spark.createDataFrame(dataset_test)
dataset_test = assembler.transform(dataset_test)
dataset_test = dataset_test.select("features", "Label")
dataset_test = dataset_test.withColumnRenamed("Label", "label")

# indexer = StringIndexer(inputCol="label_str", outputCol="label")
# dataset_test = indexer.fit(dataset_test).transform(dataset_test)
# dataset_test = dataset_test.withColumn("label", dataset_test["label"].cast(ord())
# dataset_test.show(1)

# create the trainer and set its parameters
nb = NaiveBayes(modelType="gaussian")
# nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(dataset_train)

# select example rows to display.
predictions = model.transform(dataset_test)
predictions.show(5, True)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))