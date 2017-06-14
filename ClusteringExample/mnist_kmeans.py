
from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
import math
from pyspark.mllib.feature import StandardScaler, StandardScalerModel

def parsePoint(line):				
	values = line.split(',')
	values = [0 if e == '' else int(e) for e in values]
	return values[0], values[1:]      

if __name__ == "__main__":

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    sc = SparkContext(appName="MNIST_KMEANS", conf=conf)
    
    data = sc.textFile('train.csv')  # ingest the comma delimited file
    header = data.first() # extract header
    data = data.filter(lambda x: x != header)  # remove the header 
    trainingData = data.map(parsePoint)  # parse file to generate an RDD 
    trainingData_wo_labels = trainingData.map(lambda x: x[1]) # remove label

    # normalize vector
    scaler = StandardScaler(withMean=True, withStd=True).fit(trainingData_wo_labels)
    trainingData_wo_labels = scaler.transform(trainingData_wo_labels)

    model = KMeans.train(trainingData_wo_labels, 
                        10, maxIterations=250, initializationMode="random")

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = model.centers[model.predict(point)]    # get centroid for cluster 
        return math.sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = trainingData_wo_labels.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))
 
    sc.stop()