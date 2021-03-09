#__author__ == "Bing Gong"
#__email__ == "gongbing1112@gmail.com"
#__date__ == "2021_01_10"



from pyspark import SparkConf, SparkContext
from pyspark.sql import  *
import pickle as pkl
import pyspark
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import *
from pyspark.sql.types import  DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import time
import numpy as np
import pandas as pd

sc = pyspark.SparkContext('local[*]')
sql = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()


print("sc.defaultParallelism:",sc.defaultParallelism)

def load_pkl(fl_path):
    """
    Function that read one pkl file and change it to RDD file
    This is the index, the real values are in antoher pickle file
    Note: the data type after loading pickle file is a dictionary
         e.g:  {
                  "2019-09-21":
                                 {
                                          
                                    "[4.160]": values in dataframes
                                    "[4.161]": values in dataframes
                                 }
             
               } 
    ["4.160"] meaning: signal values related different material passing through from period of time
    dataframew: three columns "from" "to" "len"
    
    args:
           fl_path: str, the full path of pickle file
    """
    with open(fl_path,"rb") as fl:
        dat = pkl.load(fl)
    #get the first key
    dat_val = dat["2019-09-21"]
    df = dat_val["[4.160]"]
    df = process_pkl(dat)
    return df

def process_vars_filtered(vars_filtered):
    """
    Preprocess the data for vars_filtered.pkl
    """
    with open(fl_path,"rb") as fl:
        dat = pkl.load(fl)
    print("dat")


def process_pkl(dat):
    """
    process the data from json to df
    """
    #get the firs tlevel of the keys
    dat_keys = dat.keys() # keys is the datetime
    data = pd.DataFrame()
    for key in dat_keys:
        dat_val = dat[key]
        dat_val_keys = dat_val.keys() # each key corresonds to each process
        df_processes = pd.DataFrame()
        for process in dat_val_keys:
            df_process = dat_val[process]
            if df_process is None:
                pass
            else:
                # get the name of columns and extend their name with the key names
                colnames = list(df_process.columns.values) 
                print("conames:",len(colnames))
                #new_colnames = [x + "_" + process for x in colnames]

                #df_process.columns = new_colnames
                df_processes = pd.concat([df_processes,df_process])
                print("lengh of new col",len(df_processes.columns.values))
        data = pd.concat([data,df_processes])        
    cols_num = data.shape[1]
    update_colnames = list(range(cols_num))
    update_colnames = [str(i) for i in update_colnames]
    data.columns = update_colnames
    data = data.dropna()
    data =  pd.concat([data,data])
    data =  pd.concat([data,data]) 
    data =  pd.concat([data,data])
    data =  pd.concat([data,data])
    data =  pd.concat([data,data])
    data =  pd.concat([data,data])
    return data



def df_to_rdd(df):
    """
    Function that change the data loaded from pkl file to rdd format
    args:
        df : a panda dataframe 
    Return:
        RDD data
    """
    cols = df.columns.values
    
    sqDF = sql.createDataFrame(df)
    df_rdd = sqDF.rdd
    print("RDD is created successfully")
    #print(df_rdd.take(10))
    df_rdd = spark.createDataFrame(df_rdd)
    #df_rdd = df_rdd.withColumn("From",df_rdd["From"].cast(DoubleType()))
    #df_rdd = df_rdd.select("From")
    print("After select column From")
    return df_rdd, cols


def norm_dat(df_rdd,colnames):
    """
    Normalize the input between 0 to 1, then we need use the MinMaxScaler
    
    """
    unlist = udf(lambda x: round(float(list(x)[0]),3),DoubleType())
    print("colnames:",colnames)
    for i in colnames:
        assembler = VectorAssembler(inputCols=[i], outputCol = str(i)+"_Vect")
        scaler = MinMaxScaler(inputCol = str(i) + "_Vect",outputCol=str(i)+"_Scaled")  
        pipline = Pipeline(stages=[assembler,scaler])
        df_rdd = pipline.fit(df_rdd).transform(df_rdd)
        df_rdd = df_rdd.withColumn(i+"_Scaled",unlist(i+"_Scaled")).drop(i+"_Vect").drop(i)
    df_rdd.show(5)
    cols_scaled = [str(i)+"_" + "Scaled" for i in colnames]
    assembler = VectorAssembler(inputCols=cols_scaled,outputCol="features")    
    trainingData = assembler.transform(df_rdd)
    return trainingData

def train(model="kmean",scaledData=None,test_id=4):
    """
    train models based on the model type
    args:
         model: str, the model type
         scaledData: the scaled data from the output of norm_data function
    """
    for i in range(5):
        kmeans = KMeans().setK(2).setSeed(1)
        model = kmeans.fit(scaledData)
        predictions = model.transform(scaledData)
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        centers = model.clusterCenters()
    for center in centers:
        print(center)
    predictions.rdd.saveAsPickleFile("k_mean_prediction_worker2_testid_"+str(test_id)+".pkl")

def vis_results():
    """
    Visulize the results of cluster
    """
    pass

    
def main(fl_path,test_id=4):
    """
    Build the pipline of ML application
    """
    time_start_load_pkl = time.time()
    df = load_pkl(fl_path)    
    time_load_pkl = time.time() - time_start_load_pkl
      
    time_start_preprocess_data = time.time()
    df_rdd, cols = df_to_rdd(df)
    # df_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    scaled_data = norm_dat(df_rdd,cols)
    time_preprocess_data = time.time() - time_start_preprocess_data

    time_start = time.time()    
    train(scaledData=scaled_data,test_id=test_id)  
    time_end = time.time()
    time_train = time_end - time_start
    print("time_train:",time_train)
    with open("time_train_worker2_testid_{}.txt".format(test_id),"w") as f:
        f.write("sc.defaultParallelism:" + str(sc.defaultParallelism)+ "\n")
        f.write("Time to load pkl file:" + str(time_load_pkl) + "\n")
        f.write("Time to preprocess data:" + str(time_preprocess_data) + "\n")
        f.write("Time to trian model:" + str(time_train) + "\n")



if __name__ == "__main__":
    fl_path = "vars_split.pkl"
    main(fl_path)    
