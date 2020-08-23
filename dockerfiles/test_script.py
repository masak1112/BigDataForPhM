import numpy as np
from pyspark.sql.types import *
data = np.load("test.npz")
dt = data['arr_0']
print ("data:",dt)
#convert the dt to spark DataFrame
df = spark.createDataFrame(dt.tolist(), ArrayType(FloatType()))
#show the first 20 raws
df.show()

#parse and save data to tfrecords
num_partition = 4

tfrecord_location = "./tfrecords/"
path = "test_samples.tfrecord"

df.repartition(num_partition).write.format("tfrecords").mode("overwrite").save(path)
