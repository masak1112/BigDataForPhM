
__author__ = "Bing Gong"
__version__ = "1.0.1"
__maintainer__ = "Bing Gong"
__email__ = "b.gong@fz-juelich.de"

import random
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import glob
from pyspark.sql.types import *
from collections import OrderedDict
#import horovod as hvd


class GenerateData:

    def __init__(self, batch_size=None, input_dir=None,mode="train",num_epochs=None):

        """
        :param batch_size: number of samples for each iteration during training validation and testing
        :param input_dir: tfrecords input directory, should include three subfolders: train/val/test
        :param mode: train/val/test
        """
       
        self.mode = mode
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_filenames(self):
        """
        get all the filenames in the input directory
        :return: None
        """
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:
            filenames = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                print("filenames:",self.filenames)
                break
        return None


    @staticmethod
    def load_npz_save_tf(file="test.npz"):

        """
        Load the npz files and save it to tfrecords
        :param file: the path of .npz file
        :return: None
        """
        dat = np.load(file)
        dt = dat['arr_0']
        test_rows = dt[:10].tolist() #take the first 30 examples
        fields  = [StructField("FloatCol"+str(i), FloatType()) for i in range(1220)]
        schema = StructType(fields)
        rdd = spark.sparkContext.parallelize(test_rows)
        df = spark.createDataFrame(rdd, schema)
        path = "test-output.tfrecord"
        df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save(path)
        df = spark.read.format("tfrecord").option("recordType", "Example").load(path)
        df.show()
        return None



    @staticmethod
    def parser(serialized_example):
        seqs = OrderedDict()
        keys_to_features = {
            'FloatCol1': tf.io.VarLenFeature(tf.float32),
            'FloatCol2': tf.io.VarLenFeature(tf.float32),
            'FloatCol3': tf.io.VarLenFeature(tf.float32),
            'FloatCol4': tf.io.VarLenFeature(tf.float32),
        }

        # for i in range(20):
        #     keys_to_features["frames/{:04d}".format(i)] = tf.FixedLenFeature((), tf.string)
        parsed_features = tf.parse_single_example(serialized_example, keys_to_features)
        print("Parse features", parsed_features)
        col2 = tf.sparse_tensor_to_dense(parsed_features["FloatCol2"])
        col1 = tf.sparse_tensor_to_dense(parsed_features["FloatCol1"])
        col3 = tf.sparse_tensor_to_dense(parsed_features["FloatCol3"])
        cols = [col1,col2,col3]
        seqs["inputs"] = cols
        print("seqs",seqs)
        return seqs

    def make_dataset(self, min =0.1, max=0.9):
        # Random samples if they are from train or val
        self.get_filenames()
        random.shuffle(self.filenames)
        dataset = tf.data.TFRecordDataset(self.filenames, buffer_size = 8 * 1024 * 1024)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = 1024, count = self.num_epochs))
        num_parallel_calls = None
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            GenerateData.parser, self.batch_size, drop_remainder = True, num_parallel_calls = num_parallel_calls))
        dataset = dataset.prefetch(self.batch_size)  # Take the data to buffer inorder to save the waiting time for GPU
        print("Daatset",dataset)
        return dataset


    def make_batch(self):
        dataset = self.make_dataset(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @staticmethod
    def make_dataset_iterator(train_dataset, val_dataset, batch_size:int):
        """
        Function to generate train, val dataset handle used for training
        """
        train_tf_dataset = train_dataset.make_dataset(batch_size)
        train_iterator = train_tf_dataset.make_one_shot_iterator()
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        train_handle = train_iterator.string_handle()
        val_tf_dataset = val_dataset.make_dataset(batch_size)
        val_iterator = val_tf_dataset.make_one_shot_iterator()
        val_handle = val_iterator.string_handle()
        #iterator = tf.data.Iterator.from_string_handle(
        #    train_handle, train_tf_dataset.output_types, train_tf_dataset.output_shapes)
        inputs = train_iterator.get_next()
        val = val_iterator.get_next()
        return inputs,train_handle, val_handle



    def num_samples(self):
        """
        Calculate the number of total samples in train ,val or testing dataset
        :return:None
        """
        #TODO
        self.num_samples = 1000
        return  None


class AEModel:

    def __init__(self,nz=None,learning_rate=None,epochs=None,batch_size=None):
        self.nz = nz
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size


    def build_graph(self,x):
        """
        Build the graph with steps including inputs, structure, loss function, optimizer and summary loss, save variables
        :return: None
        """
        self.x = x["inputs"]
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        original_global_variables = tf.global_variables()
        # ARCHITECTURE
        self.ae_network()
        #loss function
        self.total_loss = tf.compat.v1.losses.mean_squared_error(self.x, self.x_hat)
        tf.summary.scalar('loss', self.total_loss)

        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # Optimizer
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate).minimize(self.total_loss, global_step = self.global_step)
        self.outputs = {}
        self.outputs["x_hat"] = self.x_hat
        # Summary op
        self.loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        self.summary_op = tf.summary.merge_all()
        global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
        self.saveable_variables = [self.global_step] + global_variables
        return None


    def ae_network(self):
        """
        The main structure of autoencoder network
        :return:
        """
        input_dim = self.x.get_shape()
        self.x = tf.reshape(self.x,[input_dim[0],input_dim[1]])
        input_dim = self.x.get_shape().as_list()[1]
        print("input_dim:",input_dim)
        
        # Encode
        # x -> z
        f1 = fc(self.x, 256, scope = 'enc_fc1', activation_fn = tf.nn.relu)
        f3 = fc(f1, 64, scope = 'enc_fc3', activation_fn = tf.nn.relu)
        z = fc(f3, self.nz, scope = 'enc_fc4', activation_fn = tf.nn.relu)

        # Decode
        # z -> x_hat
        g1 = fc(z, 64, scope = 'dec_fc1', activation_fn = tf.nn.relu)
        g3 = fc(g1, 256, scope = 'dec_fc3', activation_fn = tf.nn.relu)
        self.x_hat = fc(g3, input_dim, scope = 'dec_fc4',
                   activation_fn = tf.sigmoid)
        print("build graph done!")

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)

        # loss fucntion for cross entroy
        #         epsilon = 1e-8
        #         recon_loss = -tf.reduce_sum(
        #             x * tf.log(epsilon+x_hat) +
        #             (1-x) * tf.log(epsilon+1-x_hat),
        #             axis=1
        #         )

        # losses = tf.reduce_mean(recon_loss)
        ## Loss funtion for mse


    def train(self,gpu_mem_frac=0.9,val_handle=None):
        self.val_handle = val_handle
        self.gpu_mem_frac = gpu_mem_frac
        #setup graph
        # List of all Variables
        variables = tf.global_variables()
        # Build a saver
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(var_list=self.saveable_variables, max_to_keep=2)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_mem_frac, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        #calculate the number of iteration per epoch
        #todo need fundtion to calculate the num_samples,here use 10 instread for testing
        step_per_epoch = int(4/self.batch_size)
        #Total iterations
        total_steps = step_per_epoch * self.epochs
        results_dict = {}
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # model.restore(sess, args.checkpoint)
            sess.graph.finalize()
            start_step = sess.run(self.global_step)
            train_losses = []
            val_losses = []
            run_start_time = time.time()
            for step in range(total_steps):
                global_step = sess.run(self.global_step)
                print("global_step:", global_step)
                val_handle_eval = sess.run(self.val_handle)
                fetches = {"global_step":self.global_step}
                fetches["train_op"] = self.train_op
                # fetches["latent_loss"] = model.latent_loss
                fetches["total_loss"] = self.total_loss
                results = sess.run(fetches)
                print("results", results)
        return None


class VisTools:
    pass


def main():
    #setup train/val datasets instance
    train_dataset = GenerateData(batch_size=2, input_dir="./data",  mode="train",num_epochs=20)
    val_dataset = GenerateData(batch_size=2, input_dir="./data",mode="val",num_epochs=20)
    inputs,train_handle, val_handle = GenerateData.make_dataset_iterator(train_dataset, val_dataset, batch_size=2)
    #setup model instance
    model = AEModel(nz=3,learning_rate=0.01,epochs=20,batch_size=2)
    model.build_graph(inputs)
    model.train(gpu_mem_frac=0.9,val_handle=val_handle)
 

if __name__=="__main__":
    main()
