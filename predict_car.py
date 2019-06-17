import tensorflow as tf
import config
import imageUtil
import argparse
import numpy as np
import os
import sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file_to_test',type=str,default='car_dataset/cars_test/00001.jpg', help='Car Image to Predict')
parser.add_argument('--bbox_x1', type=int, default=0, help='bounding box value')
parser.add_argument('--bbox_y1', type=int, default=0, help='bounding box value')
parser.add_argument('--bbox_x2', type=int, default=0, help='bounding box value')
parser.add_argument('--bbox_y2', type=int, default=0, help='bounding box value')
FLAGS, unparsed = parser.parse_known_args()

#check whether bounding box value available

image_file = FLAGS.file_to_test
bbox_values = [FLAGS.bbox_x1,FLAGS.bbox_y1,FLAGS.bbox_x2,FLAGS.bbox_y2]
job_args = (image_file, bbox_values)

image_data = imageUtil.loadImageAsArray(job_args)
imageDataScale = np.array(imageUtil.squashImageArray(image_data,299,299))

checkpoint = tf.train.latest_checkpoint(".\model_inception")
saver = tf.train.import_meta_graph(checkpoint + '.meta')
modelPath = os.path.join("./imagenetInception", config.INCEPTION_MODEL_GRAPH_DEF_FILE)

#create bottleneck
with tf.Graph().as_default() as inceptionGraph:
    with tf.gfile.FastGFile(modelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneckTensor, jpeg_data_tensor, resized_input_tensor, decoded_jpeg_data_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[config.BOTTLENECK_TENSOR_NAME, config.JPEG_DATA_TENSOR_NAME,config.RESIZED_INPUT_TENSOR_NAME,config.DECODED_JPEG_DATA_TENSOR_NAME]))

b = list()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(".\model_inception")
    if ckpt and ckpt.model_checkpoint_path:
        print("Test File " + image_file,file=sys.stderr)
        # print(ckpt.model_checkpoint_path+" :Testing this checkpoint...")
        graph = tf.get_default_graph()
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        
        # Create bottleneck for picture
        bottleneckTensor_variable = graph.get_tensor_by_name("pool_3/_reshape:0")
        decoded_value = graph.get_tensor_by_name("DecodeJpeg:0")
        bottleneck_new_values = sess.run(bottleneckTensor_variable,{decoded_value: imageDataScale})
        bottleneck_new_values = np.squeeze(bottleneck_new_values)
        b.append(bottleneck_new_values)
        testX = graph.get_tensor_by_name("input/BottleneckInputPlaceholder:0")
        keep_rate = graph.get_tensor_by_name("input/dropout_keep_rate:0")
        final_layer = graph.get_tensor_by_name('final_result:0')
        a = sess.run(final_layer,feed_dict={testX:b, keep_rate:0.5})
        prediction = np.argmax(a, 1)
        class_index = pd.read_csv('class_index.csv')
        result = class_index.loc[class_index['class'] == prediction[0]]
        print(result)