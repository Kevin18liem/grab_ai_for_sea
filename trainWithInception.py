import argparse
from InceptionV3 import *
import os.path
from tensorflow.python.framework import graph_util
import random
import numpy as np
from osUtils import *
import tensorflow as tf
import tarfile
from six.moves import urllib
import sys
import numpy as np
from PIL import Image
from config import *
from dataSetPreparator import *

random.seed(numpySeed)
tf.set_random_seed(numpySeed)
np.random.seed(numpySeed)

"""
Load the pretrained inception graph. Add FC layers as per our use case.
Also sets other parts of the model like optimizer, learning rate etc.
"""
def create_inception_graph(num_batches_per_epoch,FLAGS):
    modelFilePath = os.path.join(FLAGS.imagenet_inception_model_dir, INCEPTION_MODEL_GRAPH_DEF_FILE)
    inceptionV3 = InceptionV3(modelFilePath)
    inceptionV3.add_final_training_ops(n_classes,FLAGS.final_tensor_name,FLAGS.optimizer_name,num_batches_per_epoch, FLAGS)
    inceptionV3.add_evaluation_step()
    return inceptionV3

def calculateTrainAccuracy(sess):
    genericDataSetLoader.resetTrainBatch()
    batchAccuracies = []
    # cm_running_total = None
    while(True):
        trainX, trainY = genericDataSetLoader.getNextTrainBatch()
        if(trainX is None):
            break
        accuracy_batch, cross_entropy_value_batch = inceptionV3.evaluate(sess,trainX,trainY)
        batchAccuracies.append(accuracy_batch)
    print ("Training Accuracy:"+ str(sum(batchAccuracies) / float(len(batchAccuracies))))
def calculateTestAccuracy(sess):
    genericDataSetLoader.resetTestBatch()
    batchAccuracies = []
    # cm_running_total = None
    while(True):
        testX, testY = genericDataSetLoader.getNextTestBatch()
        if(testX is None):
            break
        accuracy_batch, cross_entropy_value_batch = inceptionV3.evaluate(sess,testX,testY)
        batchAccuracies.append(accuracy_batch)
    print ("Testing Accuracy:" +str(sum(batchAccuracies) / float(len(batchAccuracies))))
def restoreFromCheckPoint(sess,saver):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path+" :Restoring from a checkpoint...")
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        start = global_step.eval() # get last global_step
        start = start+1
    else:
        print ("Starting fresh training...")
        start = global_step.eval() # get last global_step
    return start

def trainInceptionNeuralNetwork(inceptionV3,FLAGS):
    with tf.Session(graph=inceptionV3.inceptionGraph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        genericDataSetLoader.convertToBottleNecks(sess,FLAGS,inceptionV3)
        start = 0
        print ("Start from:"+str(start)+"/"+str(numEpochs))
        prev_epoch_loss = 0
        avg_loss = 0
        #Training epochs
        for epoch in range(start,numEpochs):
            epoch_loss = 0
            genericDataSetLoader.resetTrainBatch()
            while(True):
                epoch_x, epoch_y = genericDataSetLoader.getNextTrainBatch()
                if(epoch_x is None):
                    break
                _,c = inceptionV3.train_step(sess,epoch_x,epoch_y,FLAGS.dropout_keep_rate)
                epoch_loss += c
            if(epoch == numEpochs-1):
                saver.save(sess,'model_inception/data-all.chkp',global_step=epochs)
            print ("Epoch:"+str(epoch)+'/'+str(numEpochs)+" loss:" + str(epoch_loss))
            avg_loss += epoch_loss
            prev_epoch_loss = epoch_loss
            calculateTrainAccuracy(sess)
            #Get the validation/test accuracy
            calculateTestAccuracy(sess)
        print("Average Epochs : " + str(avg_loss/numEpochs))
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--how_many_training_steps',type=int,default=500,help='How many training steps to run before ending.')
    parser.add_argument('--imagenet_inception_model_dir',type=str,default='./imagenetInception',help="""Path to classify_image_graph_def.pb,imagenet_synset_to_human_label_map.txt, and imagenet_2012_challenge_label_map_proto.pbtxt.""")
    parser.add_argument('--bottleneck_dir',type=str,default='./tmp/bottleneck',help='Path to cache bottleneck layer values as files.')
    parser.add_argument('--final_tensor_name',type=str,default='final_result',help="""The name of the output classification layer in the retrained graph.""")

    #Learning Rate and Optimizers
    parser.add_argument('--optimizer_name',type=str,default="rmsprop",help='Optimizer to be used: sgd,adam,rmsprop')
    parser.add_argument('--learning_rate_decay_factor',type=float,default=0.16,help='Learning rate decay factor.')
    parser.add_argument('--learning_rate',type=float,default=0.05,help='Initial learning rate.')
    parser.add_argument('--rmsprop_decay',type=float,default=0.9,help='Decay term for RMSProp.')
    parser.add_argument('--rmsprop_momentum',type=float,default=0.9,help='Momentum in RMSProp.')
    parser.add_argument('--rmsprop_epsilon',type=float,default=1.0,help='Epsilon term for RMSProp.')
    parser.add_argument('--num_epochs_per_decay',type=int,default=30,help='Epochs after which learning rate decays.')
    parser.add_argument('--learning_rate_type',type=str,default="exp_decay",help='exp_decay,const')

    #Normalizations/Regularizations
    parser.add_argument('--dropout_keep_rate',type=float,default=0.5)


    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly scale up the size of the
        training images by.\
        """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly multiply the training image
        input pixels up or down by.\
        """
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
        A percentage determining how much of a margin to randomly crop off the
        training images.\
        """
    )
    parser.add_argument('--flip_left_right', default=False,help="""\
        Whether to randomly flip half of the training images horizontally.\
        """,
        action='store_true')

    #parse the parameters
    FLAGS, unparsed = parser.parse_known_args()
    
    # load the prepared dataset from the pickled file
    genericDataSetLoader = dataSetPreparator()
    genericDataSetLoader.loadData("dump_dataset/carDataset.pkl")
    if oversample_minority:
        genericDataSetLoader.oversampleMinorityClass(oversampling_multiplier)
    numTrainingBatches = genericDataSetLoader.numberOfTrainingBatches()
    # load the pretrained inception graph and create the complete model
    inceptionV3 = create_inception_graph(numTrainingBatches, FLAGS)

    #train the neural network
    trainInceptionNeuralNetwork(inceptionV3,FLAGS)
