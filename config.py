import os

n_classes = 196
numChannels = 3
batch_size = 128
numpySeed = 1234
imageSizeX = 299
imageSizeY = 299
split_percentage = 0.8
outputFile = "carDataset.pkl"
numEpochs = 3000
basePath = "car_dataset"
train_dataset = os.path.join(basePath,"cars_train_csv.csv")
test_dataset = os.path.join(basePath,"cars_test_csv.csv")
oversample_minority = False

#Inception Constants
INCEPTION_MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
INCEPTION_MODEL_GRAPH_DEF_FILE = 'classify_image_graph_def.pb'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
DECODED_JPEG_DATA_TENSOR_NAME = 'DecodeJpeg:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
FINAL_MINUS_1_LAYER_SIZE = 512
FINAL_MINUS_2_LAYER_SIZE = 512


#Batch Normalization
enableBatchNormalization = False

#Local Response Normalization
enableLocalResponseNormalization = False

#image standardization
enableImageStandardization = False