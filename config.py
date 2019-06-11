import os

n_classes = 196
numChannels = 3
numpySeed = 1234
imageSizeX = 299
imageSizeY = 299
split_percentage = 0.8
outputFile = "carDataset.pkl"

basePath = "car_dataset"
train_dataset = os.path.join(basePath,"cars_train_csv.csv")
test_dataset = os.path.join(basePath,"cars_test_csv.csv")
