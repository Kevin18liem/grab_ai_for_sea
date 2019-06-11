import pandas as pd
from config import *
import math
from datetime import datetime
from random import shuffle
import imageUtil
import pickle
import numpy as np

class dataSetPreparator:

    def __init__(self):
        trainingDataX = None
        testingDataX = None
        trainingDataY = None
        trainingDataY = None
        trainingDataBbox_value = None
        testingDataBbox_value = None

    def __trainTestSplit(self,filePaths,labels,bbox_value):
        splitIndex = int(math.ceil(split_percentage*len(filePaths)))
        trainingDataX = filePaths[:splitIndex]
        trainingDataY = labels[:splitIndex]
        testingDataX = filePaths[splitIndex:]
        testingDataY = labels[splitIndex:]
        trainingDataBboxValue = bbox_value[:splitIndex]
        testingDataBboxValue = bbox_value[splitIndex:]
        return trainingDataX, trainingDataY, testingDataX, testingDataY, trainingDataBboxValue, testingDataBboxValue

    def __getCurrentTime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def __shuffle(self,list1,list2,list3):
        list1_shuf = []
        list2_shuf = []
        list3_shuf = []
        index_shuf = list(range(len(list1)))
        shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(list1[i])
            list2_shuf.append(list2[i])
            list3_shuf.append(list3[i])
        return list1_shuf,list2_shuf, list3_shuf

    def __save(self,outputPath):
        print ("Saving the processed data...")
        preparedData={}
        preparedData["trainingX"] = self.trainingDataX
        preparedData["trainingY"] = self.trainingDataY
        preparedData["testingX"] = self.testingDataX
        preparedData["testingY"] = self.testingDataY
        pklFile = open(outputPath, 'wb')
        pickle.dump(preparedData, pklFile, protocol=4)
        pklFile.close()
        print ("Data saved...")

    def prepareDataSet(self, outputFile):
        self.trainingDataX = []
        self.trainingDataY = []
        self.testingDataX = []
        self.testingDataY = []
        self.trainingDataBbox_value = []
        self.testingDataBbox_value = []

        df_train = pd.read_csv(train_dataset)

        classList = df_train['labels']
        idx = 0
        dataMap = {}
        for class_label in classList:
            if not (class_label in dataMap):
                dataMap[class_label] = {}
                dataMap[class_label]["filePaths"] = []
                dataMap[class_label]["fileLabels"] = []
                dataMap[class_label]["bbox_value"] = []
            dataMap[class_label]["filePaths"].append(df_train["fname"][idx])
            dataMap[class_label]["fileLabels"].append(df_train["class"][idx])
            bbox_list = [df_train["bbox_x1"][idx], df_train["bbox_y1"][idx], df_train["bbox_x2"][idx], df_train["bbox_y2"][idx]]
            dataMap[class_label]["bbox_value"].append(bbox_list)
            idx+=1

        for key, value in dataMap.items():
            dataMap[key]["trainingDataX"] ,dataMap[key]["trainingDataY"], dataMap[key]["testingDataX"], dataMap[key]["testingDataY"], dataMap[key]["trainingData_bbox_value"], dataMap[key]["testingData_bbox_value"]  = self.__trainTestSplit(dataMap[key]["filePaths"],dataMap[key]["fileLabels"], dataMap[key]["bbox_value"])
            self.trainingDataX.extend(dataMap[key]["trainingDataX"])
            self.trainingDataY.extend(dataMap[key]["trainingDataY"])
            self.testingDataX.extend(dataMap[key]["testingDataX"])
            self.testingDataY.extend(dataMap[key]["testingDataY"])
            self.trainingDataBbox_value.extend(dataMap[key]["trainingData_bbox_value"])
            self.testingDataBbox_value.extend(dataMap[key]["testingData_bbox_value"])
        
        self.trainingDataX,self.trainingDataY, self.trainingDataBbox_value = self.__shuffle(self.trainingDataX,self.trainingDataY, self.trainingDataBbox_value)
        self.testingDataX,self.testingDataY, self.testingDataBbox_value = self.__shuffle(self.testingDataX,self.testingDataY, self.testingDataBbox_value)

        self.__postProcessData()
        self.__save(outputFile)

    def __loadImageDataParallely(self,fileNames, bbox_values):
        imagesDataList = imageUtil.loadAndSquashImagesParallely(fileNames,imageSizeX,imageSizeY, bbox_values)
        return imagesDataList

    def __convertLabelsToOneHotVector(self,labelsList):
        labelsArray = np.array(labelsList)
        oneHotVector = np.zeros((labelsArray.shape[0],n_classes),dtype=np.int8)
        oneHotVector[np.arange(labelsArray.shape[0]), labelsArray] = 1
        return oneHotVector

    def __postProcessData(self):
        print ("Reading the training image files..."+self.__getCurrentTime())
        self.trainingDataX = self.__loadImageDataParallely(self.trainingDataX, self.trainingDataBbox_value)
        print ("Reading the training image files..."+self.__getCurrentTime())
        self.testingDataX = self.__loadImageDataParallely(self.testingDataX, self.testingDataBbox_value)

        print ("Creating one hot encoded vectors for training labels..."+self.__getCurrentTime())
        self.trainingDataY = self.__convertLabelsToOneHotVector(self.trainingDataY)
        print ("Creating one hot encoded vectors for testing labels..."+self.__getCurrentTime())
        self.testingDataY = self.__convertLabelsToOneHotVector(self.testingDataY)
