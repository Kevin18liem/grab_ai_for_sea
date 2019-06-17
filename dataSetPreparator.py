import pandas as pd
from config import *
import math
from datetime import datetime
from random import shuffle
import imageUtil
import pickle
import numpy as np
import DatasetManager
    
np.random.seed(numpySeed)

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

    def loadData(self, fileName):
        pklFile = open(fileName, 'rb')
        preparedData=pickle.load(pklFile)
        self.trainingDataX = preparedData["trainingX"]
        self.trainingDataY = preparedData["trainingY"]
        self.testingDataX = preparedData["testingX"]
        self.testingDataY = preparedData["testingY"]
        print ("Data loaded...")
        print (self.trainingDataX.shape)
        print (self.trainingDataY.shape)
        print (self.testingDataX.shape)
        print (self.testingDataY.shape)
    
    def numberOfTrainingBatches(self):
        return int(len(self.trainingDataX)/batch_size)

    def convertArrayToBottleNecks(self,sess,dataArray,category,FLAGS,inceptionV3):
        bottlenecks = []
        batchSize = 512
        i = 0
        totalNumImages = dataArray.shape[0]
        print ("Total Number of images:"+str(totalNumImages))
        while i<totalNumImages:
            minIndx = i
            maxIndx = min(dataArray.shape[0],i+batchSize)
            print (str(i)+"/"+str(dataArray.shape[0]))
            bottlenecksBatch = DatasetManager.get_random_cached_bottlenecks(sess,i,dataArray[minIndx:maxIndx],category,FLAGS.bottleneck_dir,inceptionV3)
            bottlenecks.extend(bottlenecksBatch)
            i = i + batchSize
        return np.array(bottlenecks)

    def convertToBottleNecks(self,sess,FLAGS,inceptionV3):
        print ("Converting dataset to bottlenecks...")
        self.trainingDataX = np.squeeze(self.convertArrayToBottleNecks(sess,self.trainingDataX,"train",FLAGS,inceptionV3))
        self.testingDataX = np.squeeze(self.convertArrayToBottleNecks(sess,self.testingDataX,"test",FLAGS,inceptionV3))
        print (self.trainingDataX.shape)
        print (self.testingDataX.shape)
        print ("Converted dataset to bottlenecks...")

    def resetTrainBatch(self):
        self.trainingDataOffset=0
    
    def resetTestBatch(self):
        self.testingDataOffset=0

    def selectRows(self,dataArray,rowOffset,numOfRows):
        if(rowOffset>=dataArray.shape[0]):
            return None
        elif ((rowOffset+numOfRows)>dataArray.shape[0]):
            return dataArray[rowOffset:dataArray.shape[0],:]
        return dataArray[rowOffset:rowOffset+numOfRows,:]

    def getNextTrainBatch(self):
        trainDataX = self.selectRows(self.trainingDataX,self.trainingDataOffset,batch_size)
        trainDataY = self.selectRows(self.trainingDataY,self.trainingDataOffset,batch_size)
        self.trainingDataOffset = self.trainingDataOffset+batch_size
        return trainDataX,trainDataY

    def getNextTestBatch(self):
        testDataX = self.selectRows(self.testingDataX,self.testingDataOffset,batch_size)
        testDataY = self.selectRows(self.testingDataY,self.testingDataOffset,batch_size)
        self.testingDataOffset = self.testingDataOffset+batch_size
        return testDataX,testDataY


    def analyzeDataDistribution(self):
        # self.loadData("carDataset.pkl")
        print ("Total Training Instances:"+str(self.trainingDataY.shape[0]))
        print ("Total Testing Instances:"+str(self.testingDataY.shape[0]))
        #print self.__convertOneHotVectorToLabels(self.trainingDataY)
        for classIndex in range(0,n_classes):
            print ("Distribution For Class:"+str(classIndex))
            trainDistribution = self.__convertOneHotVectorToLabels(self.trainingDataY)
            trainDistribution = np.count_nonzero(trainDistribution == classIndex)
            testDistribution = self.__convertOneHotVectorToLabels(self.testingDataY)
            testDistribution = np.count_nonzero(testDistribution == classIndex)
            print ("Instances In Training Data:"+str(trainDistribution))
            print ("Instances In Testing Data:"+str(testDistribution))
        print ("Done")

    def __convertOneHotVectorToLabels(self,oneHotVectors):
        labels = np.argmax(oneHotVectors==1,axis=1)
        return labels