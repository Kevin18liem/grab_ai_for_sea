from multiprocessing import Pool as ThreadPool
from multiprocessing import Value, Lock
from ctypes import c_int
import itertools
from functools import partial
import cv2
import numpy as np
from config import *
import pandas as pd

numParallelProcess = 16
counter = Value(c_int)
counter_lock = Lock()
np.random.seed(numpySeed)

def squashImageArray(imageDataArray,sizeX,sizeY):
    return cv2.resize(imageDataArray,(sizeX,sizeY))

def loadImageAsArray(job_args):
    imageData = cv2.imread(job_args[0])
    imPhoto = imageData[job_args[1][1]:job_args[1][3], job_args[1][0]:job_args[1][2]]
    (r,g,b) = cv2.split(imPhoto)
    imPhoto = cv2.merge([r,g,b])
    return imPhoto

def loadAndSquashImage(job_args, sizeX, sizeY, totalImages):
    with counter_lock:
        counter.value += 1
        print ("Image Processed:"+str(counter.value)+"/"+str(totalImages))
    return squashImageArray(loadImageAsArray(job_args),sizeX,sizeY)

def loadAndSquashImagesParallely(imagePaths,sizeX,sizeY, bbox_values):
    imagesDataList = []
    counter.value = 0
    pool = ThreadPool(numParallelProcess)
    job_args = list()
    for i in range(len(imagePaths)):
        temp = (imagePaths[i], bbox_values[i])
        job_args.append(temp)
    imagesDataList = pool.map(partial(loadAndSquashImage, sizeX=sizeX, sizeY=sizeY, totalImages=len(imagePaths)),job_args)
    pool.close()
    pool.join()
    return np.array(imagesDataList)