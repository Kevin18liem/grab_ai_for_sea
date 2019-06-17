from dataSetPreparator import *
from config import *
import numpy as np


filename = "dump_dataset/carDataset.pkl"

genericDataSetLoader = dataSetPreparator()
genericDataSetLoader.loadData(filename)
genericDataSetLoader.analyzeDataDistribution()