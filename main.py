import os
import pandas as pd
import numpy as np
import torch
#import ToTensor
import torchvision.transforms as transforms
#import for imread
from DataCollectionAndPreparation import *
from skimage import io
from SoftmaxTrainingClassifier import *
from DRFeatureExtraction import *
from torchsummary import summary



if __name__ == "__main__":    
    dataset = CustomDataset(csv_path="../trainLabels.csv", root_dir="../train/images_resized_300/", transform=ImageTransformer(),classification="normal")
    # X_train , X_test , Y_train , Y_test = DataCollection(dataset,csv_path="../trainLabels.csv")
    # trainloader , testloader = split_data(dataset,batch_size=4,validation_split=0.2)
    # trainloader and testloader len
    print(len(dataset))
    print(dataset[0][0]==dataset[1][0])

