import numpy
import math
from keras import optimizers
from keras.utils import to_categorical,Sequence
from keras.models import model_from_json

from models import unet3D

import SimpleITK as sitk
import os

from generateDataSetSequence import LNDbSequence

from argparse import ArgumentParser

def createModel():
    # author Rebecca Hisey
    imageSize = 64
    bottleneck_size = imageSize/4
    numFilters = 6
    regularizer = 0
    dropoutRate = 0.1
    seed = 0
    batchNormalization = 1

    unet = unet3D(imageSize,bottleneck_size,numFilters,regularizer,dropoutRate,seed,batchNormalization)
    unetModel = unet.createModel()
    return unetModel

def trainModel(unetModel,dataFolder):
    # author Rebecca Hisey
    sgd = optimizers.SGD()
    (images, segmentations) = loadSamples(dataFolder)
    datasetSequence = LNDbSequence(images,segmentations,batchSize=10)
    unetModel.compile(optimizer=sgd, loss='mean_squared_error')
    unetModel.fit(x=datasetSequence.inputs,y=datasetSequence.targets,batch_size=10)
    return unetModel

def saveModel(model,filePath):
    # author Rebecca Hisey
    JSONmodel = model.to_json()
    structureFileName = 'lungSegmentationUNet.json'
    weightsFileName = 'lungSegmentationUNet.h5'
    with open(os.path.join(filePath,structureFileName),"w") as modelStructureFile:
        modelStructureFile.write(JSONmodel)
    model.save_weights(os.path.join(filePath,weightsFileName))

def loadModel(filePath):
    # author Rebecca Hisey
    structureFileName = 'lungSegmentationUNet.json'
    weightsFileName = 'lungSegmentationUNet.h5'
    with open(os.path.join(filePath,structureFileName),"r") as modelStructureFile:
        JSONModel = modelStructureFile.read()
    model = model_from_json(JSONModel)
    model.load_weights(os.path.join(filePath,weightsFileName))
    return model

def preprocess(image):
    #author Rebecca Hisey
    image = numpy.load(image)
    image = normalize(image)
    return image

def normalize(image):
    #author Rebecca Hisey
    normImage = (image - numpy.min(image))/(numpy.max(image) - numpy.min(image))
    normImage = numpy.expand_dims(normImage, axis=0)
    normImage = numpy.expand_dims(normImage, axis=-1)
    return normImage

def loadSamples(currentDirectory):
    #author Rebecca Hisey
    R1Path = os.path.join(currentDirectory,'R1')
    images = []
    segmentations = []
    allFiles = os.listdir(R1Path)
    for imageNo in range(1,len(allFiles)):
        fileName = allFiles[imageNo]
        if 'Mask' in fileName:
            [_,imageFileName] = fileName.split('_')
            image = preprocess(os.path.join(R1Path,imageFileName))

            segmentation = preprocess(os.path.join(R1Path,fileName))
            images.append(image)
            segmentations.append(segmentation)
    return (images,segmentations)

def main(dataDirectory, saveModelDirectory):
    #author Rebecca Hisey
    unet = createModel()
# 'c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/'
    unet = trainModel(unet, dataDirectory)
# 'c:/Users/hisey/Documents/CISC_881/Challenge/'
    saveModel(unet, saveModelDirectory)


if __name__ == '__main__':
    currentDirectory = os.getcwd()

    parser = ArgumentParser(description='Trains segmentation model')
    parser.add_argument('-d', '--data', default=os.path.join(currentDirectory, 'Data'))
    parser.add_argument('-s', '--save', default=currentDirectory)

    args = parser.parse_args()
    main(args.data, args.save)
