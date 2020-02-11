import numpy
import math
import keras.models
import random
from keras import optimizers
from keras.utils import to_categorical,Sequence
from keras.models import model_from_json

from models import unet3D
from models import IoU_loss

import SimpleITK as sitk
import os
import csv

from generateDataSetSequence import LNDbSequence

class trainModelUtilities():

    def createModel(self):
        # author Rebecca Hisey
        imageSize = 80
        bottleneck_size = imageSize/4
        numFilters = 6
        regularizer = 0
        dropoutRate = 0.1
        seed = 0
        batchNormalization = 1

        unet = unet3D(imageSize,bottleneck_size,numFilters,regularizer,dropoutRate,seed,batchNormalization)
        unetModel = unet.createModel()
        return unetModel

    def crossValidation(self,modelFolder,dataFolder):
        #trainedModel = self.loadModel(modelFolder)
        unetModel = self.createModel()

        folds = self.getFolds()
        for i in range(2,3):
            (images, segmentations,valImages,valSegmentations,valIDs) = self.loadSamples(dataFolder,folds[i])
            trainedModel = self.trainModel(unetModel,images,segmentations)
            self.saveModel(trainedModel,modelFolder,i)
            self.getModelPrediction(valImages,trainedModel,dataFolder,valIDs,i)


    def getFolds(self):
        csvFileName = 'trainFolds.csv'
        with open(csvFileName,'r') as foldsCSVFile:
            fileReader = csv.reader(foldsCSVFile)
            lines = list(fileReader)
        lines = lines[1:]
        folds = [[],[],[],[]]
        for i in range(0,4):
            folds[i] = [int(x[i]) for x in lines]
        return folds


    def trainModel(self,unetModel,images,segmentations):
        # author Rebecca Hisey
        sgd = optimizers.SGD()
        dataSequence = LNDbSequence(images,segmentations,batchSize=5)
        unetModel.compile(optimizer=sgd, loss=IoU_loss)
        unetModel.fit(x=dataSequence.inputs,y=dataSequence.targets,batch_size=5,epochs=200)
        return unetModel

    def saveModel(self,model,filePath,foldNumber):
        # author Rebecca Hisey
        JSONmodel = model.to_json()
        structureFileName = 'lungSegmentationUNet_iouloss_fold' + str(foldNumber) + '.json'
        weightsFileName = 'lungSegmentationUNet_iouloss_fold' + str(foldNumber) + '.h5'
        with open(os.path.join(filePath,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
        model.save_weights(os.path.join(filePath,weightsFileName))

    def loadModel(self,filePath):
        # author Rebecca Hisey

        structureFileName = 'lungSegmentationUNet_iouloss_fold0.json'
        weightsFileName = 'lungSegmentationUNet_iouloss_fold0.h5'
        with open(os.path.join(filePath,structureFileName),"r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(filePath,weightsFileName))
        '''
        model = self.createModel()
        model.load_weights(os.path.join(filePath,'best_1st.hdf5'))
        '''
        sgd = optimizers.SGD()
        model.compile(optimizer=sgd, loss=IoU_loss)
        return model

    def getModelPrediction(self,input,model,dataFolderPath,fileID,foldNo):
        #author Rebecca Hisey
        for i in range(0,len(input)):
            output = model.predict(x=input[i])
            foldId = 'fold' + str(foldNo)
            self.writeImageFile(output,os.path.join(dataFolderPath,foldId),fileID[i])
        return output

    def writeImageFile(self,npArrayImage,filePath,fileName):
        #author Rebecca Hisey
        threeDImage = npArrayImage[0]
        threeDImage = numpy.squeeze(threeDImage,axis=-1)
        print(fileName)
        numpy.save(os.path.join(filePath,fileName+'.npy'),threeDImage)
        image = sitk.GetImageFromArray(threeDImage)
        sitk.WriteImage(image,os.path.join(filePath,fileName + '.mhd'))

    def preprocess(self,imageFilePath):
        #author Rebecca Hisey
        image = numpy.load(imageFilePath)
        image = self.normalize(image)
        return image

    def normalize(self,image):
        #author Rebecca Hisey
        normImage = (image - numpy.min(image))/(numpy.max(image) - numpy.min(image))
        normImage = numpy.expand_dims(normImage, axis=0)
        normImage = numpy.expand_dims(normImage, axis=-1)
        return normImage

    def loadSamples(self,currentDirectory,fold):
        #author Rebecca Hisey
        R1Path = os.path.join(currentDirectory,'allReviewers_80')
        images = []
        segmentations = []
        valImages = []
        valSegmentations = []
        valIDs = []
        allFiles = os.listdir(R1Path)
        for imageNo in range(1,len(allFiles)):
                fileName = allFiles[imageNo]
                fileNumber = fileName.split('-')
                fileNumber = int(''.join(x for x in fileNumber[0] if x.isdigit()))
                if not fileNumber in fold:
                    if 'Mask' in fileName:
                        [_,imageFileName] = fileName.split('_')
                        image = self.preprocess(os.path.join(R1Path,imageFileName))

                        segmentation = self.preprocess(os.path.join(R1Path,fileName))
                        images.append(image)
                        segmentations.append(segmentation)
                else:
                    fileID = fileName.split('.')
                    fileID = fileID[0]
                    if 'Mask' in fileName:
                        [_,imageFileName] = fileName.split('_')
                        image = self.preprocess(os.path.join(R1Path,imageFileName))

                        segmentation = self.preprocess(os.path.join(R1Path,fileName))
                        valImages.append(image)
                        valSegmentations.append(segmentation)
                        [fileID,_] = imageFileName.split('.')
                        valIDs.append(fileID)


        return (images,segmentations,valImages,valSegmentations,valIDs)

    '''
    def getTrainValidationTestSplit(self,images,segmentations):
        (trainValidationImages,testImages,trainValidationSegmentations,testSegmentations) = train_test_split(images,segmentations,test_size=0.15)
        (trainImages,validationImages,trainSegmentations,validationSegmentations) = train_test_split(trainValidationImages,trainValidationSegmentations,test_size=0.15)
        return((trainImages,trainSegmentations),(validationImages,validationSegmentations),(testImages,testSegmentations))
    '''
def main():
    #author Rebecca Hisey
    utilities = trainModelUtilities()
    utilities.crossValidation('c:/Users/hisey/Documents/CISC_881/Challenge/','c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/')


    #unet = utilities.loadModel('c:/Users/hisey/Documents/CISC_881/Challenge/')
    #unet = utilities.trainModel(unet, 'c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/')
    #utilities.saveModel(unet, 'c:/Users/hisey/Documents/CISC_881/Challenge/')
    #sampleImage = utilities.preprocess('c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/allReviewers_80/LNDb0308-R2-F7.npy')
    #print(sampleImage.shape)
    #segmentation = utilities.preprocess('c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/allReviewers_80/Mask_LNDb0001-R2-F1.npy')
    #modelOutput = utilities.getModelPrediction([sampleImage],unet,'c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/',['LNDb0308-R2-F7'],3)
    #utilities.writeImageFile(segmentation,'c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/','Mask_LNDb0001-R2-F1.mhd')
    #utilities.writeImageFile(modelOutput, 'c:/Users/hisey/Documents/CISC_881/Challenge/Nodules/allReviewers/','Output_LNDb0001-R1-F1.mhd')

main()

