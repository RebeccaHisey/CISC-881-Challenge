import numpy
import math
from keras.utils import Sequence

class LNDbSequence(Sequence):
    def __init__(self,inputs,outputs,batchSize):
        # author Rebecca Hisey
        self.inputs = numpy.array([x[0] for x in inputs])
        self.targets = numpy.array([x[0] for x in outputs])
        self.batchSize = batchSize

    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def __getitem__(self,index):
        # author Rebecca Hisey
        startIndex = index*self.batchSize
        indexOfNextBatch = (index + 1)*self.batchSize
        inputBatch = numpy.array([x for x in self.inputs[startIndex : indexOfNextBatch]])
        outputBatch = numpy.array([x for x in self.targets[startIndex : indexOfNextBatch]])
        return (inputBatch,outputBatch)
