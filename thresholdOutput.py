import numpy as np
import mahotas as mh

def thresholdCube(outputArray):
    """
    Author: Sindhura Thirumal
    Takes in array outputted by trainModel and thresholds to create a binary image.
    Output is a label array of each of the nodules in the cube
    """
    outputArray *= 255.0/outputArray.max()  # Scale image to be in range 0-255
    intImg = outputArray.astype(np.uint8)   # Convert data type of array to uint8
    for i in range(intImg.shape[2]):
        otsuThreshVal = mh.otsu(intImg[i])
        intImg[i] = intImg[i] > otsuThreshVal
        intImg[i], nNodules = mh.label(intImg[i])
    return intImg

def findCentroidNodule(labeledSeg):
    """
    Author: Sindhura Thirumal
    Given the labeled array of the segmentation, returns the nodule segmentation that overlaps with
    the centroid. Deletes all other nodules within cube.
    """
    centroid = [40, 40, 40]
    centroidSeg = labeledSeg[centroid]
    for i in range(labeledSeg.shape[2]):
        nNods = np.unique(labeledSeg[i])
        for j in range(len(nNods)):
            if j != centroidSeg:
                labeledSeg[i][labeledSeg[i] == j] = 0
    return labeledSeg
