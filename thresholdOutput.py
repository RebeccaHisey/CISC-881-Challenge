import numpy as np
import mahotas as mh
from scipy.ndimage.measurements import label


def thresholdCube(outputArray):
    """
    Author: Sindhura Thirumal
    Takes in array outputted by trainModel and thresholds to create a binary image.
    Output is the binary image array.
    """
    outputArray *= 255.0 / outputArray.max()  # Scale image to be in range 0-255
    intImg = outputArray.astype(np.uint8)  # Convert data type of array to uint8
    for i in range(intImg.shape[2]):
        otsuThreshVal = mh.otsu(intImg[i])  # Otsu thresholding
        intImg[i] = intImg[i] > otsuThreshVal
    return intImg


def labelSegmentation(binaryImg):
    """
    Author: Sindhura Thirumal
    Takes in a binary volume outputted by thresholdCube and outputs the labeled volume.
    """
    struct3D = np.array([[[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],

                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],

                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]], dtype='uint8')
    labeled, nSegs = label(binaryImg, structure = struct3D)
    return labeled


def findCentroidNodule(labeledSeg):
    """
    Author: Sindhura Thirumal
    Takes in the labeled array from labelSegmentation and finds the nodule segmentation that
    overlaps with the centroid. Deletes all other nodules within cube. Output is the resulting binary image,
    only containing the centroid nodule.
    """
    centroidSeg = labeledSeg[40, 40, 40]  # Get label of the nodule overlapping the centroid
    for i in range(labeledSeg.shape[2]):
        nNods = np.unique(labeledSeg[i])  # Get segmentation labels from slice
        for j in nNods:
            if j != centroidSeg:
                labeledSeg[i][labeledSeg[i] == j] = 0  # If nodule does not belong to centroid nodule, delete it
    return labeledSeg
