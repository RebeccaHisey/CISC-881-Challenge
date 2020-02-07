import numpy as np
import mahotas as mh

def thresholdCube(outputArray):
    """
    Author: Sindhura Thirumal
    Takes in array outputted by trainModel and thresholds to create a binary image.
    Output is a label array of each of the nodules in the cube
    """
    # author Sindhura Thirumal
    # Scale image to be in range 0-255
    outputArray *= 255.0/outputArray.max()
    # Convert data type of array to uint8
    intImg = outputArray.astype(np.uint8)
    for i in range(intImg.shape[2]):
        otsuThreshVal = mh.otsu(intImg[i])
        intImg[i] = intImg[i] > otsuThreshVal
        intImg[i], nNodules = mh.label(intImg[i])
    return intImg

