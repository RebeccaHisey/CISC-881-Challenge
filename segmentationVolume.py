import SimpleITK as sitk
import numpy as np
import os

def segVol(img, segmentation):
    """
    Author: Sindhura Thirumal
    Takes in an image and its segmentation and returns the volume of the segmentation
    """
    imgArray = sitk.GetArrayFromImage(img)
    maskArray = sitk.GetArrayFromImage(segmentation)
    spacing = img.GetSpacing()
    pixelCount = np.count_nonzero(maskArray)
    volume = pixelCount * spacing[0] * spacing[1] * spacing[2]
    return volume