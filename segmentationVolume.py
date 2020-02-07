import SimpleITK as sitk
import numpy as np
import os

def segVol(img, segmentation):
    """
    Calculate the volume of segmentation, given the original image
    """