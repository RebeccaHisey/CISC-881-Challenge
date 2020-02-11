# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:06:18 2020

@author: emilyk
"""

import SimpleITK as sitk
import numpy as np
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import directed_hausdorff

def hausdorff_dist(seg_calc, seg_real, img_calc, img_real):
    positions_real_fin = []
    positions_calc_fin = []
    space_calc = img_calc.GetSpacing()
    space_real = img_real.GetSpacing()
    
    #Calculate coordinates for predicated/calculated image
    calc_slice = find_boundaries(seg_calc)
    positions_calc = zip(*np.where(calc_slice>0))
    positions_calc = list(positions_calc)
    positions_calc = np.array(positions_calc)
    positions_calc = (positions_calc * [space_calc[0], space_calc[1], space_calc[2]])
    positions_calc = (positions_calc).tolist()
    positions_calc_fin.append(positions_calc)
    
    #Calculate coordinates for real image
    real_slice = find_boundaries(seg_real)
    positions_real = zip(*np.where(real_slice>0))
    positions_real = list(positions_real)
    positions_real = np.array(positions_real)
    positions_real = (positions_real * [space_real[0], space_real[1], space_real[2]])
    positions_real = positions_real.tolist()
    positions_real_fin.append(positions_real)

    #Find Hausdorff distance in both directions
    dh_pc_pr = (directed_hausdorff(positions_calc, positions_real)[0])
    dh_pr_pc = (directed_hausdorff(positions_real, positions_calc)[0])
    dh = max(dh_pc_pr, dh_pr_pc)
    return dh
