# -*- coding: utf-8 -*-
"""
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
    for k in range(len(seg_calc)):
        calc_slice = seg_calc[k]
        real_slice = seg_real[k]
        
        #Find boundary locations of validation output
        if (np.sum(calc_slice) != 0):
            calc_slice = find_boundaries(calc_slice)
            positions_calc = zip(*np.where(calc_slice>0))
            positions_calc = list(positions_calc)
            positions_calc = np.array(positions_calc)
            calc_sp_1 = space_calc[0]*space_calc[2]*(int(k+1))
            calc_sp_2 = space_calc[1]*space_calc[2]*(int(k+1))
            positions_calc = (positions_calc * [calc_sp_1, calc_sp_2])
            positions_calc = (positions_calc).tolist()
            positions_calc_fin.append(positions_calc)
            
        #Find boundary locations of original input
        if (np.sum(real_slice) != 0):
            real_slice = find_boundaries(real_slice)
            positions_real = zip(*np.where(real_slice>0))
            positions_real = list(positions_real)
            positions_real = np.array(positions_real)
            real_sp_1 = space_real[0]*space_real[2]*(int(k+1))
            real_sp_2 = space_real[1]*space_real[2]*(int(k+1))
            positions_real = (positions_real * [real_sp_1, real_sp_2])
            positions_real = positions_real.tolist()
            positions_real_fin.append(positions_real)

    #Find Directed Hausdorff, return maximum
    dh_pc_pr = (directed_hausdorff(positions_calc, positions_real)[0])
    dh_pr_pc = (directed_hausdorff(positions_real, positions_calc)[0])
    dh = max(dh_pc_pr, dh_pr_pc)
    return dh
