# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:10:47 2020
@author: emilyk
"""
import SimpleITK as sitk
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

dirname = Path.cwd()
from segmentationVolume import segVol
from utils import IoU #utils script used from iW-Net, by github user gmaresta
from hausdorffDistance import hausdorff_dist
from thresholdOutput import thresholdCube, labelSegmentation, findLargestNodule

pears = []
haus = []
jacc = []
bi = []
std = []

np.set_printoptions(threshold=1200, suppress=True)

fold = 4    

for t in range(fold):
    print('Fold', t)
    # Import validation output
    data = list((dirname / "ValidationOutput" / ("fold" + str(t))).glob("*.mhd"))
    file_df = pd.DataFrame(
        {
            "filename": [str(file) for file in data],
            "ct": [re.search(r"LNDb\d+", file.name).group() for file in data],
            "finding": [re.search(r"F\d+", file.name).group() for file in data],
        }
    )
    
    # Import real/reference data

    vol_calc = []
    vol_real = []
    j_index = []
    haus_dist = []
    
    #Brian added following for loop and modified code to average metrics for reviewers
    # Each group has all reviewer segmentations for the given finding
    # e.g. LNDb0312, F1 -> {R1, R2, R3}
    for (ct, finding), file_group in file_df.groupby(["ct", "finding"]): 
        
        j_index_sum = 0
        haus_dist_sum = 0
        vol_calc_sum = 0
        vol_real_sum = 0
        group_size = file_group.shape[0]
        for _, file in file_group.iterrows():
            # Processing for validation output
            image_calc = sitk.ReadImage(file["filename"])
            seg_calc_arr = thresholdCube(sitk.GetArrayFromImage(image_calc))
            seg_calc_arr = labelSegmentation(seg_calc_arr)
            seg_calc_arr = findLargestNodule(seg_calc_arr)
            seg_calc = sitk.GetImageFromArray(seg_calc_arr)
    
            # Processing for original input
            data_file = Path(file["filename"]).stem + ".npy"
            image_real_arr = np.load(dirname / "OriginalData" / data_file)
            image_real = sitk.GetImageFromArray(image_real_arr)
            seg_real_arr = np.load(dirname / "OriginalData" / ("Mask_" + data_file))
            seg_real = sitk.GetImageFromArray(seg_real_arr)
    
            # Volume, Jaccard index and Hausdorff Distance Calculation
            if float(segVol(image_real, seg_real)) <= 2 or float(segVol(image_real, seg_real)) == 0:
                group_size = group_size -1
    
            else:
                vol_calc_sum += float(segVol(image_calc, seg_calc))
                vol_real_sum += float(segVol(image_real, seg_real))
                haus_dist_sum += hausdorff_dist(
                    seg_calc_arr, seg_real_arr, image_calc, image_real
                    )
                j_index_sum += 1 - IoU(seg_calc_arr, seg_real_arr)
                
        # Volume Calculation
        if group_size > 0:
            vol_calc.append(vol_calc_sum / group_size)
            vol_real.append(vol_real_sum / group_size)
    
        # Jaccard index and Hausdorff Distance Calculation
        # Note that this is averaged per-finding for each reviewer
            j_index.append(j_index_sum / group_size)
            haus_dist.append(haus_dist_sum / group_size)
    
    # Commented code was originally intended to compute performance metrics only for slices where original data had data.
    # This was intended to remove some of the additional nodules our network picked up before function to remove other nodules was created
    #    count = 0
    #    for t in range(len(seg_real_arr)):
    #        if (np.sum(seg_real_arr[t]) > 0 and count == 0):
    #            start = t
    #            count = 1
    #        if (np.sum(seg_real_arr[t]) == 0 and count == 1):
    #            end = t
    #            break
    #    j_index.append(IoU(seg_calc_arr[start:end], seg_real_arr[start:end]))
    #    vol_calc.append(float(segVol(image_calc[start:end], seg_calc[start:end])))
    #    vol_real.append(float(segVol(image_real, seg_real)))
    #    haus_dist.append(hausdorff_dist(seg_calc_arr[start:end], seg_real_arr[start:end], image_calc[start:end], image_real[start:end]))
    
    # Final Performance Calculations 
    #Calculate Jaccard Index and Hausdorff Distance
    mean_j = np.mean(j_index)
    mean_h = np.mean(haus_dist)
    vol_calc = np.array(vol_calc)
    vol_real = np.array(vol_real)

    #Scatter Plot to see outliers
    plt.figure(1)
    plt.scatter(vol_calc, vol_real)
    
    #Pearson correlation coefficient, bias, and standard deviation calculation
    prsn = (pearsonr(vol_calc, vol_real)[0])
    bias = np.sum((abs(vol_calc - vol_real))) / len(vol_calc)
    stdev = np.std(((vol_calc - vol_real)))
    
    #Performance for current fold
    print(
        "pearsonr =",
        prsn,
        "bias =",
        ((bias)),
        "standarddev =",
        ((stdev)),
        "jaccard =",
        mean_j,
        "hausdorff =",
        mean_h,
    )
    bi.append(bias)
    pears.append(prsn)
    jacc.append(mean_j)
    haus.append(mean_h)
    std.append(stdev)
    
#Final performance with average for four folds
print(
        "pearsonr =",
        np.mean(pears),
        "bias =",
        np.mean(bi),
        "standarddev =",
        np.mean(std),
        "jaccard =",
        np.mean(jacc),
        "hausdorff =",
        np.mean(haus),
    )
