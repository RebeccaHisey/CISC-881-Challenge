"""
Created on Fri Feb  7 09:10:47 2020

@author: emilyk
"""
import SimpleITK as sitk
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

dirname = Path.cwd()
from segmentationVolume import segVol
from utils import IoU
from hausdorffDistance import hausdorff_dist
from thresholdOutput import thresholdCube

np.set_printoptions(threshold=1200, suppress=True)

# Import validation output
data = list((dirname / "ValidationOutput" / "fold0").glob("*.mhd"))
file_df = pd.DataFrame(
    {
        "filename": [str(file) for file in data],
        "ct": [re.search(r"LNDb\d+", file.name).group() for file in data],
        "finding": [re.search(r"F\d+", file.name).group() for file in data],
    }
)

# Import real/reference data
# TODO do we still need this?
data_real = (dirname / "OriginalData").glob("*.npy")

vol_calc = []
vol_real = []
j_index = []
haus_dist = []

# Each group has all reviewer segmentations for the given finding
# e.g. LNDb0312, F1 -> {R1, R2, R3}
for _, file_group in file_df.groupby(["ct", "finding"]):
    group_size = file_group.shape[0]

    j_index_sum = 0
    haus_dist_sum = 0
    vol_calc_sum = 0
    vol_real_sum = 0
    for _, file in file_group.iterrows():
        # Processing for validation output
        image_calc = sitk.ReadImage(file["filename"])
        seg_calc_arr = thresholdCube(sitk.GetArrayFromImage(image_calc))
        seg_calc = sitk.GetImageFromArray(seg_calc_arr)

        # Processing for original input
        data_file = Path(file["filename"]).stem + ".npy"
        image_real_arr = np.load(dirname / "OriginalData" / data_file)
        image_real = sitk.GetImageFromArray(image_real_arr)
        seg_real_arr = np.load(dirname / "OriginalData" / ("Mask_" + data_file))
        seg_real = sitk.GetImageFromArray(seg_real_arr)

        # Volume Calculation
        vol_calc_sum += float(segVol(image_calc, seg_calc))
        vol_real_sum += float(segVol(image_real, seg_real))

        # Jaccard index and Hausdorff Distance Calculation
        j_index_sum += 1 - IoU(seg_calc_arr, seg_real_arr)
        haus_dist_sum += hausdorff_dist(
            seg_calc_arr, seg_real_arr, image_calc, image_real
        )

    # Volume Calculation
    vol_calc.append(vol_calc_sum / group_size)
    vol_real.append(vol_real_sum / group_size)

    # Jaccard index and Hausdorff Distance Calculation
    # Note that this is averaged per-finding for each reviewer
    j_index.append(j_index_sum / group_size)
    haus_dist.append(haus_dist_sum / group_size)

# Commented code was originally intended to compute performance metrics only for slices where original data had data.
# This was intended to remove some of the additional nodules our network picked up.
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

# Final Performance Calculations (Done by normalizing each metric by maximum case)
mean_j = 1 - np.mean(j_index / np.max(j_index))
mean_h = 1 - np.mean(haus_dist / np.max(haus_dist))
vol_calc = np.array(vol_calc)
vol_real = np.array(vol_real)
prsn = 1 - pearsonr(vol_calc, vol_real)[0]
bias = np.sum(abs(vol_calc - vol_real)) / len(vol_calc)
max_diff = np.max(abs(vol_calc - vol_real))
stdev = np.std(abs(vol_calc - vol_real))
print(
    "pearsonr =",
    prsn,
    "bias =",
    (1 - (bias / max_diff)),
    "standarddev =",
    (1 - (stdev / max_diff)),
    "jaccard =",
    mean_j,
    "hausdorff =",
    mean_h,
)
print((prsn + (1 - (bias / max_diff)) + (1 - (stdev / max_diff)) + mean_j + mean_h) / 5)
