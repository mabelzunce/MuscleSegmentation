import SimpleITK as sitk
import numpy as np
import csv
import os
from utils import maskSize


dataPath ='D:/1LumbarSpineDixonData/2DManualSegmentations/'
outputPath ='D:/1LumbarSpineDixonData/'

muscleNames = ['Left P','Right P','Left QL','Right QL','Left ES+M','Right ES+M']
folder = os.listdir(dataPath)
folder = sorted(folder)
tagMask = '_labels.mhd'
auxName = str

with open(outputPath + 'CSA_ManualSegmentations2D.csv', mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject', *muscleNames))
    csv_writer.writerow(header)
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
            maskArray = sitk.GetArrayFromImage(mask).astype(np.float64)
        else:
            continue
        segmentationSize = maskSize(maskArray)
        voxelSize = np.prod(list(mask.GetSpacing()))
        muscleSize = segmentationSize * voxelSize/100
        csvRow = (auxName, *muscleSize)
        csvRow = list(csvRow)
        print(auxName)
        csv_writer.writerow(csvRow)
