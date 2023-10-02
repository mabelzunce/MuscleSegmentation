import SimpleITK as sitk
import numpy as np
import csv
import os


dataPath = 'D:/1LumbarSpineDixonData/3D Images Resampled/'
outputPath = 'D:/1LumbarSpineDixonData/'

muscleNames = ['Left P','Left I','Left QL','Left ES+M','Right P','Right I','Right QL','Right ES+M','Avg']
folder = os.listdir(dataPath)
folder = sorted(folder)
tagFatFraction = '_ff.mhd'
tagMask = '_seg.mhd'
auxName = str

with open(outputPath + 'fatfraction3d.csv',mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject', *muscleNames))
    csv_writer.writerow(header)
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            ffImage = sitk.ReadImage(dataPath + auxName + tagFatFraction)
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
        else:
            continue
        print(auxName)
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        fatFraction = []
        for n in range(maskMaxValue):
            binaryMask = sitk.BinaryThreshold(mask, n+1,n+1,1,0)
            maskSize = np.sum(sitk.GetArrayViewFromImage(binaryMask))
            segmentedFF = sitk.Mask(ffImage, binaryMask)
            ffSum = np.sum(sitk.GetArrayViewFromImage(segmentedFF))
            fatFraction.append(ffSum/maskSize)
        csvRow = (auxName, *fatFraction, np.mean(fatFraction))
        csvRow = list(csvRow)
        csv_writer.writerow(csvRow)
