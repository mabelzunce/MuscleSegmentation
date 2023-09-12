import SimpleITK as sitk
import numpy as np
import csv
import os


dataPath ='../../Data/LumbarSpine3D/InputImages/'
outputPath ='../../Data/LumbarSpine3D/'

muscleNames = ['Left Psoas','Left Iliacus','Left Quadratus','Left Multifidus','Right Psoas','Right Iliacus','Right Quadratus','Right Multifidus','Avg']
folder  = os.listdir(dataPath)
folder = sorted(folder)
tagFatFraction = '_ff.mhd'
tagMask = '_seg.mhd'
auxName = str

with open(outputPath + 'fatfraction.csv',mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject' , *muscleNames))
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            ffImage = sitk.ReadImage(dataPath + auxName + tagFatFraction)
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
        else:
            continue
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
        csv_writer.writerows([header, csvRow])
