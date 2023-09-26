import SimpleITK as sitk
import numpy as np
import csv
import os
from utils import maskSize


dataPath ='../../Data/LumbarSpine3D/InputImages/'
outputPath ='../../Data/LumbarSpine3D/'

muscleNames = ['Left Psoas','Left Iliacus','Left Quadratus','Left Multifidus','Right Psoas','Right Iliacus','Right Quadratus','Right Multifidus']
folder = os.listdir(dataPath)
folder = sorted(folder)
tagMask = '_seg.mhd'
auxName = str

with open(outputPath + 'volume.csv', mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject', *muscleNames))
    csv_writer.writerow(header)
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
            maskArray = sitk.GetArrayFromImage(mask)
        else:
            continue
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        muscleSize = maskSize(maskArray)
        csvRow = (auxName, *muscleSize)
        csvRow = list(csvRow)
        csv_writer.writerow(csvRow)
