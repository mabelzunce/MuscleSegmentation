import SimpleITK as sitk
import numpy as np
import csv
import os
from utils import maskSize


dataPath ='/media/german/SSD Externo/1LumbarSpineDixonData/'
outputPath ='/media/german/SSD Externo/LumbarSpineDixonDataResampled/'

muscleNames = ['Left P','Left I','Left QL','Left ES+M','Right P','Right I','Right QL','Right ES+M']
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
