import SimpleITK as sitk
import numpy as np
import csv
import os
from utils import dice2d


dataPath ='D:/1LumbarSpineDixonData/2D Images/3DsegSlice/'
outputPath ='D:/1LumbarSpineDixonData/'

muscleNames = ['Left Multifidus','Right Multifidus','Left Quadratus','Right Quadratus','Left Psoas','Right Psoas','Avg']
folder = os.listdir(dataPath)
folder = sorted(folder)
tag2d = '_seg.mhd'
tag3d = '_3DsegSlice.mhd'
auxName = str

with open(outputPath + 'intermodelDice.csv',mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject', *muscleNames))
    csv_writer.writerow(header)
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            seg2d = sitk.ReadImage(dataPath + auxName + tag2d)
            seg3d = sitk.ReadImage(dataPath + auxName + tag3d)
        else:
            continue
        seg2d = sitk.GetArrayFromImage(seg2d)
        seg3d = sitk.GetArrayFromImage(seg3d)
        diceScore =[]
        for n in range(6):
            seg2 = (seg2d == n+1) * 1
            seg3 = (seg3d == n + 1) * 1
            diceScore.append(dice2d(seg2,seg3))
        diceScore.append(np.mean(diceScore))
        row = (auxName,*list(diceScore))
        csv_writer.writerow(row)