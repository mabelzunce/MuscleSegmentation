import SimpleITK as sitk
import numpy as np
import csv
import os


dataPath = 'D:/1LumbarSpineDixonData/2D Images/'
outputPath = 'D:/Resultados Antropometricos CSV/'

muscleNames = ['ES+M Izq','ES+M Der','CL Izq','CL Der','P Izq','P Der','Promedio']
folder = os.listdir(dataPath)
folder = sorted(folder)
tagInPhase = '_I.mhd'
tagMask = '_seg.mhd'
auxName = str

with open(outputPath + 'csa_1.csv',mode='w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    header = list(('subject', *muscleNames))
    csv_writer.writerow(header)
    for files in folder:
        name = os.path.splitext(files)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            ipImage = sitk.ReadImage(dataPath + auxName + tagInPhase)
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
        else:
            continue
        imageSpacing = ipImage.GetSpacing()
        pixelSize = np.product(imageSpacing)
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        CSA = []
        for n in range(maskMaxValue):
            binaryMask = sitk.BinaryThreshold(mask, n+1,n+1,1,0)
            maskSize = np.sum(sitk.GetArrayViewFromImage(binaryMask))
            CSA.append((maskSize * pixelSize)/100)

        csvRow = (auxName, *CSA, np.mean(CSA))
        csvRow = list(csvRow)
        csv_writer.writerow(csvRow)
