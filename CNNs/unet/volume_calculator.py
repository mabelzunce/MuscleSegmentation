import SimpleITK as sitk
import numpy as np
import csv
import os


dataPath = 'D:/Dixon German Balerdi/resampled/'
outputPath = 'D:/Dixon German Balerdi/'

muscleNames = ['P Izq','I Izq','CL Izq','ES+M Izq','P Der','I Der','CL Der','ES+M Der','Promedio']
folder = os.listdir(dataPath)
folder = sorted(folder)
tagInPhase = '_i.mhd'
tagMask = '_segmentation.mhd'
auxName = str

with open(outputPath + 'volume.csv',mode='w', newline="") as csv_file:
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
        print(auxName)
        imageSpacing = ipImage.GetSpacing()
        voxelSize = np.product(imageSpacing)
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        voxel = []
        for n in range(maskMaxValue):
            binaryMask = sitk.BinaryThreshold(mask, n + 1, n + 1, 1, 0)
            maskSize = np.sum(sitk.GetArrayViewFromImage(binaryMask))
            voxel.append((maskSize * voxelSize) / 1000)

        csvRow = (auxName, *voxel, np.mean(voxel))
        csvRow = list(csvRow)
        csv_writer.writerow(csvRow)
