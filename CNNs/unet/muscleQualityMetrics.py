import SimpleITK as sitk
import numpy as np
import os
import csv

############################ DATA PATHS ##############################################
dataPath = "D:/PROYECTO FINAL/1LumbarSpineDixonData/BlandAltman/FatFractionValidacion/"
outputPath = "D:/PROYECTO FINAL/1LumbarSpineDixonData/BlandAltman/"
muscleNames = ['LP', 'LI', 'LQ', 'LM+Es', 'RP', 'RI', 'RQ', 'RM+Es', 'Avg']

files = os.listdir(dataPath)
files = sorted(files) # must be sorted, otherwise masks and images could be mixed
auxName = str
tagFatFraction = '_ff.mhd'
tagMask = '_labels.mhd'

# Abrir el archivo CSV en modo de escritura
with open(outputPath + 'labelsFatFraction.csv', mode="w", newline="") as archivo_csv:
    # Crear un objeto escritor CSV
    escritor_csv = csv.writer(archivo_csv)
    header = list(("Subject", *muscleNames))
    escritor_csv.writerow(header)

    for filename in files:
        name = os.path.splitext(filename)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            ffImage = sitk.ReadImage(dataPath + auxName + tagFatFraction)
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
        else:
            continue
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        fatFraction = []
        for n in range(maskMaxValue):
            binaryMask = sitk.BinaryThreshold(mask, n+1, n+1, 1, 0)
            maskSize = np.sum(sitk.GetArrayViewFromImage(binaryMask))
            segmentedff = sitk.Mask(ffImage, binaryMask)
            ffSum = np.sum(sitk.GetArrayViewFromImage(segmentedff))
            fatFraction.append(ffSum/maskSize)
        csvRow = (auxName, *fatFraction, np.mean(fatFraction))
        csvRow = list(csvRow)
        # Escribir la lista en el archivo CSV
        escritor_csv.writerow(csvRow)
    archivo_csv.close

with open(outputPath + 'labelsSize.csv', mode="w", newline="") as archivo_csv:
    # Crear un objeto escritor CSV
    escritor_csv = csv.writer(archivo_csv)
    header = list(("Subject",*muscleNames))
    escritor_csv.writerow(header)

    for filename in files:
        name = os.path.splitext(filename)[0]
        if name.split('_')[0] != auxName:
            auxName = name.split('_')[0]
            ffImage = sitk.ReadImage(dataPath + auxName + tagFatFraction)
            mask = sitk.ReadImage(dataPath + auxName + tagMask)
            voxelSize = np.prod(list(mask.GetSpacing()))
        else:
            continue
        maskMaxValue = np.max(sitk.GetArrayViewFromImage(mask)).astype(np.uint8)
        size = []
        for n in range(maskMaxValue):
            binaryMask = sitk.BinaryThreshold(mask, n+1, n+1, 1, 0)
            maskSize = np.sum(sitk.GetArrayViewFromImage(binaryMask))
            maskSize = maskSize * voxelSize/1000
            size.append(maskSize)
        csvRow = (auxName, *size, np.mean(size))
        csvRow = list(csvRow)
        # Escribir la lista en el archivo CSV
        escritor_csv.writerow(csvRow)
    archivo_csv.close