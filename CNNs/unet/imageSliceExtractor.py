import os
import csv
import SimpleITK as sitk

dataPath = 'D:/1LumbarSpineDixonData/3D Images/'
outPath = 'D:/1LumbarSpineDixonData/2D Images/'
extension = '_I.mhd'

folder = sorted(os.listdir(dataPath))

with open(dataPath + 'SliceHeight.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) #skip header

    for row in csv_reader:
        name = row[0].split(',')[0]
        height = int(row[0].split(',')[1])
        slice = sitk.ReadImage(dataPath + name + extension)[:,:,height]
        sitk.WriteImage(slice, outPath + name + '_I.mhd')
        print(name)

