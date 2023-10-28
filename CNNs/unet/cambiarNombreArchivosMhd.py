import SimpleITK as sitk
import os

dataPath = 'D:/MuscleSegmentation/Data/LumbarSpine2D/TrainingSet/'
outputPath = 'D:/MuscleSegmentation/Data/LumbarSpine2D/ManualSegmentations/'
files = os.listdir(dataPath)
taglabels= 'labels'
for filename in files:
    name, extension = os.path.splitext(filename)
    name1 = name.split('_')[0]
    name2 = name.split('_')[1]
    name3 = name.split('_')[-1]
    if not name.endswith(taglabels) or  extension.endswith('raw') or not name2.__contains__(name1):
        continue
    filenameImage = dataPath + filename
    sitkImage = sitk.ReadImage(filenameImage)
    sitk.WriteImage(sitkImage, outputPath + name1 + '_labels.mhd')