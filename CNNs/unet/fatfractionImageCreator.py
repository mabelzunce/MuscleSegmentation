import SimpleITK as sitk
import numpy as np
import os


dataPath ='../../Data/LumbarSpine3D/InputImages/'
outputPath = '../../Data/LumbarSpine3D/'
folder = sorted(os.listdir(dataPath))

extensionImages = '.mhd'
suffixArray = ['_F', '_I', '_O', '_W']
inPhaseSuffix = '_I'
outOfPhaseSuffix = '_O'
waterSuffix = '_W'
fatSuffix = '_F'

auxName=str
for files in folder:
    name = os.path.splitext(files)[0]
    if name.split('_')[0] != auxName:
        auxName = name.split('_')[0]
        fatImage = sitk.Cast(sitk.ReadImage(dataPath + auxName + fatSuffix + extensionImages), sitk.sitkFloat32)
        waterImage = sitk.Cast(sitk.ReadImage(dataPath + auxName + waterSuffix + extensionImages), sitk.sitkFloat32)
    else:
        continue
    waterfatImage = sitk.Add(fatImage, waterImage)
    fatfractionImage = sitk.Divide(fatImage, waterfatImage)
    fatfractionImage = sitk.Cast(sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0), sitk.sitkFloat32)

    sitk.WriteImage(fatfractionImage, outputPath + auxName + '_ff' + extensionImages)