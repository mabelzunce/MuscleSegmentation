#! python3
# This scripts registers pre marathon cases, which had been previously manually segmented, to the respective post
# marathon scan.


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

nameFixed = 'ID00003'
nameMoving = '7390413'
imagesPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\'
movingImagesPath = 'D:\\Martin\\Data\\MuscleSegmentation\\RNOH_TLC\\GoodToUse\\Segmented\\'
fixedImage = sitk.ReadImage(imagesPath + nameFixed + '_bias.mhd')
# Read moving images:
tagsOfImagesToUse = ['_I', '_O']
vectorOfImages = sitk.VectorOfImage()
for tags in tagsOfImagesToUse:
    vectorOfImages.push_back(sitk.ReadImage(movingImagesPath + nameMoving + '\\ForLibrary\\' + nameMoving + tags + '.mhd'))
# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\testMultipleIO\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Parameter files:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid_NCC'
paramFileNonRigid = 'Parameters_BSpline_NCC_4000iters_8192samples'#{,'Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}


# elastixImageFilter filter
elastixImageFilter = sitk.ElastixImageFilter()
# Parameter maps:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                               + paramFileRigid + '.txt'))
parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                               + paramFileNonRigid + '.txt'))
# Registration:
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(vectorOfImages)
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
# Get the images:
resultImage = elastixImageFilter.GetResultImage()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()
# Write image:
outputFilename = outputPath + nameFixed + '_' + nameMoving + '.mhd'
sitk.WriteImage(resultImage, outputFilename)
outputFilename = outputPath + nameFixed + '.mhd'
sitk.WriteImage(fixedImage, outputFilename)