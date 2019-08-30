#! python3


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

pathWithIntermediateFiles = 'D:\MuscleSegmentationEvaluation\20190812Paper\V1.0\NonrigidNCC_N3_MaxProb_Mask\ID00010\FullIntermediates'

# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\"
# Parameter files:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid'
paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}
# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\Normalized\\"
# Look for the raw files:
files = os.listdir(libraryPath)
extensionImages = 'mhd'
rawImagesNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
#    # Use only the marathon study
#    if str(name).startswith("ID"):
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
        rawImagesNames.append(name + '.' + extensionImages)
print("Number of atlases: {0}".format(len(rawImagesNames)))
print("List of files: {0}\n".format(rawImagesNames))

# Create an output directory for each parameter file:
for paramFiles in paramFilesToTest:
    newPath = outputPath + paramFiles + "\\"
    if not os.path.exists(newPath):
        os.mkdir(newPath)

#Test all the images with all the parameter sets:
for filename in rawImagesNames:
    fixedImage = sitk.ReadImage(libraryPath + filename)
    nameFixed, extension = os.path.splitext(filename)
    for filenameMoving in rawImagesNames:
        movingImage = sitk.ReadImage(libraryPath + filenameMoving)
        nameMoving, extension = os.path.splitext(filenameMoving)
        if filename != filenameMoving:
            for paramFiles in paramFilesToTest:

                # elastixImageFilter filter
                elastixImageFilter = sitk.ElastixImageFilter()
                # Parameter maps:
                parameterMapVector = sitk.VectorOfParameterMap()
                parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                               + paramFileRigid + '.txt'))
                parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                               + paramFiles + '.txt'))
                # Registration:
                elastixImageFilter.SetFixedImage(fixedImage)
                elastixImageFilter.SetMovingImage(movingImage)
                elastixImageFilter.SetParameterMap(parameterMapVector)
                elastixImageFilter.Execute()
                # Get the images:
                resultImage = elastixImageFilter.GetResultImage()
                transformParameterMap = elastixImageFilter.GetTransformParameterMap()
                # Write image:
                outputFilename = outputPath + paramFiles + '\\' + nameFixed + '_' + nameMoving + '.mhd'
                sitk.WriteImage(resultImage, outputFilename)