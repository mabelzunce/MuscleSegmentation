#! python3


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os
import time as time

# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\"
# Parameter files:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid_NCC'
paramFilesToTest = ['Parameters_BSpline_NCC', 'Parameters_BSpline_NCC_1000iters_4096samples', 'Parameters_BSpline_NCC_8192samples', 'Parameters_BSpline_NCC_2000iters_8192samples', 'Parameters_BSpline_NCC_4000iters_8192samples']
paramFilesToTest = ['Parameters_BSpline_NCC_4000iters_8192samples_3levels']
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



#Test all the images with all the parameter sets:
fixedImageNames = [rawImagesNames[4]]
movingImageNames = [rawImagesNames[5]]
metricValues = []
executionTimes = []
for filename in fixedImageNames:
    fixedImage = sitk.ReadImage(libraryPath + filename)
    nameFixed, extension = os.path.splitext(filename)
    # Create an output directory for each parameter file:
    outputPathThisCase = outputPath + nameFixed + "\\"
    if not os.path.exists(outputPathThisCase):
        os.mkdir(outputPathThisCase)

    sitk.WriteImage(fixedImage, outputPathThisCase + 'fixedImage.mhd')
    for filenameMoving in movingImageNames:
        movingImage = sitk.ReadImage(libraryPath + filenameMoving)
        nameMoving, extension = os.path.splitext(filenameMoving)
        if filename != filenameMoving:
            for paramFiles in paramFilesToTest:
                # Time to compute execution time:
                startTime = time.time()
                # The registration
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
                # Get metric:
                imRegMethod = sitk.ImageRegistrationMethod()
                imRegMethod.SetMetricAsCorrelation()
                metricValue = imRegMethod.MetricEvaluate(fixedImage, resultImage)
                print(metricValue)
                metricValues.append(metricValue)
                # Stop time
                stopTime = time.time()
                # Execution times:
                executionTimes.append(stopTime-startTime)
                # Write image:
                outputFilename = outputPathThisCase + '\\' + nameFixed + '_' + nameMoving + '_' + paramFiles + '.mhd'
                sitk.WriteImage(resultImage, outputFilename)

print('Similarity values: {0}'.format(metricValues))

print('Execution times: {0}'.format(executionTimes))