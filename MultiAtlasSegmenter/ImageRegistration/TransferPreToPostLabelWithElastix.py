#! python3
# This scripts registers pre marathon cases, which had been previously manually segmented, to the respective post
# marathon scan.


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

# Data
nameCase = 'ID00061'
srcSide = 'right'
dstSide = 'left'
prePath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\Segmented\\' + nameCase + '\\ForLibrary\\'
postPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\NotSegmented\\' + nameCase + '\\ForLibrary\\'
nameImage = nameCase + '_I'
nameLabels = nameCase + '_labels'
extImages = '.mhd'
# Read images:
preImage = sitk.ReadImage(prePath + nameImage + extImages)
preLabels = sitk.ReadImage(prePath + nameLabels + extImages)
postImage = sitk.ReadImage(postPath + nameImage + extImages)

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
elastixImageFilter.SetFixedImage(postImage)
elastixImageFilter.SetMovingImage(preImage)
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
# Get the images:
resultImage = elastixImageFilter.GetResultImage()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()
# Compute normalized cross correlation:
imRegMethod = sitk.ImageRegistrationMethod()
imRegMethod.SetMetricAsCorrelation()
metricValue = imRegMethod.MetricEvaluate(sitk.Cast(postImage, sitk.sitkFloat32), sitk.Cast(resultImage, sitk.sitkFloat32))
print("Normalized cross-correlation between reflected left and right: {0}".format(metricValue))

# Transfer the labels:
# Apply its transform:
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetMovingImage(preLabels)
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
transformixImageFilter.Execute()
postLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

# Write registered image:
outputFilename = postPath + nameCase + '_pre2post.mhd'
sitk.WriteImage(resultImage, outputFilename)
# Write transferred labels:
outputFilename = postPath + nameLabels + '.mhd'
sitk.WriteImage(postLabels, outputFilename)