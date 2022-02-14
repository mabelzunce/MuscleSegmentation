#! python3
# This scripts registers pre marathon cases, which had been previously manually segmented, to the respective post
# marathon scan.


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os
# save files when debugging:
DEBUG = True

# Data
nameCase = 'ID00060'
srcSide = 'right'
dstSide = 'left'
dataPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\NotSegmented\\' + nameCase + '\\ForLibrary\\'
nameImage = nameCase + '_I'
nameLabels = nameCase + '_labels_' + srcSide
extImages = '.mhd'
# Read images:
image = sitk.ReadImage(dataPath + nameImage + extImages)
labels = sitk.ReadImage(dataPath + nameLabels + extImages)
# Array to replace at the end:
ndaLabels = sitk.GetArrayFromImage(labels)


# Divide the images in two halves:
imageSize = image.GetSize()
halfImageSize = np.round([imageSize[0]/2, imageSize[1], imageSize[2]]).astype('int')

rightImage = image[: halfImageSize[0], :, :]
rightLabels = labels[: halfImageSize[0], :, :]
leftImage = image[halfImageSize[0]-1:, :, :]
leftLabels = labels[halfImageSize[0]-1:, :, :]

if DEBUG:
    ndaImage = sitk.GetArrayFromImage(image)
    ndaRightImage = sitk.GetArrayFromImage(rightImage)
    ndaLeftImage = sitk.GetArrayFromImage(leftImage)
    ndaImage[:, :, : halfImageSize[0]] = ndaRightImage
    ndaImage[:, :, halfImageSize[0]-1:] = ndaLeftImage
    # Write half iamges:
    testImage = sitk.GetImageFromArray(ndaImage)
    testImage.CopyInformation(image)
    sitk.WriteImage(testImage, dataPath + nameCase + '_resorted' + '.mhd')
    sitk.WriteImage(leftImage, dataPath + nameCase + '_left' + '.mhd')
    sitk.WriteImage(rightImage, dataPath + nameCase + '_right' + '.mhd')

if srcSide == 'left':
    srcImage = leftImage
    srcLabels = leftLabels
    dstImage = rightImage
    dstLabels = rightLabels
else:
    srcImage = rightImage
    srcLabels = rightLabels
    dstImage = leftImage
    dstLabels = leftLabels

# Invert horizontal axis to help the registration:
originalDirection = srcImage.GetDirection()
srcImage = sitk.Flip(srcImage, [True, False, False])
srcLabels = sitk.Flip(srcLabels, [True, False, False])
# Flip changes the voxel order but also the coordinate system, what it doesn't make sense, so I changed it back manually.
srcImage.SetDirection(originalDirection)
srcLabels.SetDirection(originalDirection)
#leftImage = leftImage[::-1,:,:]
#leftLabels = leftLabels[::-1,:,:]

sitk.WriteImage(srcImage, dataPath + nameCase + '_' + srcSide + '_reflected' + '.mhd')

# Parameter files for the registration:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid_NMI'
paramFileAffine = 'Parameters_Affine_NMI'
paramFileNonRigid = 'Parameters_BSpline_NMI_4000iters_8192samples'#{,'Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}
paramFileNonRigid = 'WithDeformationPenalty\\Parameters_BSpline_NCC_1000iters_2048samples'

# elastixImageFilter filter
elastixImageFilter = sitk.ElastixImageFilter()
# Parameter maps:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                               + paramFileRigid + '.txt'))
parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                               + paramFileAffine + '.txt'))
parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                               + paramFileNonRigid + '.txt'))
# Registration:
elastixImageFilter.SetFixedImage(dstImage)
elastixImageFilter.SetMovingImage(srcImage)
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
# Get the image and the transform:
resultImage = elastixImageFilter.GetResultImage()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()
# Compute normalized cross correlation:
imRegMethod = sitk.ImageRegistrationMethod()
imRegMethod.SetMetricAsCorrelation()
metricValue = imRegMethod.MetricEvaluate(sitk.Cast(srcImage, sitk.sitkFloat32), sitk.Cast(resultImage, sitk.sitkFloat32))
print("Normalized cross-correlation between reflected left and right: {0}".format(metricValue))

# Transfer the labels:
# Apply its transform:
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetMovingImage(srcLabels)
transformixImageFilter.SetTransformParameterMap(transformParameterMap)
transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
transformixImageFilter.Execute()
dstLabelsFromSrc = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

if DEBUG:
    # Write registered image:
    sitk.WriteImage(resultImage, dataPath + nameCase + '_' + srcSide + '2' + dstSide + '.mhd')
    sitk.WriteImage(dstLabelsFromSrc, dataPath + nameCase + '_labels_' + srcSide + '2' + dstSide + '.mhd')

ndaDstLabels = sitk.GetArrayFromImage(dstLabelsFromSrc)
ndaDstImage = sitk.GetArrayFromImage(resultImage)
print("Sahpes: {0},{1}".format(ndaLabels.shape, ndaDstLabels.shape))


# Now complete the image:
if srcSide == 'left':
    # Adjust labels:
    ndaDstLabels[(ndaDstLabels < 5) & (ndaDstLabels > 0)] = ndaDstLabels[(ndaDstLabels < 5) & (ndaDstLabels > 0)] + 4
    ndaDstLabels[ndaDstLabels == 9] = 10
    ndaLabels[:, :, : halfImageSize[0]] = ndaDstLabels
    if DEBUG:
        ndaImage[:, :, : halfImageSize[0]] = ndaDstImage
else:
    ndaDstLabels[(ndaDstLabels < 9) & (ndaDstLabels > 4)] = ndaDstLabels[(ndaDstLabels < 9) & (ndaDstLabels > 4)] - 4
    ndaDstLabels[ndaDstLabels == 10] = 9
    ndaLabels[:, :, halfImageSize[0]-1:] = ndaDstLabels
    if DEBUG:
        ndaImage[:, :, halfImageSize[0]-1:] = ndaDstImage

if DEBUG:
    testImage = sitk.GetImageFromArray(ndaImage)
    testImage.CopyInformation(image)
    sitk.WriteImage(testImage, dataPath + nameCase + '_resorted_after_reg' + '.mhd')


newLabels = sitk.GetImageFromArray(ndaLabels)
newLabels.CopyInformation(labels)
sitk.WriteImage(newLabels, dataPath + nameCase + '_labels_transferred_' + srcSide + '2' + dstSide + '.mhd')