#! python3


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V0.5\\Normalized\\"
# Look for the raw files:
files = os.listdir(libraryPath)
extensionImages = 'mhd'
rawImagesNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
        rawImagesNames.append(name + '.' + extensionImages)
print("Number of atlases: {0}".format(len(rawImagesNames)))
print("List of files: {0}\n".format(rawImagesNames))

# Concatenate the ND images into one (N+1)D image

vectorOfImages = sitk.VectorOfImage()
# Use only 5 images:
#for filename in rawImagesNames:
for i in range(0,4):
    vectorOfImages.push_back(sitk.ReadImage(libraryPath + rawImagesNames[i]))

image = sitk.JoinSeries(vectorOfImages)

# Register
elastixImageFilter = sitk.ElastixImageFilter()
print(elastixImageFilter.GetOutputDirectory())
elastixImageFilter.SetFixedImage(image)
elastixImageFilter.SetMovingImage(image)
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
elastixImageFilter.Execute()


# Get the images:
resultImage = elastixImageFilter.GetResultImage()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()

# Write image:
sitk.WriteImage(resultImage, "averageImage.mhd")


# fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
# moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
#
# transformDomainMeshSize=[8]*moving.GetDimension()
# tx = sitk.BSplineTransformInitializer(fixed,
#                                       transformDomainMeshSize )
#
# print("Initial Parameters:");
# print(tx.GetParameters())
#
# R = sitk.ImageRegistrationMethod()
# R.SetMetricAsCorrelation()
#
# R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
#                        numberOfIterations=100,
#                        maximumNumberOfCorrections=5,
#                        maximumNumberOfFunctionEvaluations=1000,
#                        costFunctionConvergenceFactor=1e+7)
# R.SetInitialTransform(tx, True)
# R.SetInterpolator(sitk.sitkLinear)
#
# R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
#
# outTx = R.Execute(fixed, moving)
#
# print("-------")
# print(outTx)
# print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
# print(" Iteration: {0}".format(R.GetOptimizerIteration()))
# print(" Metric value: {0}".format(R.GetMetricValue()))
#
# sitk.WriteTransform(outTx,  sys.argv[3])
#
# if ( not "SITK_NOSHOW" in os.environ ):
#
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed);
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(100)
#     resampler.SetTransform(outTx)
#
#     out = resampler.Execute(moving)
#     simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
#     simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
#     cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
#     sitk.Show( cimg, "ImageRegistration1 Composition" )