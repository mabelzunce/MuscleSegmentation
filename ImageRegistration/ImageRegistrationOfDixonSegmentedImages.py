#! python3


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os
import time
import csv
sys.path.append('..\MultiAtlasSegmentation')
import DixonTissueSegmentation as DixonTissueSeg
import SitkImageManipulation as sitkIm
import SegmentationPerformanceMetrics as segmentationMetrics
nameFixed = 'ID00061'
nameMoving = 'ID00001'
############################### TARGET FOLDER ###################################
libraryVersion = 'V1.2'
libraryFolder = '\\Rigid\\' #''\\NativeResolutionAndSize\\'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\'
# Look for the raw files in the library:
# First check if the folder OutOfLibrary exist and has atlases (can be there because of an aborted run, and if the atlas
# is not copied back, the library will be incomplete:
extensionImages = 'mhd'
extensionImagesBin = 'raw'
if os.path.exists(targetPath + 'OutOfLibrary\\'):
    files = os.listdir(targetPath + 'OutOfLibrary\\')
    for filename in files:
        name, extension = os.path.splitext(filename)
        # if its an image copy it back to the main folder:
        if str(extension).endswith(extensionImages) or str(extension).endswith(extensionImagesBin):
            os.rename(targetPath + 'OutOfLibrary\\' + filename, targetPath + filename)
# Now get the name of all the atlases in the library:
files = os.listdir(targetPath)
targetImagesNames = []
targetLabelsNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
#    # Use only the marathon study
#    if str(name).startswith("ID"):
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
        # Intensity image:
        targetImagesNames.append(name + '.' + extensionImages)
        # Label image:
        targetLabelsNames.append(name + '_labels.' + extensionImages)

targetImagesNames = targetImagesNames[0:4] # Just 4 images for faster testing.
print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of files: {0}\n".format(targetImagesNames))

fixedImage =  sitk.ReadImage(targetPath + nameFixed + '.mhd')
# Include labels also:
fixedLabels =  sitk.ReadImage(targetPath + nameFixed + '_labels.mhd')
numLabels = 10
# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\test\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

################################ PARAMETERS ########################################
# Parameter files for the registration:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
subfolderNonRigidTest = ''#''WithDeformationPenalty\\'
paramFileRigid = 'Parameters_Rigid_NCC'
useAffine = True
paramFileAffine = 'Parameters_Affine_NCC'

paramFileNonRigid = ('Par0000bspline', 'Parameters_BSpline_NCC_1000iters_2048samples_5mmgrid_fast_RndmCoord',
                     'Parameters_BSpline_NCC_1000iters_2048samples', 'Parameters_BSpline_NCC_1000iters_2048samples_5mmgrid',
                     'Parameters_BSpline_NCC_1000iters_2048samples_5mmgrid_fast')
paramFileNonRigid = ['Par0000bspline', 'Par0000bspline_ncc', 'Par0023_Deformable', 'Par0023_Deformable_ncc']
paramFileNonRigid = ['BSplineStandardGradDesc_NCC_1000iters_2000samples']
paramFileNonRigid = ['Par0000bspline_ncc', 'Par0023_Deformable', 'Par0023_Deformable_ncc','Parameters_BSpline_NCC_2000iters_4096samples','Parameters_BSpline_NMI_2000iters_4096samples',
                     'BSplineStandardGradDesc_NMI_2000iters_3000samples','BSplineStandardGradDesc_NMI_2000iters_3000samples_15mm','BSplineStandardGradDesc_NMI_2000iters_3000samples_15mm_RndSparseMask',
                    'Parameters_BSpline_NCC_4000iters_8192samples_3levels']
#paramFileNonRigid = 'WithDeformationPenalty\\Parameters_BSpline_NCC_1000iters_2048samples'

################################ MASKS ###########################################
# Create masks to be used in the registration:
maskFixed = DixonTissueSeg.GetBodyMaskFromInPhaseDixon(fixedImage)
maskFixedSoftTissue = DixonTissueSeg.GetSoftTissueMaskFromInPhaseDixon(fixedImage)

outputPath = outputPath + nameFixed + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
sitk.WriteImage(maskFixed, outputPath + 'maskFixed.mhd', True)
sitk.WriteImage(fixedImage, outputPath + 'fixed.mhd', True)
sitk.WriteImage(fixedLabels, outputPath + 'fixedLabels.mhd', True)
################################## REGISTRATION WITH DEFORMATION #######################################
# Evaluate parameters of registration methods with deformation penalty:
metric1WeightValues = ([20, 40, 60, 100], [20, 40, 60, 100])
metric1WeightValues = ([0],[0])
iterationValues = [2000]
numberOfSamples = [3000]
finalGridSpacingValues_mm = [5, 15]#(FinalGridSpacingInPhysicalUnits 5.0 5.0 5.0)


for i in range(1, len(targetImagesNames)):
    # Create csv files for output:
    # Read target image:
    targetFilename = targetImagesNames[i]
    targetImageFilename = targetPath + targetFilename
    movingImage = sitk.ReadImage(targetImageFilename)
    fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
    path, filename = os.path.split(targetImageFilename)
    nameMoving, extension = os.path.splitext(filename)
    outputPathThisCase = outputPath + nameMoving + '\\'
    if not os.path.exists(outputPathThisCase):
        os.makedirs(outputPathThisCase)

    movingImage = sitk.ReadImage(targetPath + nameMoving + '.mhd')
    # Include labels also:
    movingLabels = sitk.ReadImage(targetPath + nameMoving + '_labels.mhd')

    sitk.WriteImage(movingImage, outputPathThisCase + 'moving.mhd', True)
    sitk.WriteImage(movingLabels, outputPathThisCase + 'movingLabels.mhd', True)

    maskMoving = DixonTissueSeg.GetBodyMaskFromInPhaseDixon(movingImage)
    maskMovingSoftTissue = DixonTissueSeg.GetSoftTissueMaskFromInPhaseDixon(movingImage)

    sitk.WriteImage(maskMoving, outputPathThisCase + 'maskMoving.mhd', True)
    for i in range(0, len(paramFileNonRigid)):
        for useMask in [0, 1, 2]:  # Three levels of mask, 0: nonMask, 1: mask only in the fixed, 2: mask in both
            paramsNonRigid = paramFileNonRigid[i]
            # This parameter file:
            outputPathThisFile = outputPathThisCase + paramsNonRigid + '\\mask_{0}'.format(useMask) + "\\Affine_{0}".format(useAffine) + '\\'
            if not os.path.exists(outputPathThisFile):
                os.makedirs(outputPathThisFile)
            # Create a log file:
            logFilename = outputPathThisFile + 'log.txt'
            log = open(logFilename, 'w')
            fMethods = open(outputPathThisFile + 'fMethods.csv', 'w')
            fDice = open(outputPathThisFile + 'dice.csv', 'w')
            fSensitivity = open(outputPathThisFile + 'sensitivity.csv', 'w')
            fPrecision = open(outputPathThisFile + 'precision.csv', 'w')
            csvWriterMethods = csv.writer(fMethods)
            csvWriterDice = csv.writer(fDice)
            csvWriterSensitivity = csv.writer(fSensitivity)
            csvWriterPrecision = csv.writer(fPrecision)
            for weigthDeformity in metric1WeightValues[i]:
                for iterations in iterationValues:
                    for numSamples in numberOfSamples:
                        for finalGridSpacing_mm in finalGridSpacingValues_mm:
                            log.write('############### Regitrations with {0} iteration, {1} samples and {2} deformity weight ############\n'.format(iterations, numSamples,weigthDeformity ))

                            # elastixImageFilter filter
                            elastixImageFilter = sitk.ElastixImageFilter()
                            # Parameter maps:
                            parameterMapVector = sitk.VectorOfParameterMap()
                            parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                                           + paramFileRigid + '.txt'))
                            # parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                            #                                                               + paramFileAffine + '.txt'))
                            parametersBspline = elastixImageFilter.ReadParameterFile(parameterFilesPath + subfolderNonRigidTest
                                                                     + paramsNonRigid + '.txt')

                            # Change weights for deformation:
                            parametersBspline['Metric1Weight'] = [str(weigthDeformity)]
                            # Change number of iterations:
                            parametersBspline['MaximumNumberOfIterations'] = [str(iterations)]
                            parametersBspline['NumberOfSpatialSamples'] = [str(numSamples)]
                            # Change the final grid spacing:
                            parametersBspline['FinalGridSpacingInPhysicalUnits'] = [str(finalGridSpacing_mm)]

                            parameterMapVector.append(parametersBspline)

                            outputPathThisReg = outputPathThisFile + 'iter{0}_samples{1}_defWeight{2}_finGridmm{3}'.format(iterations,
                                                                                                              numSamples,
                                                                                                              weigthDeformity,
                                                                                                              finalGridSpacing_mm) + "\\"
                            # Registration:
                            #elastixImageFilter.SetLogFileName(outputPathThisReg)
                            #elastixImageFilter.LogToFileOn()
                            elastixImageFilter.SetFixedImage(fixedImage)
                            elastixImageFilter.SetMovingImage(movingImage)
                            elastixImageFilter.SetParameterMap(parameterMapVector)
                            # Set mask if needed:
                            if useMask > 0:
                                elastixImageFilter.SetFixedMask(maskFixed)
                            if useMask > 1:
                                elastixImageFilter.SetMovingMask(maskMoving)
                            if useMask > 2:
                                elastixImageFilter.SetFixedMask(maskFixedSoftTissue)
                                elastixImageFilter.SetMovingMask(maskMovingSoftTissue)

                            if not os.path.exists(outputPathThisReg):
                                os.makedirs(outputPathThisReg)
                            #elastixImageFilter.LogToConsoleOff()
                            #elastixImageFilter.LogToFileOn()
                            # Elastix can't create the file:
                            #dummy = open(outputPathThisReg + 'elastix.log', 'w')
                            #dummy.close()
                            #elastixImageFilter.SetLogFileName(outputPathThisReg + 'elastix.log')

                            # Execute
                            startTime = time.time()
                            elastixImageFilter.Execute()
                            endTime = time.time()
                            print("Registration time for {0}: {1} sec\n".format(paramsNonRigid, endTime-startTime))
                            log.write("Registration time: {0} sec\n".format(endTime-startTime))
                            # Get the images:
                            resultImage = elastixImageFilter.GetResultImage()
                            transformParameterMap = elastixImageFilter.GetTransformParameterMap()

                            # Get metric values:
                            imRegMethod = sitk.ImageRegistrationMethod()
                            imRegMethod.SetMetricAsCorrelation()
                            metricNCCValue = imRegMethod.MetricEvaluate(fixedImage, resultImage)
                            imRegMethod.SetMetricAsMattesMutualInformation()
                            metricNMIValue = imRegMethod.MetricEvaluate(fixedImage, resultImage)
                            print("NCC: {0}. NMI: {1}\n".format(metricNCCValue, metricNMIValue))
                            log.write("Metrics. NCC: {0}. NMI: {1}\n".format(metricNCCValue, metricNMIValue))
                            # Get metric values with masks:
                            imRegMethod = sitk.ImageRegistrationMethod()
                            imRegMethod.SetMetricMovingMask(maskMoving)
                            imRegMethod.SetMetricFixedMask(maskFixed)
                            imRegMethod.SetMetricAsCorrelation()
                            metricNCCValue = imRegMethod.MetricEvaluate(fixedImage, resultImage)
                            imRegMethod.SetMetricAsMattesMutualInformation()
                            metricNMIValue = imRegMethod.MetricEvaluate(fixedImage, resultImage)
                            print("NCC mask: {0}. NMI mask: {1}\n".format(metricNCCValue, metricNMIValue))
                            log.write("Metrics in mask. NCC: {0}. NMI: {1}\n".format(metricNCCValue, metricNMIValue))

                            # Write image:
                            outputFilename = outputPathThisReg + nameFixed + '_' + nameMoving + '.mhd'
                            sitk.WriteImage(resultImage, outputFilename, True)
                            # Write transform:
                            sitk.WriteParameterFile(transformParameterMap[0], outputPathThisReg + 'Transform')
                            # Now transfer labels to get dice scores:
                            # Apply its transform:
                            transformixImageFilter = sitk.TransformixImageFilter()
                            transformixImageFilter.LogToConsoleOff()
                            transformixImageFilter.SetMovingImage(movingLabels)
                            transformixImageFilter.SetTransformParameterMap(transformParameterMap)
                            transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
                            transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
                            transformixImageFilter.Execute()
                            propagatedLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
                            outputFilename = outputPathThisReg + nameFixed + '_' + nameMoving + '_labels.mhd'
                            sitk.WriteImage(propagatedLabels, outputFilename, True)

                            # Get metrics:
                            # first overall:
                            metrics = segmentationMetrics.GetOverlapMetrics(fixedLabels, propagatedLabels, 0)
                            metricsByLabel = segmentationMetrics.GetOverlapMetrics(fixedLabels, propagatedLabels, numLabels)
                            dice = metrics['dice']
                            volumeSimilarity = metrics['volumeSimilarity']
                            sensitivity = metrics['sensitivity']
                            precision = metrics['precision']
                            print("Overlap Similarity Metrics. Dice: {0}, Volume Similarity: {1}, Sensitivity:{2}, Precision:{3}\n\n".format(dice, volumeSimilarity, sensitivity, precision))
                            log.write("Overlap Similarity Metrics. Dice: {0}, Volume Similarity: {1}, Sensitivity:{2}, Precision:{3}\n\n".format(dice, volumeSimilarity, sensitivity, precision))

                            csvWriterMethods.writerow([useMask, weigthDeformity, iterations, numSamples, finalGridSpacing_mm])
                            csvWriterDice.writerow(metricsByLabel['dice'])
                            csvWriterSensitivity.writerow(metricsByLabel['sensitivity'])
                            csvWriterPrecision.writerow(metricsByLabel['precision'])
                            fMethods.flush()
                            fDice.flush()
                            fSensitivity.flush()
                            fPrecision.flush()
                            log.flush()


log.close()
fMethods.close()
fDice.close()
fSensitivity.close()
fPrecision.close()
