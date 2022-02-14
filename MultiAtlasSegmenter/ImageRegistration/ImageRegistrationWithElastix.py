#! python3
# Script to test the final image registration for the multi-atlas segmentation.
# It's based on the results obtained in ImageRegistrationWithElastixParameterSweep.py
# Secuencial image registration without mask and then with mask, using NMI as metric, 30 mm minimum griding (better than
# smaller grids.

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
nameFixed = 'ID00006'
nameMoving = 'ID00001'
############################### TARGET FOLDER ###################################
libraryVersion = 'V1.2'
libraryFolder = '\\NativeResolutionAndSize\\' #''\\NativeResolutionAndSize\\'
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
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels') and (name!=nameFixed):
        # Intensity image:
        targetImagesNames.append(name + '.' + extensionImages)
        # Label image:
        targetLabelsNames.append(name + '_labels.' + extensionImages)

targetImagesNames = targetImagesNames # Just 4 images for faster testing.
print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of files: {0}\n".format(targetImagesNames))

fixedImage =  sitk.ReadImage(targetPath + nameFixed + '.mhd')
# Include labels also:
fixedLabels =  sitk.ReadImage(targetPath + nameFixed + '_labels.mhd')
numLabels = 10
# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\TestTwostagesMethod\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

################################ PARAMETERS ########################################
# Parameter files for the registration:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
subfolderNonRigidTest = ''#''WithDeformationPenalty\\'
paramFileRigid = 'Parameters_Rigid_NCC'
useAffine = False
paramFileAffine = 'Parameters_Affine_NCC'
paramFileNonRigid = 'Parameters_BSpline_NMI_4000iters_2048samples_regions'
#paramFileNonRigid = 'WithDeformationPenalty\\Parameters_BSpline_NCC_1000iters_2048samples'

################################ MASKS ###########################################
# Create masks to be used in the registration:
maskBodyFixed = DixonTissueSeg.GetBodyMaskFromInPhaseDixon(fixedImage)
maskFixedSoftTissue = DixonTissueSeg.GetSoftTissueMaskFromInPhaseDixon(fixedImage)

outputPath = outputPath + nameFixed + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
sitk.WriteImage(maskBodyFixed, outputPath + 'maskBodyFixed.mhd', True)
sitk.WriteImage(maskFixedSoftTissue, outputPath + 'maskFixedSoftTissue.mhd', True)
sitk.WriteImage(fixedImage, outputPath + 'fixed.mhd', True)
sitk.WriteImage(fixedLabels, outputPath + 'fixedLabels.mhd', True)
################################## REGISTRATION TUNING #######################################
# Evaluate parameters of registration methods with deformation penalty:
metric1WeightValues = ([20, 40, 60, 100], [20, 40, 60, 100])
metric1WeightValues = [0]
# The number of iterations for each of the registrations (with and without mask)
iterationsPerStage = [2000, 4000]
numSamples = 2000
finalGridSpacing_mm = 15#(FinalGridSpacingInPhysicalUnits 5.0 5.0 5.0)
maskValues = range(0,3)
useFixedMask = (0,0)
useMovingMask = (0,0)
erodeMask = False
########################## Create a numpy matrix for the metrics:
# Will compute the dice after each stage:
diceMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
volumeSimilarityMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
sensitivityMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
precisionMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))

# Execute:
fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)

outputPathThisFile = outputPath + '{0}_masks_{1}{2}_erode{3}_iters0_{4}_iters1_{5}_samples_{6}'.format(paramFileNonRigid, useFixedMask, useMovingMask, erodeMask, iterationsPerStage[0], iterationsPerStage[1], numSamples) + '\\'
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
indicesToProcess = (10,13,2,0) # When a quick test is needed
targetImagesNamesToProcess= []
for i in indicesToProcess:
    targetImagesNamesToProcess.append(targetImagesNames[i])
#targetImagesNames=targetImagesNamesToProcess
#del targetImagesNames[0:12]
for o in range(0, len(targetImagesNames)):

    ######## REGISTRATION PARAMETERS ##########
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)

    parametersBspline = elastixImageFilter.ReadParameterFile(
        parameterFilesPath + subfolderNonRigidTest
        + paramFileNonRigid + '.txt')
    # Change weights for deformation:
    #parametersBspline['Metric1Weight'] = [str(weigthDeformity)
    parametersBspline['NumberOfSpatialSamples'] = [str(numSamples)]
    # Change the final grid spacing:
    parametersBspline['FinalGridSpacingInPhysicalUnits'] = [str(finalGridSpacing_mm)]
    # Erode mask:
    if erodeMask:
        parametersBspline['ErodeMask'] = ['true']
    else:
        parametersBspline['ErodeMask'] = ['false']



    ################# TARGET IMAGE #######################
    # Read target image:
    targetFilename = targetImagesNames[o]
    targetImageFilename = targetPath + targetFilename
    movingImage = sitk.ReadImage(targetImageFilename)
    path, filename = os.path.split(targetImageFilename)
    nameMoving, extension = os.path.splitext(filename)

    movingImage = sitk.ReadImage(targetPath + nameMoving + '.mhd')
    # Include labels also:
    movingLabels = sitk.ReadImage(targetPath + nameMoving + '_labels.mhd')
    maskBodyMoving = DixonTissueSeg.GetBodyMaskFromInPhaseDixon(movingImage)
    maskMovingSoftTissue = DixonTissueSeg.GetSoftTissueMaskFromInPhaseDixon(movingImage)

    outputPathThisCase = outputPathThisFile + nameMoving + '\\'
    if not os.path.exists(outputPathThisCase):
        os.makedirs(outputPathThisCase)

    # first stage, no mask
    # second stage with mask:
    resultImageStage = []
    transformParameterMap = []
    propagatedLabels = []
    for i in range(0,len(iterationsPerStage)):
        # Change number of iterations:
        parametersBspline['MaximumNumberOfIterations'] = [str(iterationsPerStage[i])]
        # Transforms:
        # Parameter maps:
        parameterMapVector = sitk.VectorOfParameterMap()
        if i == 0:
            parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                           + paramFileRigid + '.txt'))
            if useAffine:
                parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                               + paramFileAffine + '.txt'))
        parameterMapVector.append(parametersBspline)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        log.write('############### Regitrations stage 1 of {0} with {1} iteration and {2} samples ############\n'.format(nameMoving, iterationsPerStage[i], numSamples))
        elastixImageFilter.SetMovingImage(movingImage)

        # Set mask if needed:
        if useFixedMask[i] == 1:
            maskFixed = maskBodyFixed
        elif useFixedMask[i] == 2:
            maskFixed = maskFixedSoftTissue
        if useFixedMask[i] == 0:
            elastixImageFilter.RemoveFixedMask()
        else:
            elastixImageFilter.SetFixedMask(maskFixed)
            sitk.WriteImage(maskFixed, outputPathThisCase + nameFixed + '_mask_stage{0}.mhd'.format(i), True)
        if useMovingMask[i] == 1:
            maskMoving = maskBodyMoving
        elif useMovingMask[i] == 2:
            maskMoving = maskMovingSoftTissue
        if useMovingMask[i] == 0:
            elastixImageFilter.RemoveMovingMask()
        else:
            elastixImageFilter.SetFixedMask(maskMoving)
            sitk.WriteImage(maskMoving,
                            outputPathThisCase + nameMoving + '_mask_stage{0}.mhd'.format(i), True)
        # Write parameter map:
        for j in range(0, len(elastixImageFilter.GetParameterMap())):
            elastixImageFilter.WriteParameterFile(elastixImageFilter.GetParameterMap()[j],
                                                  outputPathThisCase + 'parameterFile{0}_stage{1}'.format(j,i))
        # Execute
        startTime = time.time()
        elastixImageFilter.Execute()
        endTime = time.time()
        print("Registration time for stage 1: {0} sec\n".format(endTime - startTime))
        log.write("Registration time for stage 1: {0} sec\n".format(endTime - startTime))
        # Get the images:
        resultImageStage.append(elastixImageFilter.GetResultImage())
        transformParameterMap.append(elastixImageFilter.GetTransformParameterMap())
        # Transfer the labels after stage 1:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.SetMovingImage(movingLabels)
        transformixImageFilter.SetTransformParameterMap(transformParameterMap[i])
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        propagatedLabels.append(sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8))
        # The same for all the moving masks (I have to keep all the masks updated, as in the next stages the mask used could be
        # other:
        transformixImageFilter.SetMovingImage(maskBodyMoving)
        transformixImageFilter.Execute()
        maskBodyMoving = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        transformixImageFilter.SetMovingImage(maskMovingSoftTissue)
        transformixImageFilter.Execute()
        maskMovingSoftTissue = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # Write images:
        outputFilename = outputPathThisCase + nameFixed + '_' + nameMoving + '_stage{0}.mhd'.format(i)
        sitk.WriteImage(resultImageStage[i], outputFilename, True)
        sitk.WriteParameterFile(transformParameterMap[i][0], outputPathThisCase + 'Transform')
        outputFilename = outputPathThisCase + nameFixed + '_' + nameMoving + '_labels_stage{0}.mhd'.format(i)
        sitk.WriteImage(propagatedLabels[i], outputFilename, True)

        # moving image for the next stage is the output of this one:
        movingImage = resultImageStage[i]
        movingLabels = propagatedLabels[i]


     # Now get metrics:
    imRegMethod = sitk.ImageRegistrationMethod()
    metricNCCValue = []
    metricNMIValue = []
    for i in range(0, len(iterationsPerStage)):
        # Get metric values:
        imRegMethod.SetMetricAsCorrelation()
        metricNCCValue.append(imRegMethod.MetricEvaluate(fixedImage, resultImageStage[i]))
        imRegMethod.SetMetricAsMattesMutualInformation()
        metricNMIValue.append(imRegMethod.MetricEvaluate(fixedImage, resultImageStage[i]))
        log.write("Metrics. NCC: {0}. NMI: {1}\n".format(metricNCCValue[i], metricNMIValue[i]))
        # Get metrics:
        # first overall:
        metrics = segmentationMetrics.GetOverlapMetrics(fixedLabels, propagatedLabels[i], 0)
        metricsByLabel = segmentationMetrics.GetOverlapMetrics(fixedLabels, propagatedLabels[i], numLabels)
        dice = metrics['dice']
        volumeSimilarity = metrics['volumeSimilarity']
        sensitivity = metrics['sensitivity']
        precision = metrics['precision']
        print("Overlap Similarity Metrics Stage {0}. Dice: {0}, Volume Similarity: {1}, Sensitivity:{2}, Precision:{3}\n\n".format(i, dice, volumeSimilarity, sensitivity, precision))
        log.write("Overlap Similarity Metrics Stage {0}. Dice: {0}, Volume Similarity: {1}, Sensitivity:{2}, Precision:{3}\n\n".format(i, dice, volumeSimilarity, sensitivity, precision))

        csvWriterMethods.writerow([i])
        csvWriterDice.writerow(metricsByLabel['dice'])
        csvWriterSensitivity.writerow(metricsByLabel['sensitivity'])
        csvWriterPrecision.writerow(metricsByLabel['precision'])
        fMethods.flush()
        fDice.flush()
        fSensitivity.flush()
        fPrecision.flush()
        log.flush()

        diceMatrix[i, o] = dice
        volumeSimilarityMatrix[i, o] = volumeSimilarity
        sensitivityMatrix[i, o] = sensitivity
        precisionMatrix[i, o] = precision

    # Write numpy matrix (keep updating the results):
    np.savetxt(outputPath + 'diceMatrix.csv', diceMatrix, delimiter=',')
    np.savetxt(outputPath + 'volumeSimilarityMatrix.csv', volumeSimilarityMatrix, delimiter=',')
    np.savetxt(outputPath + 'sensitivityMatrix.csv', sensitivityMatrix, delimiter=',')
    np.savetxt(outputPath + 'precisionMatrix.csv', precisionMatrix, delimiter=',')


log.close()
fMethods.close()
fDice.close()
fSensitivity.close()
fPrecision.close()
