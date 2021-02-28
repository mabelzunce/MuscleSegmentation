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
import winshell

########################## FUNCTION THAT MERGES TISSUE SEGMENTED WITH LABELS SEGMENTED IMAGES ###################
def MergeTissueAndLabelsImages(dixonTissuesImage, muscleLabelsImage):
    numTissues = 3
    numLabels = 11
    # The tissue segmented images have as labels softTissue = 1, Mixed = 2, Fat = 3.
    # The muscle labels have labels from 1 to 11, being 9 and 10 femur labels and 11 undecided.
    # We will create an image with labels from 1 to 8 and then softTissueOutOfLabels=12 Mixed=13, Fat=14
    outputImage = sitk.Image(muscleLabelsImage.GetSize(), sitk.sitkUInt8)
    outputImage.CopyInformation(muscleLabelsImage)
    # First the tissue images:
    for i in range(1,numTissues+1):
        maskFilter = sitk.MaskImageFilter()
        maskFilter.SetMaskingValue(i)
        maskFilter.SetOutsideValue(numLabels + i)
        outputImage = maskFilter.Execute(outputImage, dixonTissuesImage)
    # Now the muscle labels # TODO: apply soft tissue mask to the labels.
    for i in range(1,numLabels+1):
        maskFilter = sitk.MaskImageFilter()
        maskFilter.SetMaskingValue(i)
        maskFilter.SetOutsideValue(i)
        outputImage = maskFilter.Execute(outputImage, muscleLabelsImage)

    return outputImage

########################## FUNCTION THAT ESTIMATES MEAN AND STD DEVIATION FAT FRACTION IMAGES FOR THE ATLAS ###################
# atlas: the atlas image with a label for each muscle
# fatFractionRegisteredImages:
# labels: array with the label indices to process, if =0 all labels in the image are processed.
# erodeMasks: erodes each mask before applying it to avoide edge problems.
def GetMeanAndStdValuesForEachLabelFromSetOfImages(atlas, fatFractionRegisteredImages, labels = 0, erodeMasks = True):
    meanImage = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    meanImage.CopyInformation(atlas)
    stdImage = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    stdImage.CopyInformation(atlas)
    # Get the number of labels:
    if labels == 0:
        labelStatsFilter = sitk.LabelStatisticsImageFilter()
        labelStatsFilter.Execute(atlas, atlas)
        numLabels = labelStatsFilter.GetNumberOfLabels()
        labels = range(1,numLabels)

    # Go through each label:
    for i in labels:
        # Mask this label (negative):
        maskThisLabel = atlas != i
        # Because I am using the mask negated, I need to dilate the mask instead.
        if erodeMasks:
            filter = sitk.BinaryDilateImageFilter()
            filter.SetKernelRadius((1,1,0))
            maskThisLabel = filter.Execute(maskThisLabel)

        # Output for this label:
        outputThisLabel = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
        outputThisLabel.CopyInformation(atlas)
        # To compute teh standard deviation, I use the sum of the squares method:
        outputThisLabelSquared = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
        outputThisLabelSquared.CopyInformation(atlas)
        # Go through the registered images:
        for j in range(0,len(fatFractionRegisteredImages)):
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(1)
            maskFilter.SetOutsideValue(0)
            # Out of the mask, keeps the values, in the mask sets the output value (0), that's why we use the ivnerted
            # mask.
            maskedImage = maskFilter.Execute(fatFractionRegisteredImages[j], maskThisLabel)
            # Apply smoothing to the input intensity image (fat fraction) in this case:
            #maskedImage = sitk.SmoothingRecursiveGaussian(maskedImage, (1.0,1.0,0.5))
            outputThisLabel = sitk.Add(outputThisLabel, maskedImage)
            outputThisLabelSquared  = sitk.Add(outputThisLabelSquared, sitk.Multiply(maskedImage, maskedImage))

        # Get the standard deviation:
        outputThisLabelSquared = sitk.Sqrt(sitk.Subtract(outputThisLabelSquared, sitk.Divide(sitk.Multiply(outputThisLabel, outputThisLabel), len(fatFractionRegisteredImages))))
        # Divide the sum to get the mean value:
        outputThisLabel = sitk.Divide(outputThisLabel, len(fatFractionRegisteredImages))

        # Now add this to the final Mean and Std image:
        meanImage = sitk.Add(meanImage, outputThisLabel)
        stdImage = sitk.Add(stdImage, outputThisLabelSquared)
    # Return a dictionary:
    return {'mean': meanImage, 'std': stdImage}

########################## FUNCTION THAT ESTIMATES MEAN AND STD DEVIATION FAT FRACTION IMAGES FOR THE ATLAS CONSTRAINED BY THE REGISTERED LABELS ###################
# atlas: the atlas image with a label for each muscle
# labelsRegisteredImages: list with labels images registered to the atlas.
# fatFractionRegisteredImages: list with fat fraction images registered to the atlas.
# labels: array with the label indices to process, if =0 all labels in the image are processed.
# erodeMasks: erodes each mask before applying it to avoide edge problems.
def GetMeanAndStdValuesForEachLabelFromSetOfImagesWithOwnLabels(atlas, labelsRegisteredImages, fatFractionRegisteredImages, labels = 0, erodeMasks = True):
    meanImage = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    meanImage.CopyInformation(atlas)
    stdImage = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    stdImage.CopyInformation(atlas)
    # Get the number of labels:
    if labels == 0:
        labelStatsFilter = sitk.LabelStatisticsImageFilter()
        labelStatsFilter.Execute(atlas, atlas)
        numLabels = labelStatsFilter.GetNumberOfLabels()
        labels = range(1,numLabels)

    # Go through each label:
    for i in labels:
        # Mask this label (negative):
        maskThisLabel = atlas != i
        if erodeMasks:
            # Because I am using the mask negated, I need to dilate the mask instead.
            filter = sitk.BinaryDilateImageFilter()
            filter.SetKernelRadius((1,1,0))
            maskThisLabel = filter.Execute(maskThisLabel)

        # Output for this label:
        outputThisLabel = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
        outputThisLabel.CopyInformation(atlas)
        # To compute teh standard deviation, I use the sum of the squares method:
        outputThisLabelSquared = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
        outputThisLabelSquared.CopyInformation(atlas)
        # Normalization image (with the sum of the number of registeredimages labels for each atlas label:
        normalizationImage = sitk.Image(atlas.GetSize(), sitk.sitkUInt8)
        normalizationImage.CopyInformation(atlas)
        # Go through the registered images:
        for j in range(0,len(fatFractionRegisteredImages)):
            # Restrict the atlas label to the labels in registered image:
            maskThisLabelConstrained = sitk.Or(maskThisLabel, labelsRegisteredImages[j] != i)
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(1)
            maskFilter.SetOutsideValue(0)
            # Out of the mask, keeps the values, in the mask sets the output value (0), that's why we use the ivnerted
            # mask.
            maskedImage = maskFilter.Execute(fatFractionRegisteredImages[j], maskThisLabelConstrained)
            # Apply smoothing to the input intensity image (fat fraction) in this case:
            maskedImage = sitk.SmoothingRecursiveGaussian(maskedImage, (1.0,1.0,0.5))
            outputThisLabel = sitk.Add(outputThisLabel, maskedImage)
            outputThisLabelSquared  = sitk.Add(outputThisLabelSquared, sitk.Multiply(maskedImage, maskedImage))
            # I need the sum for each label before normalizing as each voxel will have a different n:
            normalizationImage = sitk.Add(normalizationImage, maskThisLabelConstrained)

        # Get the standard deviation:
        outputThisLabelSquared = sitk.Sqrt(sitk.Subtract(outputThisLabelSquared, sitk.Divide(sitk.Multiply(outputThisLabel, outputThisLabel), sitk.Cast(normalizationImage, sitk.sitkFloat32))))
        # Divide the sum to get the mean value:
        outputThisLabel = sitk.Divide(outputThisLabel, sitk.Cast(normalizationImage, sitk.sitkFloat32))

        # Now add this to the final Mean and Std image:
        meanImage = sitk.Add(meanImage, outputThisLabel)
        stdImage = sitk.Add(stdImage, outputThisLabelSquared)
    # Return a dictionary:
    return {'mean': meanImage, 'std': stdImage}
############################### REFERENCE CASE AND FOLDER WITH ALL CASES ###################################
referenceCase = 'ID00006'
numLabels = 8
gender = 'Men' #'Men', 'Women' or 'All'
libraryVersion = 'V1.2'
libraryFolder = '\\NativeResolutionAndSize\\' #''\\NativeResolutionAndSize\\'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\Library' + libraryVersion + '\\'
#targetMuscleLabelsPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\Segmented\\'
# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\RegistrationParameters\\TestDixonTissueSegmented\\"
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Filenames and extensions:
suffixInPhaseImages = '_I'
suffixSegmentedImages = '_tissue_segmented'
suffixFatFractionImages = '_fat_fraction'
suffixMuscleSegmentedImages = '_labels'
extensionImages = '.mhd'
extensionImagesBin = '.raw'

# Look for the raw files in the TARGET PATH:
files = os.listdir(targetPath)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
targetImagesNames = []
targetImagesFilenames = []
targetInPhaseImagesFilenames = []
targetMuscleImagesFilenames = []
for filename in files:
    name, extension = os.path.splitext(filename)
    # if name is a lnk, get the path:
    if str(extension).endswith(extensionShortcuts):
        # This is a shortcut:
        shortcut = winshell.shortcut(targetPath + filename)
        indexStart = shortcut.as_string().find(strForShortcut)
        dataPath = shortcut.as_string()[indexStart + len(strForShortcut):] + '\\'
    else:
        dataPath = targetPath + filename + '\\'
    # Check if the images are available:
    filename = dataPath + 'ForLibrary\\' + name + suffixSegmentedImages + extensionImages
    if os.path.exists(filename):
        # Check that is not the reference image:
        if name != referenceCase:
            segmentedImageFilename = filename
            inPhaseImageFilename = dataPath + 'ForLibrary\\' + name + suffixInPhaseImages + extensionImages
            muscleSegmentedImageFilename =  dataPath + 'ForLibrary\\' + name + suffixMuscleSegmentedImages + extensionImages
            # if its an image copy it back to the main folder:
            if os.path.exists(segmentedImageFilename) and os.path.exists(inPhaseImageFilename) and os.path.exists(muscleSegmentedImageFilename):
                # Intensity image:
                targetImagesNames.append(name)
                targetImagesFilenames.append(segmentedImageFilename)
                targetInPhaseImagesFilenames.append(inPhaseImageFilename)
                targetMuscleImagesFilenames.append(muscleSegmentedImageFilename)
        else:
            # Look for the file of the reference image:
            referencePath = dataPath + 'ForLibrary\\'
            referenceMuscleLabelsPath = dataPath + 'ForLibrary\\'
            referenceFilename = filename
            referenceMuscleLabelsFilename = dataPath + 'ForLibrary\\' + name + suffixMuscleSegmentedImages + extensionImages
            referenceInPhaseFilename = dataPath + 'ForLibrary\\' + name + suffixInPhaseImages + extensionImages

# If not reference abort:
if not os.path.exists(referenceFilename):
    print('Reference image not found.')
    exit(-1)

#targetImagesNames = targetImagesNames[0:4] # Just 4 images for faster testing.
print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of cases: {0}\n".format(targetImagesNames))

############################### READ REFERENCE IMAGE ###########################
# To create the fixed Dixon image, I fuse the tissue segmented image with the labels segmetned images:
fixedTissuesImage = sitk.Cast(sitk.ReadImage(referenceFilename), sitk.sitkUInt8)
fixedLabelsImage = sitk.Cast(sitk.ReadImage(referenceMuscleLabelsFilename), sitk.sitkUInt8)
fixedInPhase = sitk.Cast(sitk.ReadImage(referenceInPhaseFilename), sitk.sitkFloat32)



################################ PARAMETERS ########################################
# Parameter files for the registration:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
subfolderNonRigidTest = ''#''WithDeformationPenalty\\'
paramFileRigid = 'Parameters_Rigid_NCC'
useAffine = False
paramFileAffine = 'Parameters_Affine_NCC'
paramFileNonRigid = 'Parameters_BSpline_NMI_4000iters_2048samples'

outputPath = outputPath + referenceCase + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
################################## REGISTRATION TUNING #######################################
# Evaluate parameters of registration methods with deformation penalty:
metric1WeightValues = ([20, 40, 60, 100], [20, 40, 60, 100])
metric1WeightValues = [0]
# The number of iterations for each of the registrations (with and without mask)
iterationsPerStage = [2000, 2000]
numSamples = 2000
finalGridSpacing_mm = 15#(FinalGridSpacingInPhysicalUnits 5.0 5.0 5.0)
maskValues = range(0,3)
useFixedMask = (1,2)
useMovingMask = (0,0)
erodeMask = False







############################################ MAIN LOOP FOR REGISTRATION #################################
########################## Create a numpy matrix for the metrics:
# Will compute the dice after each stage:
diceMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
volumeSimilarityMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
sensitivityMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))
precisionMatrix = np.zeros((len(iterationsPerStage), len(targetImagesNames)))


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

#del targetImagesNames[0:12]
# The fixed image will be the dixon segmented image:
fixedImage = fixedTissuesImage
for o in indicesToProcess: #range(0, len(targetImagesNames)):

    ######## REGISTRATION PARAMETERS ##########
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff() # Only set it to on if runing erros
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
    nameMoving = targetImagesNames[o]
    # Read target image:
    movingImage = sitk.ReadImage(targetImagesFilenames[o])
    # Include labels and in-phasealso:
    movingLabels = sitk.ReadImage(targetMuscleImagesFilenames[o])
    movingInPhase = sitk.ReadImage(targetInPhaseImagesFilenames[o])
    maskBodyMoving = movingImage > 0
    maskMovingSoftTissue = movingImage == 1

    outputPathThisCase = outputPathThisFile + targetImagesNames[o] + '\\'
    if not os.path.exists(outputPathThisCase):
        os.makedirs(outputPathThisCase)

    # first stage, no mask
    # second stage with mask:
    resultImageStage = []
    transformParameterMap = []
    propagatedLabels = []
    inPhaseImageStage = []
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
            maskFixed = fixedImage > 0
        elif useFixedMask[i] == 2:
            maskFixed = fixedImage == 1
        if useFixedMask[i] == 0:
            elastixImageFilter.RemoveFixedMask()
        else:
            elastixImageFilter.SetFixedMask(maskFixed)
            sitk.WriteImage(maskFixed, outputPathThisCase + referenceCase + '_mask_stage{0}.mhd'.format(i), True)
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
        # Repeat for the inphase iamge:
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "3")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
        transformixImageFilter.SetMovingImage(movingInPhase)
        transformixImageFilter.Execute()
        inPhaseImageStage.append(transformixImageFilter.GetResultImage())

        # Write images:
        outputFilename = outputPathThisCase + referenceCase + '_' + nameMoving + '_stage{0}.mhd'.format(i)
        sitk.WriteImage(resultImageStage[i], outputFilename, True)
        sitk.WriteParameterFile(transformParameterMap[i][0], outputPathThisCase + 'Transform')
        outputFilename = outputPathThisCase + referenceCase + '_' + nameMoving + '_labels_stage{0}.mhd'.format(i)
        sitk.WriteImage(propagatedLabels[i], outputFilename, True)
        outputFilename = outputPathThisCase + referenceCase + '_' + nameMoving + '_inphase_stage{0}.mhd'.format(i)
        sitk.WriteImage(inPhaseImageStage[i], outputFilename, True)

        # moving image for the next stage is the output of this one:
        movingImage = resultImageStage[i]
        movingLabels = propagatedLabels[i]
        movingInPhase = inPhaseImageStage[i]


     # Now get metrics:
    imRegMethod = sitk.ImageRegistrationMethod()
    metricNCCValue = []
    metricNMIValue = []
    for i in range(0, len(iterationsPerStage)):
        # Get metric values:
        imRegMethod.SetMetricAsCorrelation()
        metricNCCValue.append(imRegMethod.MetricEvaluate(fixedInPhase, inPhaseImageStage[i]))
        imRegMethod.SetMetricAsMattesMutualInformation()
        metricNMIValue.append(imRegMethod.MetricEvaluate(fixedInPhase, inPhaseImageStage[i]))
        log.write("Metrics. NCC: {0}. NMI: {1}\n".format(metricNCCValue[i], metricNMIValue[i]))
        # Get metrics:
        # first overall:
        metrics = segmentationMetrics.GetOverlapMetrics(fixedLabelsImage, propagatedLabels[i], 0)
        metricsByLabel = segmentationMetrics.GetOverlapMetrics(fixedLabelsImage, propagatedLabels[i], numLabels)
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