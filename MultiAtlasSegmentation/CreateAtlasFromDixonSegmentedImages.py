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
referenceCase = 'ID00029'
gender = 'Men' #'Men', 'Women' or 'All'
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask_results\\Postproc\\'
targetMuscleLabelsPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask\\'
outputPath = "D:\\Martin\\MarathonStudy\\NormativeValues\\Shape\\" + referenceCase + '\\' + gender + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Filenames and extensions:
suffixSegmentedImages = '_tissue_segmented'
suffixFatFractionImages = '_fat_fraction'
nameMuscleSegmentedImages = 'segmentedImage'
suffixMuscleSegmentedImages = ''
extensionImages = '.mhd'
extensionImagesBin = '.raw'

# Look for the file of the reference image:
referencePath = targetPath + referenceCase + '\\'
referenceMuscleLabelsPath = targetMuscleLabelsPath + referenceCase + '\\'
referenceFilename = referencePath + referenceCase + suffixSegmentedImages + extensionImages
referenceMuscleLabelsFilename = referenceMuscleLabelsPath + nameMuscleSegmentedImages + suffixMuscleSegmentedImages + extensionImages
referenceFatFractionFilename = referencePath + referenceCase + suffixFatFractionImages + extensionImages
if not os.path.exists(referenceFilename):
    print('Reference image not found.')
    exit(-1)

# Look for the raw files in the TARGET PATH:
files = os.listdir(targetPath)
allTargetImagesNames = []
allTargetImagesFilenames = []
allTargetFatFractionImagesFilenames = []
allTargetMuscleImagesFilenames = []
for filename in files:
    if os.path.isdir(targetPath + filename):
        name, extension = os.path.splitext(filename)
        # Check that is not the reference image:
        if name != referenceCase:
            segmentedImageFilename = targetPath + filename + '\\' + name + suffixSegmentedImages + extensionImages
            fatFractionImageFilename = targetPath + filename + '\\' + name + suffixFatFractionImages + extensionImages
            muscleSegmentedImageFilename = targetMuscleLabelsPath + name + '\\' + nameMuscleSegmentedImages + suffixMuscleSegmentedImages + extensionImages
            # if its an image copy it back to the main folder:
            if os.path.exists(segmentedImageFilename) and os.path.exists(fatFractionImageFilename) and os.path.exists(muscleSegmentedImageFilename):
                # Intensity image:
                allTargetImagesNames.append(name)
                allTargetImagesFilenames.append(segmentedImageFilename)
                allTargetFatFractionImagesFilenames.append(fatFractionImageFilename)
                allTargetMuscleImagesFilenames.append(muscleSegmentedImageFilename)

# Leave only the data for a given gender:
# File with gender:
genderFilename = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\genderMarathon.csv'
caseNamesForGender = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\marathonCasesNamesForStats.csv'
# Read the csv file with the landmarks and store them:
atlasNamesInGenderFile = list()
genderAllCases = list()
# Read gender:
with open(genderFilename, newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        genderAllCases.append(int(row[0]))
# Read names:
with open(caseNamesForGender, newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        atlasNamesInGenderFile.append(row[0])

# Create a list with only the target cases that will be those where the gender matches the model gender:
targetImagesNames = []
targetImagesFilenames = []
targetFatFractionImagesFilenames = []
targetMuscleImagesFilenames = []
if gender == 'All':
    targetImagesFilenames = allTargetImagesFilenames
    targetImagesNames = allTargetImagesNames
else:
    for i in range(0, len(allTargetImagesNames)):
        ind = atlasNamesInGenderFile.index(allTargetImagesNames[i]) # This will throw an exception if the case is not available:
        # Now check the gender, if mathces to the model gender, added to the list:
        if ((gender == 'Men') and (genderAllCases[ind] == 1)) or ((gender == 'Women') and (genderAllCases[ind] == 0)):
            targetImagesNames.append(allTargetImagesNames[i])
            targetImagesFilenames.append(allTargetImagesFilenames[i])
            targetFatFractionImagesFilenames.append(allTargetFatFractionImagesFilenames[i])
            targetMuscleImagesFilenames.append(allTargetMuscleImagesFilenames[i])



#targetImagesNames = targetImagesNames[0:4] # Just 4 images for faster testing.
print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of cases: {0}\n".format(targetImagesNames))

############################### READ REFERENCE IMAGE ###########################
# To create the fixed Dixon image, I fuse the tissue segmented image with the labels segmetned images:
fixedTissuesImage = sitk.Cast(sitk.ReadImage(referenceFilename), sitk.sitkUInt8)
fixedLabelsImage = sitk.Cast(sitk.ReadImage(referenceMuscleLabelsFilename), sitk.sitkUInt8)
fixedTissuesImage.CopyInformation(fixedLabelsImage) # The tissue segmented image lost that info.
fixedImage = MergeTissueAndLabelsImages(fixedTissuesImage, fixedLabelsImage)
sitk.WriteImage(fixedImage, outputPath + referenceCase + '_AllLabels' + extensionImages, True)
# Read fat fraction images:
fixedFatFraction = sitk.Cast(sitk.ReadImage(referenceFatFractionFilename), sitk.sitkFloat32)
fixedFatFraction.CopyInformation(fixedLabelsImage) # The tissue segmented image lost that info.



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

paramFileNonRigid = 'Parameters_BSpline_NCC_2000iters_2048samples'
#paramFileNonRigid = paramFileAffine
#paramFileNonRigid = 'WithDeformationPenalty\\Parameters_BSpline_NCC_1000iters_2048samples'

outputPath = outputPath + paramFileNonRigid + '\\'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
################################## REGISTRATION WITH DEFORMATION #######################################
# Evaluate parameters of registration methods with deformation penalty:
iterations = [2000]
numSamples = [3000]
finalGridSpacing_mm = [15]#(FinalGridSpacingInPhysicalUnits 5.0 5.0 5.0)

# elastixImageFilter filter
elastixImageFilter = sitk.ElastixImageFilter()
parametersBspline = elastixImageFilter.ReadParameterFile(parameterFilesPath + paramFileNonRigid + '.txt')

# Change number of iterations:
#parametersBspline['MaximumNumberOfIterations'] = [str(iterations)]
#parametersBspline['NumberOfSpatialSamples'] = [str(numSamples)]
# Change the final grid spacing:
#parametersBspline['FinalGridSpacingInPhysicalUnits'] = [str(finalGridSpacing_mm)]

metricValues = []
executionTimes = []
registeredImages = []
fatFractionImages = []
# Start with the fixed images:
fatFractionImages.append(fixedFatFraction)
registeredImages.append(fixedImage)
# Now go through all the images:
for i in range(len(targetImagesNames)):
    caseMoving = targetImagesNames[i]
    # Create moving image:
    movingTissuesImage = sitk.Cast(sitk.ReadImage(targetImagesFilenames[i]), sitk.sitkUInt8)
    movingFatFractionImage = sitk.Cast(sitk.ReadImage(targetFatFractionImagesFilenames[i]), sitk.sitkUInt8)
    movingLabelsImage = sitk.Cast(sitk.ReadImage(targetMuscleImagesFilenames[i]), sitk.sitkUInt8)
    movingTissuesImage.CopyInformation(movingLabelsImage)  # The tissue segmented image lost that info.
    movingFatFractionImage.CopyInformation(movingLabelsImage)  # The fat fraction segmented image lost that info.
    movingImage = MergeTissueAndLabelsImages(movingTissuesImage, movingLabelsImage)
    sitk.WriteImage(movingImage, outputPath + caseMoving + '_AllLabels' + extensionImages, True)

    # Time to compute execution time:
    startTime = time.time()
    # The registration
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    parameterMapVector.append(parametersBspline)

    # Registration:
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get the images:
    outputTransformParameterMap = elastixImageFilter.GetTransformParameterMap()
    # Apply transform for a nearest neighbour output:
    # Apply its transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetMovingImage(movingImage)
    transformixImageFilter.SetTransformParameterMap(outputTransformParameterMap)
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    outputImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
    registeredImages.append(outputImage)
    # Transfer also the fat fraction images:
    transformixImageFilter.SetMovingImage(movingFatFractionImage)
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "1")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
    transformixImageFilter.Execute()
    outputFatFractionImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkFloat32)
    fatFractionImages.append(outputFatFractionImage)

    # Get metric:
    imRegMethod = sitk.ImageRegistrationMethod()
    imRegMethod.SetMetricAsCorrelation()
    metricValue = imRegMethod.MetricEvaluate(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(outputImage, sitk.sitkFloat32))
    print(metricValue)
    metricValues.append(metricValue)
    # Stop time
    stopTime = time.time()
    # Execution times:
    executionTimes.append(stopTime-startTime)

    # Write image:
    outputFilename = outputPath + '\\' + referenceCase + '_' + caseMoving + '.mhd'
    sitk.WriteImage(outputImage, outputFilename, True)

print('Similarity values: {0}'.format(metricValues))

print('Execution times: {0}'.format(executionTimes))

# Do majority voting:
numLabels = 14
finalSegmentedImage = sitk.LabelVoting(registeredImages, numLabels) # Majority Voting only with the selected atlases.
sitk.WriteImage(finalSegmentedImage,  outputPath + referenceCase + '_ModelMajorityVoting.mhd', True)

# Generate fat fraction atlas:
labelsToProcess = range(1, 10)
meanStdFatFractionImage = GetMeanAndStdValuesForEachLabelFromSetOfImages(finalSegmentedImage, fatFractionImages, labelsToProcess, True)
#meanStdFatFractionImage = GetMeanAndStdValuesForEachLabelFromSetOfImagesWithOwnLabels(finalSegmentedImage, registeredImages, fatFractionImages, labelsToProcess, True)
sitk.WriteImage(meanStdFatFractionImage['std'],  outputPath + referenceCase + '_MeanFFModelMajorityVoting.mhd', True)
sitk.WriteImage(meanStdFatFractionImage['mean'],  outputPath + referenceCase + '_StdFFMajorityVoting.mhd', True)