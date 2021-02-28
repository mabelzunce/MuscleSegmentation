#! python3


from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os
import time
import csv
sys.path.append('..\LabelPropagation')
import DixonTissueSegmentation as DixonTissueSeg
import SitkImageManipulation as sitkIm
import SegmentationPerformanceMetrics as segmentationMetrics
import winshell
import MajorityVoting as MV
from PostprocessingLabels import MergeTissueAndLabelsImages

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
        # Divide the sum to get the mean value:
        outputThisLabel = sitk.Divide(outputThisLabel, len(fatFractionRegisteredImages)) # Mean in this region
        # Get the standard deviation:
        outputThisLabelSquared = sitk.Sqrt(sitk.Subtract(sitk.Divide(outputThisLabelSquared, len(fatFractionRegisteredImages)), sitk.Multiply(outputThisLabel, outputThisLabel)))

        # Now add this to the final Mean and Std image:
        meanImage = sitk.Add(meanImage, outputThisLabel)
        stdImage = sitk.Add(stdImage, outputThisLabelSquared)
    # Return a dictionary:
    return {'mean': meanImage, 'std': stdImage}

########################## FUNCTION THAT ESTIMATES MEDIAN, IQR and 10-90% FAT FRACTION IMAGES FOR THE ATLAS ###################
# atlas: the atlas image with a label for each muscle
# fatFractionRegisteredImages:
# labels: array with the label indices to process, if =0 all labels in the image are processed.
# erodeMasks: erodes each mask before applying it to avoide edge problems.
# For the ranges, returns an imafe gor lower and upper limits
def GetMedianAndIqrValuesForEachLabelFromSetOfImages(atlas, fatFractionRegisteredImages, labels = 0, erodeMasks = True):
    medianImage = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    medianImage.CopyInformation(atlas)
    perc25Image = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    perc25Image.CopyInformation(atlas)
    perc75Image = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    perc75Image.CopyInformation(atlas)
    perc10Image = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    perc10Image.CopyInformation(atlas)
    perc90Image = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
    perc90Image.CopyInformation(atlas)
    # Array with size of the image:
    sizeImage = atlas.GetSize()
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
        # Create a numpy array for all the image:
        ndaAllImages = np.zeros((sizeImage[2], sizeImage[1], sizeImage[0], len(fatFractionRegisteredImages)), dtype = np.float)
        # Go through the registered images:
        for j in range(0,len(fatFractionRegisteredImages)):
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(1)
            maskFilter.SetOutsideValue(0)
            # Out of the mask, keeps the values, in the mask sets the output value (0), that's why we use the ivnerted
            # mask.
            maskedImage = maskFilter.Execute(fatFractionRegisteredImages[j], maskThisLabel)
            # Apply smoothing to the input intensity image (fat fraction) in this case:
            maskedImage = sitk.SmoothingRecursiveGaussian(maskedImage, (1.0,1.0,1.0))
            # Get a numpy array:
            ndaAllImages[:,:,:,j] = sitk.GetArrayFromImage(maskedImage)
        medianImageThisLabel = sitk.Cast(sitk.GetImageFromArray(np.median(ndaAllImages, axis = 3)), sitk.sitkFloat32)
        medianImageThisLabel.CopyInformation(atlas)
        medianImage = sitk.Add(medianImage, medianImageThisLabel)
        auxImage = sitk.Cast(sitk.GetImageFromArray(np.percentile(ndaAllImages, 25, axis = 3)), sitk.sitkFloat32)
        auxImage.CopyInformation(atlas)
        perc25Image = sitk.Add(perc25Image, auxImage)
        auxImage = sitk.Cast(sitk.GetImageFromArray(np.percentile(ndaAllImages, 75, axis=3)), sitk.sitkFloat32)
        auxImage.CopyInformation(atlas)
        perc75Image = sitk.Add(perc75Image, auxImage)
        auxImage = sitk.Cast(sitk.GetImageFromArray(np.percentile(ndaAllImages, 10, axis=3)), sitk.sitkFloat32)
        auxImage.CopyInformation(atlas)
        perc10Image = sitk.Add(perc10Image, auxImage)
        auxImage = sitk.Cast(sitk.GetImageFromArray(np.percentile(ndaAllImages, 90, axis=3)), sitk.sitkFloat32)
        auxImage.CopyInformation(atlas)
        perc90Image = sitk.Add(perc90Image, auxImage)
        # Return a dictionary:
    return {'median': medianImage, 'p25': perc25Image, 'p75': perc75Image, 'p10': perc10Image, 'p90': perc90Image}


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
debug = 1
prePregistered = 1 # If images have been already registered and saved use this flag and they will be loaded instead of computed:
referenceCase = 'ID00048'
gender = 'Men' #'Men', 'Women' or 'All'
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask_results\\Postproc\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\AllWithLinks\\'
targetAutomatedSegmentationPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask\\'
outputPath = "D:\\Martin\\MarathonStudy\\NormativeValues\\Shape\\" + referenceCase + '\\' + gender + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)


# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
extensionImagesBin = '.raw'
inPhaseSuffix = '_I'#
outOfPhaseSuffix = '_O'#
waterSuffix = '_W'#
fatSuffix = '_F'#
suffixFatFractionImages = '_fat_fraction'
suffixSegmentedImages = '_tissue_segmented'
suffixManualMuscleLabels = '_labels'
nameMuscleSegmentedImages = 'segmentedImage'
suffixMuscleSegmentedImages = ''
dixonSuffixInOrder = (inPhaseSuffix, outOfPhaseSuffix, waterSuffix, fatSuffix)
# Look for the folders or shortcuts:
files = os.listdir(targetPath)
allDixonImagesFilenames = []
allTargetImagesNames = []
allTargetImagesFilenames = []
allTargetFatFractionImagesFilenames = []
allTargetMuscleImagesFilenames = []
for filename in files:
    name, extension = os.path.splitext(filename)
    # if name is a lnk, get the path:
    if str(extension).endswith(extensionShortcuts):
        # This is a shortcut:
        shortcut = winshell.shortcut(targetPath + filename)
        indexStart = shortcut.as_string().find(strForShortcut)
        dataPath = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
    else:
        dataPath = targetPath + filename + '\\'
    # Check if the images are available:
    filename = dataPath + 'ForLibrary\\' + name + inPhaseSuffix + extensionImages
    if os.path.exists(filename):
        allTargetImagesNames.append(name)
        # Add in-phase, if all images needed, uncomment:
        # Add dixon images in order:
        #for suffix in dixonSuffixInOrder:
        #    filename = dataPath + 'ForLibrary\\' + name + suffix + extensionImages
        allDixonImagesFilenames.append(filename)
        # Now add tissue segmented images and fat fraction:
        filename = dataPath + 'ForLibrary\\' + name + suffixSegmentedImages + extensionImages
        filenameFatFraction = dataPath + 'ForLibrary\\' + name + suffixFatFractionImages + extensionImages
        if os.path.exists(filename) and os.path.exists(filenameFatFraction):
            allTargetImagesFilenames.append(filename)
            allTargetFatFractionImagesFilenames.append(filenameFatFraction)
        else:
            print('Dixon tissue segmented image or fat fraction not found for:' + name)
            exit(-1)
        # Check for the segmented image, that can be manual or automated
        filename = dataPath + 'ForLibrary\\' + name + suffixManualMuscleLabels + extensionImages
        if os.path.exists(filename):
            allTargetMuscleImagesFilenames.append(filename)
        else:
            # Try with the automated labels:
            filename = targetAutomatedSegmentationPath + name + '\\' + nameMuscleSegmentedImages + suffixMuscleSegmentedImages + extensionImages
            if os.path.exists(filename):
                allTargetMuscleImagesFilenames.append(filename)
            else:
                print('Muscle labels not found for:' + name)
                exit(-1)

# Previous version with only automated:
# # Look for the raw files in the TARGET PATH:
# files = os.listdir(targetPath)
# allTargetImagesNames = []
# allTargetImagesFilenames = []
# allTargetFatFractionImagesFilenames = []
# allTargetMuscleImagesFilenames = []
# for filename in files:
#     if os.path.isdir(targetPath + filename):
#         name, extension = os.path.splitext(filename)
#         # Check that is not the reference image:
#         if name != referenceCase:
#             segmentedImageFilename = targetPath + filename + '\\' + name + suffixSegmentedImages + extensionImages
#             fatFractionImageFilename = targetPath + filename + '\\' + name + suffixFatFractionImages + extensionImages
#             muscleSegmentedImageFilename = targetMuscleLabelsPath + name + '\\' + nameMuscleSegmentedImages + suffixMuscleSegmentedImages + extensionImages
#             # if its an image copy it back to the main folder:
#             if os.path.exists(segmentedImageFilename) and os.path.exists(fatFractionImageFilename) and os.path.exists(muscleSegmentedImageFilename):
#                 # Intensity image:
#                 allTargetImagesNames.append(name)
#                 allTargetImagesFilenames.append(segmentedImageFilename)
#                 allTargetFatFractionImagesFilenames.append(fatFractionImageFilename)
#                 allTargetMuscleImagesFilenames.append(muscleSegmentedImageFilename)
                
# Look for the file of the reference image:
indexReference = allTargetImagesNames.index(referenceCase)
referenceFilename = allTargetImagesFilenames[indexReference]
referenceMuscleLabelsFilename = allTargetMuscleImagesFilenames[indexReference]
referenceFatFractionFilename = allTargetFatFractionImagesFilenames[indexReference]
if not os.path.exists(referenceFilename):
    print('Reference image not found.')
    exit(-1)

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

paramFileNonRigid = 'Parameters_BSpline_NMI_2000iters_2048samples'
#paramFileNonRigid = paramFileAffine
#paramFileNonRigid = 'WithDeformationPenalty\\Parameters_BSpline_NCC_1000iters_2048samples'

outputPath = outputPath + paramFileNonRigid + '\\'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
################################## REGISTRATION WITH DEFORMATION #######################################
# Evaluate parameters of registration methods with deformation penalty:
iterations = 2000
numSamples = 3000
finalGridSpacing_mm = 10#(FinalGridSpacingInPhysicalUnits 5.0 5.0 5.0)

# elastixImageFilter filter
elastixImageFilter = sitk.ElastixImageFilter()
parametersBspline = elastixImageFilter.ReadParameterFile(parameterFilesPath + paramFileNonRigid + '.txt')

# Change number of iterations:
parametersBspline['MaximumNumberOfIterations'] = [str(iterations)]
parametersBspline['NumberOfSpatialSamples'] = [str(numSamples)]
# Change the final grid spacing:
parametersBspline['FinalGridSpacingInPhysicalUnits'] = [str(finalGridSpacing_mm)]

metricValues = []
executionTimes = []
registeredImages = []
fatFractionImages = []
# Start with the fixed images:
fatFractionImages.append(fixedFatFraction)
registeredImages.append(fixedImage)
# Now go through all the images:
if not prePregistered:
    for i in range(len(targetImagesNames)):
        caseMoving = targetImagesNames[i]
        # Create moving image:
        movingTissuesImage = sitk.Cast(sitk.ReadImage(targetImagesFilenames[i]), sitk.sitkUInt8)
        movingFatFractionImage = sitk.Cast(sitk.ReadImage(targetFatFractionImagesFilenames[i]), sitk.sitkFloat32)
        movingLabelsImage = sitk.Cast(sitk.ReadImage(targetMuscleImagesFilenames[i]), sitk.sitkUInt8)
        movingTissuesImage.CopyInformation(movingLabelsImage)  # The tissue segmented image lost that info.
        movingFatFractionImage.CopyInformation(movingLabelsImage)  # The fat fraction segmented image lost that info.
        movingImage = MergeTissueAndLabelsImages(movingTissuesImage, movingLabelsImage)
        if debug:
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
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.LogToConsoleOn()
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
        #transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()
        #transformixImageFilter.ComputeDeformationFieldOn()
        #transformixImageFilter.ComputeSpatialJacobianOn()
        #outputPathJacobian = outputFilename = outputPath + '\\Jacobians\\' + caseMoving + '\\'
        #if not os.path.exists(outputPathJacobian):
        #    os.makedirs(outputPathJacobian)
        #transformixImageFilter.SetOutputDirectory(outputPathJacobian)
        transformixImageFilter.Execute()
        outputImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # Write the deformation field:
        #deformationField = transformixImageFilter.GetDeformationField()
        #sitk.WriteImage(deformationField, outputPathJacobian + 'deformationField.mhd', True) # It doesn't work
        registeredImages.append(outputImage)

        # Transfer also the fat fraction images:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.SetTransformParameterMap(outputTransformParameterMap)
        transformixImageFilter.SetMovingImage(movingFatFractionImage)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "1")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
        transformixImageFilter.Execute()
        outputFatFractionImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkFloat32)
        fatFractionImages.append(outputFatFractionImage)
        if debug:
            sitk.WriteImage(outputFatFractionImage, outputPath + caseMoving + '_fat_fraction' + extensionImages, True)
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
else:
    # If doing only post rpocessing read the iamges:
    finalSegmentedImage = sitk.ReadImage(outputPath + referenceCase + '_ModelMajorityVoting.mhd')
    for i in range(len(targetImagesNames)):
        caseMoving = targetImagesNames[i]
        registeredImages.append(sitk.ReadImage(outputPath + '\\' + referenceCase + '_' + caseMoving + '.mhd'))
        fatFractionImages.append(sitk.ReadImage(outputPath + caseMoving + '_fat_fraction' + extensionImages))

# Do majority voting:
numLabels = 14
finalSegmentedImage = sitk.LabelVoting(registeredImages, numLabels) # Majority Voting only with the selected atlases.
sitk.WriteImage(finalSegmentedImage,  outputPath + referenceCase + '_ModelMajorityVoting.mhd', True)

labelsToProcess = range(1, 11)
medianIqrFatFractionImage = GetMedianAndIqrValuesForEachLabelFromSetOfImages(finalSegmentedImage, fatFractionImages, labelsToProcess, erodeMasks = True)
sitk.WriteImage(medianIqrFatFractionImage['median'], outputPath + referenceCase + '_MedianFFModelMajorityVoting.mhd',
                False)
sitk.WriteImage(medianIqrFatFractionImage['p25'], outputPath + referenceCase + '_p25FFMajorityVoting.mhd', False)
sitk.WriteImage(medianIqrFatFractionImage['p75'], outputPath + referenceCase + '_p75FFMajorityVoting.mhd', False)
sitk.WriteImage(medianIqrFatFractionImage['p10'], outputPath + referenceCase + '_p10FFMajorityVoting.mhd', False)
sitk.WriteImage(medianIqrFatFractionImage['p90'], outputPath + referenceCase + '_p90FFMajorityVoting.mhd', False)
print('Similarity values: {0}'.format(metricValues))
print('Execution times: {0}'.format(executionTimes))

# Generate fat fraction atlas:
labelsToProcess = range(1, 11)
meanStdFatFractionImage = GetMeanAndStdValuesForEachLabelFromSetOfImages(finalSegmentedImage, fatFractionImages, labelsToProcess, True)
#meanStdFatFractionImage = GetMeanAndStdValuesForEachLabelFromSetOfImagesWithOwnLabels(finalSegmentedImage, registeredImages, fatFractionImages, labelsToProcess, True)
sitk.WriteImage(meanStdFatFractionImage['mean'],  outputPath + referenceCase + '_MeanFFModelMajorityVoting.mhd', False)
sitk.WriteImage(meanStdFatFractionImage['std'],  outputPath + referenceCase + '_StdFFMajorityVoting.mhd', False)

# Generate probaility maps:
#probMaps = MV.GetProbailityMapsFromSegmentedImages(registeredImages, numLabels)
#for i in range(0, len(probMaps)):
#    sitk.WriteImage(probMaps[i], outputPath + "ProbMap_label_{0}.mhd".format(i + 1), True)