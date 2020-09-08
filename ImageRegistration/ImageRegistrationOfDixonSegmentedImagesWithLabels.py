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

############################### REFERENCE CASE AND FOLDER WITH ALL CASES ###################################
referenceCase = 'ID00002'
gender = 'Women' #'Men', 'Women' or 'All'
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask_results\\Postproc\\'
outputPath = "D:\\Martin\\MarathonStudy\\NormativeValues\\Shape\\" + referenceCase + '\\' + gender + '\\'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Filenames and extensions:
suffixSegmentedImages = '_tissue_segmented'
extensionImages = '.mhd'
extensionImagesBin = '.raw'

# Look for the file of the reference image:
referencePath = targetPath + referenceCase + '\\'
referenceFilename = referencePath + referenceCase + suffixSegmentedImages + extensionImages
if not os.path.exists(referenceFilename):
    print('Reference image not found.')
    exit(-1)

# Look for the raw files in the TARGET PATH:
files = os.listdir(targetPath)
allTargetImagesNames = []
allTargetImagesFilenames = []
for filename in files:
    if os.path.isdir(targetPath + filename):
        name, extension = os.path.splitext(filename)
        # Check that is not the reference image:
        if name != referenceCase:
            segmentedImageFilename = targetPath + filename + '\\' + name + suffixSegmentedImages + extensionImages
            # if its an image copy it back to the main folder:
            if os.path.exists(segmentedImageFilename):
                # Intensity image:
                allTargetImagesNames.append(name)
                allTargetImagesFilenames.append(segmentedImageFilename)

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



#targetImagesNames = targetImagesNames[0:4] # Just 4 images for faster testing.
print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of cases: {0}\n".format(targetImagesNames))

############################### READ REFERENCE IMAGE ###########################
fixedImage = sitk.Cast(sitk.ReadImage(referenceFilename), sitk.sitkUInt8)

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
paramFileNonRigid = paramFileAffine
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
for i in range(len(targetImagesNames)):
    filenameMoving = targetImagesFilenames[i]
    caseMoving = targetImagesNames[i]
    movingImage = sitk.ReadImage(filenameMoving)
    nameMoving, extension = os.path.splitext(filenameMoving)

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
numLabels = 4
finalSegmentedImage = sitk.LabelVoting(registeredImages, numLabels) # Majority Voting only with the selected atlases.

sitk.WriteImage(finalSegmentedImage,  outputPath + referenceCase + '_ModelMajorityVoting.mhd', True)