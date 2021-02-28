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
############################### REFERENCE CASE AND FOLDER WITH ALL CASES ###################################
debug = 0
caseToProcess = 'ID00011'
atlasName = 'ID00003'
atlasPath = 'D:\\Martin\\MarathonStudy\\NormativeValues\\Shape\\' + atlasName + '\\Women\\Parameters_BSpline_NCC_2000iters_2048samples\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\AllWithLinks\\'
targetAutomatedSegmentationPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.1\\NonrigidNCC_1000_2048_N5_MaxProb_Mask\\'
outputPath = atlasPath + 'EvaluationOf' + caseToProcess + '\\' 
if not os.path.exists(outputPath):
    os.makedirs(outputPath)


# Constants:
extensionShortcuts = '.lnk'
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
numLabels = 14

# Read atlas:
atlasImage = sitk.ReadImage(atlasPath + atlasName + '_ModelMajorityVoting.mhd')
atlasMeanFFImage = sitk.ReadImage(atlasPath + atlasName + '_MeanFFModelMajorityVoting.mhd')
atlasMedianFFImage = sitk.ReadImage(atlasPath + atlasName + '_MeanFFModelMajorityVoting.mhd')
atlasStdFFImage = sitk.ReadImage(atlasPath + atlasName + '_StdFFMajorityVoting.mhd')
atlasP75FFImage = sitk.ReadImage(atlasPath + atlasName + '_p75FFMajorityVoting.mhd')
atlasP90FFImage = sitk.ReadImage(atlasPath + atlasName + '_p90FFMajorityVoting.mhd')

# Read input image:
# If using links need to get the real path:
shortcut = winshell.shortcut(targetPath + caseToProcess + extensionShortcuts)
indexStart = shortcut.as_string().find(strForShortcut)
dataPath = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'

filename = dataPath + '\\' + 'ForLibrary\\' + caseToProcess + suffixSegmentedImages + extensionImages
filenameFatFraction = dataPath + '\\' + 'ForLibrary\\' + caseToProcess + suffixFatFractionImages + extensionImages
filenameLabels = dataPath + '\\' + 'ForLibrary\\' + caseToProcess + suffixManualMuscleLabels + extensionImages
if not os.path.exists(filename):
    # Check in the automated folder:
    filenameLabels = targetAutomatedSegmentationPath + caseToProcess + '\\' + nameMuscleSegmentedImages + suffixMuscleSegmentedImages + extensionImages
    if not os.path.exists(filenameLabels):
        print('Muscle labels not found for:' + caseToProcess)
        exit(-1)
inputDixonsegmentedImage = sitk.ReadImage(filename)
inputFatFractionImage = sitk.ReadImage(filenameFatFraction)
inputLabelsImage = sitk.ReadImage(filenameLabels)
inputAllLabelsImage = MergeTissueAndLabelsImages(inputDixonsegmentedImage, inputLabelsImage)

sitk.WriteImage(inputFatFractionImage, outputPath + 'inputFF' + extensionImages)

################################ PARAMETERS ########################################
# Parameter files for the registration:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
subfolderNonRigidTest = ''#''WithDeformationPenalty\\'
paramFileRigid = 'Parameters_Rigid_NCC'
useAffine = True
paramFileAffine = 'Parameters_Affine_NCC'

paramFileNonRigid = 'Parameters_BSpline_NMI_2000iters_2048samples'

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


########## REGISTRATION ################
movingImage = inputAllLabelsImage
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
elastixImageFilter.SetFixedImage(atlasImage) # rEGISTER TO THE ATLAS
elastixImageFilter.SetMovingImage(movingImage)
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
# Get the transform:
outputTransformParameterMap = elastixImageFilter.GetTransformParameterMap()# Apply transform for a nearest neighbour output:
# Apply its transform:
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.LogToConsoleOff()
transformixImageFilter.SetMovingImage(movingImage)
transformixImageFilter.SetTransformParameterMap(outputTransformParameterMap)
transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
transformixImageFilter.Execute()
outputImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
sitk.WriteImage(outputImage, outputPath + 'registered_segmented' + extensionImages, False)

# Apply transform to the FF image
# Transfer also the fat fraction images:
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.LogToConsoleOff()
transformixImageFilter.SetTransformParameterMap(outputTransformParameterMap)
transformixImageFilter.SetMovingImage(inputFatFractionImage)
transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "1")
transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
transformixImageFilter.Execute()
outputFatFractionImage = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkFloat32)
sitk.WriteImage(outputFatFractionImage, outputPath + 'registered_fat_fraction' + extensionImages, False)

# Get metric:
# Detect voxels with higher FF than 75 and 90%:
diff = sitk.Subtract(outputFatFractionImage, atlasP75FFImage)
maskP75 = sitk.Greater(diff, 0)
sitk.WriteImage(maskP75, outputPath + 'mask_ver_p75' + extensionImages, True)
diff = sitk.Subtract(outputFatFractionImage, atlasP90FFImage)
maskP90 = sitk.Greater(diff, 0)
sitk.WriteImage(maskP90, outputPath + 'mask_ver_p90' + extensionImages, True)

inverse_transformationFilter = sitk.TransformixImageFilter()
transf_parameter_map = transformixImageFilter.GetTransformParameterMap()
transf_parameter_map[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
transf_parameter_map[0]["InitialTransformParametersFileName"] = ["NoInitialTransform"]
transf_parameter_map[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
transf_parameter_map[1]["InitialTransformParametersFileName"] = ["NoInitialTransform"]
sitk.PrintParameterMap(transf_parameter_map[1])
inverse_transformationFilter.SetMovingImage(outputImage)
inverse_transformationFilter.SetTransformParameterMap(transf_parameter_map)
inverse_transformationFilter.Execute()
outputImage_back = inverse_transformationFilter.GetResultImage()
sitk.WriteImage(outputImage_back, outputPath + 'registered_segmented_back' + extensionImages, False)