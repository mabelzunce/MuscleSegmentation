#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import warnings
import sys
sys.path.append('../../MultiAtlasSegmenter/MultiAtlasSegmentation/')
import SegmentationPerformanceMetrics

#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0        # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
numLabels = 8
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
paramFileNonRigid = 'Parameters_BSpline_NCC_500iters_2048samples'  # Par0000bspline_500'

manualSegmentationPath = '/home/martin/data_imaging/Muscle/LumbarSpine/Manual/'
automatedSegmentationPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Processed/'# Base data path.
dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/'
outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/PseudoDice/'# Base data path.
#dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/1stPhase/C00011/'# Base data path.
#outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/C00011/'# Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

correctNames = True # If names don't have a C at the start fo the ID, add it.

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_I'
postDflt = ''#'reburnt' #''
postSuffix = 'post'
t3Suffix = 'T3'
tagAutLabels = '_segmentation'
tagManLabels = '_labels'
atlasImageFilenames = [] # Filenames of the intensity images
atlasNames = ["C00043", "C00074", "C00108", "C00115"] # Two females, two males. Two cto5k high bmi, two cyclists low bmi.

# Get the filenames of the atlases:
filenamesAtlasInPhase = list()
filenamesAtlasSeg = list()
for atlasName in atlasNames:
    filenameAtlasInPhase = dataPath + atlasName + os.path.sep + atlasName + tagInPhase + extensionImages
    filenameAtlasSegm = manualSegmentationPath + atlasName + tagManLabels + extensionImages
    if os.path.exists(filenameAtlasInPhase) and os.path.exists(filenameAtlasSegm):
        filenamesAtlasInPhase.append(filenameAtlasInPhase)
        filenamesAtlasSeg.append(filenameAtlasSegm)
    else:
        warnings.warn("Atlas {0} missing.".format(atlasName))


# LOOK FOR THE FOLDERS OF THE SEGMENTED IMAGES TO EVALUATE:
#files = os.listdir(dataPath)
nameSubjectsToEvaluate = list()
filenamesAutoSeg = list()
subdirs = os.listdir(automatedSegmentationPath) #Folders
casesToProcess = sorted(subdirs)
casesToProcess = ["C00001", "C00025", "C00027", "C00036", "C00038", "C00043", "C00047", "C00050", "C00061", "C00062",
                  "C00063", "C00071", "C00074", "C00075", "C00077", "C00090", "C00098", "C00104", "C00108", "C00109",
                  "C00111", "C00112", "C00115", "C00119", "C00123", "C00125"]

for subjectDir in casesToProcess:
    name, extension = os.path.splitext(subjectDir)
    if os.path.isdir(automatedSegmentationPath + name):
        dataInSubdir = os.listdir(automatedSegmentationPath + name)
        for filenameInSubdir in dataInSubdir:
            nameInSubdir, extension = os.path.splitext(filenameInSubdir)
            if (extension == extensionImages and nameInSubdir.endswith(tagAutLabels)):
                filenamesAutoSeg.append(automatedSegmentationPath + name + os.path.sep + filenameInSubdir)
                nameSubjectsToEvaluate.append(name)

# GET THE IN-PHASE IMAGES FOR THESE SUBJECTS
filanmesInPhase = [] # Filenames of the intensity images
for i in range(len(nameSubjectsToEvaluate)):
    filenameInPhase = dataPath + nameSubjectsToEvaluate[i] + os.path.sep + nameSubjectsToEvaluate[i] + tagInPhase + extensionImages
    if os.path.exists(filenameInPhase):
        filanmesInPhase.append(filenameInPhase)
    else:
        filanmesInPhase.append("")
        warnings.warn("Subject {0} missing".format(nameSubjectsToEvaluate[i]))


# START GETTING PSEUDO DICE SCORES
# By registering each segmentation to the 4 atlases.
pseudoDicesScores = np.zeros([len(nameSubjectsToEvaluate), len(atlasNames), numLabels])
pseudoDicesScoresOverall = np.zeros([len(nameSubjectsToEvaluate), len(atlasNames)])
similarityMetrics = np.zeros([len(nameSubjectsToEvaluate), len(atlasNames)])
for i in range(len(nameSubjectsToEvaluate)):
    # Read image and automated segmentation:
    autSeg = sitk.ReadImage(filenamesAutoSeg[i])
    inphaseAutSeg = sitk.ReadImage(filanmesInPhase[i])
    print('Processing {0}'.format(nameSubjectsToEvaluate[i]))
    # Now register (affine) each image to the atlas and compare segmentation:
    for j in range(len(atlasNames)):

        atlasImage = sitk.ReadImage(filenamesAtlasInPhase[j])
        atlasLabels = sitk.ReadImage(filenamesAtlasSeg[j])

        # Register the images:
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
        elastixImageFilter.SetFixedImage(inphaseAutSeg)
        elastixImageFilter.SetMovingImage(atlasImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.LogToConsoleOff()
        # Execute
        elastixImageFilter.Execute()
        # Get the image:
        registeredImage =  elastixImageFilter.GetResultImage()
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()

        # Apply the transform to the labels:
        # Apply its transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.SetMovingImage(atlasLabels)
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        registeredLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        if DEBUG:
            sitk.WriteImage(registeredImage, outputPath + nameSubjectsToEvaluate[i] + "_" + atlasNames[j] + ".mhd", True)
            sitk.WriteImage(registeredLabels, outputPath + nameSubjectsToEvaluate[i] + "_" + atlasNames[j] + "_labels.mhd", True)

        # Compute normalized cross correlation:
        imRegMethod = sitk.ImageRegistrationMethod()
        imRegMethod.SetMetricAsCorrelation()
        similarityMetrics[i,j] = imRegMethod.MetricEvaluate(sitk.Cast(inphaseAutSeg, sitk.sitkFloat32), registeredImage)

        # Compute the dice score:
        overlapMetrics = SegmentationPerformanceMetrics.GetOverlapMetrics(autSeg, registeredLabels, numLabels+1) #inphaseAutSeg as reference, the registered atlas the moving
        pseudoDicesScores[i,j,:] = overlapMetrics["dice"][1:numLabels+1]
        overlapMetricsOverall = SegmentationPerformanceMetrics.GetOverlapMetrics(autSeg, registeredLabels, 0)
        pseudoDicesScoresOverall[i, j] = overlapMetricsOverall["dice"]
        # Save csvs:
        for k in range(numLabels):
            np.savetxt(outputPath + "pseudoDicesScores_label_{0}.csv".format(k+1), pseudoDicesScores[:,:,k], delimiter=",")
        np.savetxt(outputPath + "pseudoDicesScoresOverall.csv", pseudoDicesScoresOverall, delimiter=",")
        np.savetxt(outputPath + "similarityMetrics.csv", similarityMetrics, delimiter=",")
        np.savetxt(outputPath + "subjectNames.csv", nameSubjectsToEvaluate[:i], delimiter=",", fmt='%s')



