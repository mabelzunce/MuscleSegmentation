#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SimpleITK as sitk
import SitkImageManipulation as sitkIm
import winshell
import numpy as np
import sys
import os
import csv

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### LIBRARY ATLASES, CONFIGURATION, ETC FOR THIS VERSION ###################################
dataPath = 'D:\\Martin\\Data\\MuscleSegmentation\\' # Base data path.
libraryVersion = 'V1.2'
atlasesPath = dataPath + 'Library' + libraryVersion + '\\'
cropAtLesserTrochanter = False # Flag to indicate if a cropping at the level of the lesser trochanter is done to
                                # homogeneize the field of view.

# Get the atlases names and files:
# Look for the folders or shortcuts:
files = os.listdir(atlasesPath)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagSequence = '_I_bias'
tagLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
for filename in files:
    name, extension = os.path.splitext(filename)
    # if name is a lnk, get the path:
    if str(extension).endswith(extensionShortcuts):
        # This is a shortcut:
        shortcut = winshell.shortcut(atlasesPath + filename)
        indexStart = shortcut.as_string().find(strForShortcut)
        dataPathThisAtlas = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
    else:
        dataPathThisAtlas = atlasesPath + filename + '\\'
    # Check if the images are available:
    filename = dataPathThisAtlas + 'ForLibrary\\' + name + tagSequence + '.' + extensionImages
    filenameAtlas = dataPathThisAtlas + 'ForLibrary\\' + name + tagLabels + '.' + extensionImages
    if os.path.exists(filename):
        # Atlas name:
        atlasNames.append(name)
        # Intensity image:
        atlasImageFilenames.append(filename)
        # Labels image:
        atlasLabelsFilenames.append(filenameAtlas)
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Get landmarks for lesser trochanter (in case a cropping i performed for each atlases):
landmarksFilename = 'Landmarks.csv'
tagsLandmarks = ('Cases', 'LT-L', 'LT-R', 'ASIS-L', 'ASIS-R')
lesserTrochanterForAtlases = np.zeros((len(atlasNames),2)) # Two columns for left and right lesser trochanters.
# Read the csv file with the landmarks and store them:
atlasNamesInLandmarksFile = list()
lesserTrochLeftInLandmarksFile = list()
lesserTrochRighttInLandmarksFile = list()
with open(dataPath+landmarksFilename, newline='\n') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        atlasNamesInLandmarksFile.append(row[tagsLandmarks[0]])
        lesserTrochLeftInLandmarksFile.append(row[tagsLandmarks[1]])
        lesserTrochRighttInLandmarksFile.append(row[tagsLandmarks[2]])
# find the lesse trochanter for each atlas in the library:
for i in range(0, len(atlasNames)):
    ind = atlasNamesInLandmarksFile.index(atlasNames[i]) # This will throw an exception if the landmark is not available:
    # save the landmarks for left and right lesser trochanter:
    lesserTrochanterForAtlases[i,0] = int(lesserTrochLeftInLandmarksFile[ind])
    lesserTrochanterForAtlases[i,1] = int(lesserTrochRighttInLandmarksFile[ind])
# Get the minimum of left and right:
lesserTrochanterForAtlases = lesserTrochanterForAtlases.min(axis=1).astype(int)

# Labels:
numLabels = 11 # 10 for muscles and bone


############################## OUTPUT ######################
# Library path:
baseLibraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\'
rawLibraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\'
normalizedLibraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\Normalized\\'
rigidLibraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\Rigid\\'
affineLibraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\Affine\\'
if not os.path.exists(rawLibraryPath):
    os.makedirs(rawLibraryPath)
if not os.path.exists(normalizedLibraryPath):
    os.makedirs(normalizedLibraryPath)
if not os.path.exists(rigidLibraryPath):
    os.makedirs(rigidLibraryPath)
if not os.path.exists(affineLibraryPath):
    os.makedirs(affineLibraryPath)

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg


############################# USE ONE CASE AS A REFERENCE FOR THE REGISTRATION #####################
indexReference = 5#len(atlasNames) - 2
print("Case use as a reference for the registration: {0}\n".format(atlasNames[indexReference]))
refAtlasImage = sitk.ReadImage(atlasImageFilenames[indexReference])

##########################################################################################
################################### PREPARE ALL THE IMAGES ###################################
for i in range(0, len(atlasNames)):

    ############## 0) RAW IMAGE WITH BIAS CORRECTION #############
    # Read target image:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])

    # Cast the image as float:
    atlasImage = sitk.Cast(atlasImage, sitk.sitkFloat32)
    # if requested crop at the lesser trochanter:
    if cropAtLesserTrochanter:
        atlasImage = atlasImage[:,:,lesserTrochanterForAtlases[i]:]
        atlasLabels = atlasLabels[:, :, lesserTrochanterForAtlases[i]:]
    # If we don't want to keep the cooridnate system in the header:
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        sitkIm.ResetImageCoordinates(atlasImage)
        sitkIm.ResetImageCoordinates(atlasLabels)
    # Apply bias correction:
    shrinkFactor = (1,1,1) # Having problems otherwise:
#    atlasImage = ApplyBiasCorrection(atlasImage, shrinkFactor)

    # Write image in the raw folder:
    sitk.WriteImage(atlasImage, rawLibraryPath + atlasNames[i] + '.' + extensionImages)
    sitk.WriteImage(atlasLabels, rawLibraryPath + atlasNames[i] + tagLabels + '.' + extensionImages)

    ########################################################################
    ##### MOVE BACK FILES
    # Move the files for the atlas to the backupfolder:
    #for fileToMove in filesToMove:
    #    os.rename(backupFolder + fileToMove, libraryPath + fileToMove)

    ############## 1) RIGID REGISTRATION #############
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(refAtlasImage)
    elastixImageFilter.SetMovingImage(atlasImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasImage = elastixImageFilter.GetResultImage()
    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasLabels)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    atlasLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Write image in the raw folder:
    sitk.WriteImage(atlasImage, rigidLibraryPath + atlasNames[i] + '.' + extensionImages)
    sitk.WriteImage(atlasLabels, rigidLibraryPath + atlasNames[i] + tagLabels + '.' + extensionImages)

    ############### 2) AFFINE REGISTRATION ########################
#    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
#                                                                   + paramFileAffine + '.txt'))
#    elastixImageFilter.SetFixedImage(refAtlasImage)
#    elastixImageFilter.SetMovingImage(atlasImage)
#    elastixImageFilter.SetParameterMap(parameterMapVector)
#    elastixImageFilter.Execute()
#    # Get result and apply transform to labels:
#    # Get the images:
#    atlasImage = elastixImageFilter.GetResultImage()
#    # Apply transform:
#    transformixImageFilter = sitk.TransformixImageFilter()
#    transformixImageFilter.SetMovingImage(atlasLabels)
#    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
#    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
#    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
#    transformixImageFilter.Execute()
#    atlasLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
#    # Write image in the raw folder:
#    sitk.WriteImage(atlasImage, affineLibraryPath + atlasNames[i] + '.' + extensionImages)
#    sitk.WriteImage(atlasLabels, affineLibraryPath + atlasNames[i] + tagLabels + '.' + extensionImages)