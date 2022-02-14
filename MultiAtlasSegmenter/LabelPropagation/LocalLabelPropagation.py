#! python3


from __future__ import print_function

import SimpleITK as sitk
from LocalNormalizedCrossCorrelation import LocalNormalizedCrossCorrelation
import numpy as np
import sys
import os

##### SCRIPT TO TEST LABEL PROPAGATION ##################

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### TARGET FOLDER ###################################
caseName = "ID00003"
#caseName = "7390413"
dixonTags = ("I","O",'W',"F")
if caseName.startswith('ID'):
    basePath = "D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\{0}\\ForLibrary\\".format(caseName)
else:
    basePath = "D:\\Martin\\Data\\MuscleSegmentation\\ForLibrary\\{0}\\ForLibrary\\".format(caseName)

dixonImages = []
for i in range(0,len(dixonTags)):
    # Read target image:
    targetImageFilename = basePath + caseName + "_" + dixonTags[i] + ".mhd"
    dixonImages.append(sitk.ReadImage(targetImageFilename))
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        dixonImages[i].SetOrigin((0,0,0))
        dixonImages[i].SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
############################### TARGET FOLDER AND IMAGE ###################################
libraryVersion = 'V1.0'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\' #'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\'
# Look for the raw files in the library:
files = os.listdir(targetPath)
extensionImages = 'mhd'
targetImageFilename = targetPath + caseName + '_bias.mhd'
targetLabelsFilename = targetPath + caseName + '_labels.mhd'
fixedImage = sitk.ReadImage(targetImageFilename)
if not USE_COSINES_AND_ORIGIN:
    # Reset the origin and direction to defaults.
    fixedImage.SetOrigin((0,0,0))
    fixedImage.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
path, filename = os.path.split(targetImageFilename)
nameFixed, extension = os.path.splitext(filename)
#nameCaseFixed = nameFixed
index_dash = nameFixed.index('_')
nameCaseFixed = nameFixed[:index_dash]

###################### OUTPUT #####################
# Output path:
outputBasePath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\NonrigidNCC_N5_MaxProb_PluginTest\\'
if not os.path.exists(outputBasePath):
    os.mkdir(outputBasePath)
outputPath = outputBasePath + nameCaseFixed + "\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\Normalized\\"

# Number of Atlases to select:
numSelectedAtlases = 5

# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################


############# 1) ATLAS LIBRARY ##############################
# To the function I just need to pass the folder name, but first we need to be sure
# that the atlas is not in the library folder:
backupFolder = libraryPath + "OutOfLibrary\\"
if not os.path.exists(backupFolder):
    os.mkdir(backupFolder)
files = os.listdir(libraryPath)
extensionImages = 'mhd'
filesToMove = []
for filename in files:
    if filename.startswith(nameCaseFixed):
        # Add to files to move:
        filesToMove.append(filename)
# Move the files for the atlas to the backupfolder:
for fileToMove in filesToMove:
    os.rename(libraryPath + fileToMove, backupFolder + fileToMove)

############## 2) REGISTER THE IMAGE AND GET LOCAL SIMILARITY METRICS ###########################
############### 2) IMAGE REGISTRATION ###########################
# 1) Image registration between atlases and target images:
registeredImages = []
transformParameterMaps = []
similarityValue = []
similarityValueElastix = []
# Register to each atlas:
for i in range(0, atlasImagesNames.__len__()):
    filenameAtlas = atlasImagesNames[i]
    movingImage = sitk.ReadImage(libraryPath + filenameAtlas)
    nameMoving, extension = os.path.splitext(filenameAtlas)
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileBspline + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(targetImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.SetOutputDirectory(tempPath)
    #logFilename = 'reg_log_{0}'.format(i) + '.txt' # iT DOESN'T WORK WITH DIFFERENT LOG NAMES
    logFilename = 'reg_log' + '.txt'
    elastixImageFilter.SetLogFileName(logFilename)
    elastixImageFilter.Execute()
    # Get the images:
    registeredImages.append(elastixImageFilter.GetResultImage())
    transformParameterMaps.append(elastixImageFilter.GetTransformParameterMap())
    # Get the similarity value:
    fullLogFilename = tempPath + logFilename
    # Compute normalized cross correlation:
    imRegMethod = sitk.ImageRegistrationMethod()
    imRegMethod.SetMetricAsCorrelation()
    metricValue = imRegMethod.MetricEvaluate(targetImage, registeredImages[i])
    # metricValue = sitk.NormalizedCorrelation(registeredImages[i], mask, targetImage) # Is not working
    similarityValue.append(metricValue)
    similarityValueElastix.append(GetFinalMetricFromElastixLogFile(fullLogFilename))
    print(similarityValue[i])
    # If debugging, write image:
    if debug:
        outputFilename = outputPath + '\\' + nameMoving + '_to_target' + '.mhd'
        sitk.WriteImage(registeredImages[i], outputFilename)

###########################################

########################################################################
##### MOVE BACK FILES
# Move the files for the atlas to the backupfolder:
for fileToMove in filesToMove:
    os.rename(backupFolder + fileToMove, libraryPath + fileToMove)