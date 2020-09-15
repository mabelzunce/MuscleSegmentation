from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
import DixonTissueSegmentation
import matplotlib.pyplot as plt
import SimpleITK as sitk
import PostprocessingLabels
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### TARGET FOLDER ###################################
caseName = "ID00003"
#caseName = "7390413"
dixonTags = ("I","O",'W',"F")
if caseName.startswith('ID'):
    basePath = "D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\Segmented\\{0}\\ForLibrary\\".format(caseName)
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

###################### OUTPUT #####################
# Output path:
outputBasePath = "D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\V1.0\\NonrigidNCC_N5_MaxProb_DixonMask\\"
if not os.path.exists(outputBasePath):
    os.mkdir(outputBasePath)
outputPath = outputBasePath + caseName + "\\"
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
    if filename.startswith(caseName):
        # Add to files to move:
        filesToMove.append(filename)
# Move the files for the atlas to the backupfolder:
for fileToMove in filesToMove:
    os.rename(libraryPath + fileToMove, backupFolder + fileToMove)

########################################################################
################ CALL MULTI ATLAS SEGMENTATION #########################
# Target image in-phase:     dixonImages[0]
# Apply bias correction filter:
#inputImage = sitk.Shrink(fixedImage, [int(sys.argv[3])] * inputImage.GetDimension())
#maskImage = sitk.Shrink(maskImage, [int(sys.argv[3])] * inputImage.GetDimension())
#biasFilter = sitk.N4BiasFieldCorrectionImageFilter()
#biasFilter.
#fixedImage = sitk.N4BiasFieldCorrection(fixedImage)
############################################

# Soft tissue masL
# 1)Not any restriction:
#softTissueMask = sitk.Greater(dixonImages[0], 0)
# 2) Dixon soft tissue:
dixonSegmentedImage = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages)
softTissueMask = dixonSegmentedImage == 1 #sitk.Equal(dixonSegmentedImage,1)
fatTissueMask = dixonSegmentedImage == 3
segmented2D = PostprocessingLabels.FilterUnconnectedRegionsPerSlices(fatTissueMask, 1)

plt.figure()
plt.imshow(sitk.GetArrayViewFromImage(segmented2D)[50, :, :], cmap=plt.cm.Greys_r)
# Show dixon segmented image and the soft tissue mask:
#plt.subplot(121)
#z = int(dixonSegmentedImage.GetDepth() / 2)
#plt.imshow(sitk.GetArrayViewFromImage(dixonSegmentedImage)[z, :, :], cmap=plt.cm.Greys_r)
#plt.axis('off')
#plt.title("Dixon Segmentation")
#plt.subplot(122)
#plt.imshow(sitk.GetArrayViewFromImage(softTissueMask)[z, :, :], cmap=plt.cm.Greys_r)
#plt.axis('off')
#plt.title("Soft Tissue Mask")
#plt.show()

########################################################################
################ CALL MULTI ATLAS SEGMENTATION #########################
# Call the multiatlas segmentation with the in phase image:
dixonImages[0] = sitk.Cast(dixonImages[0], sitk.sitkFloat32)
# Set the mask with the same properties as the target image:
softTissueMask.SetSpacing(dixonImages[0].GetSpacing())
softTissueMask.SetOrigin(dixonImages[0].GetOrigin())
softTissueMask.SetDirection(dixonImages[0].GetDirection())
if DEBUG:
    sitk.WriteImage(softTissueMask, outputPath + 'softTissueMask.mhd')
    sitk.WriteImage(dixonSegmentedImage, outputPath + 'dixonSegmentedImage.mhd')
dictResults = MultiAtlasSegmentation(dixonImages[0], softTissueMask, libraryPath, outputPath, DEBUG)

########################################################################
##### MOVE BACK FILES
# Move the files for the atlas to the backupfolder:
for fileToMove in filesToMove:
    os.rename(backupFolder + fileToMove, libraryPath + fileToMove)
