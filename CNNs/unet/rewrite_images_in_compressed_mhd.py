#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
dataTimePoint1Path = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/1stPhase/'
dataTimePoint2Path = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/2ndPhase/'
dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/Reburnt/'# Base data path.
outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed2/'# Base data path.
#dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/1stPhase/C00011/'# Base data path.
#outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/C00011/'# Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

correctNames = True # If names don't have a C at the start fo the ID, add it.

# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = ''
postDflt = 'reburnt'#'reburnt'#'reburnt' #''
postSuffix = 'post'
t3Suffix = 'T3'
checkTimePoint1 = False
checkTimePoint2 = False
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images

folderIndex = []
tagArray = []
#data = data[20:]
for filename in data:
    name, extension = os.path.splitext(filename)
    if os.path.isdir(dataPath + name):
        dataInSubdir = os.listdir(dataPath + name)
        for filenameInSubdir in dataInSubdir:
            nameInSubdir, extension = os.path.splitext(filenameInSubdir)
            if extension == extensionImages:
                # Read image and write it again:
                image = sitk.ReadImage(dataPath + name + os.path.sep + filenameInSubdir)
                if name[0] != "C":
                    outputName = "C" + name + postDflt
                else:
                    outputName = name + postDflt
                if filenameInSubdir[0] != "C":
                    nameInSubdir = "C" + nameInSubdir
                # If exists in first time point, this is the post scan, I check for the standard name and without the first letter in case there is no C in the nae.
                if (os.path.isdir(dataTimePoint1Path + name) or os.path.isdir(dataTimePoint1Path + name[1:])) and checkTimePoint1:
                    splitName = nameInSubdir.split("_")  # The images have suffixes
                    # There are three time points in some cases:
                    if os.path.isdir(dataTimePoint2Path + name) and checkTimePoint2:
                        outputName = outputName + t3Suffix
                        filenameInSubdir = splitName[0] + t3Suffix + "_" + splitName[1]
                    else:
                        outputName = outputName + postSuffix
                        filenameInSubdir = splitName[0] + postSuffix + "_" + splitName[1]
                else:
                    filenameInSubdir = nameInSubdir
                outputFilenameInSubdir = filenameInSubdir + extension
                if not os.path.exists(outputPath + outputName):
                    os.makedirs(outputPath + outputName)
                sitk.WriteImage(image, outputPath + outputName + os.path.sep + outputFilenameInSubdir, True)
                # print("Written image: {0}".format(outputPath + filename))
    else:
        if extension == extensionImages:
            # Read image and write it again:
            image = sitk.ReadImage(dataPath + filename)
            sitk.WriteImage(image, outputPath + filename, True)
            #print("Written image: {0}".format(outputPath + filename))



