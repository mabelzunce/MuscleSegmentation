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
manualSegmentationPath = '/home/martin/data_imaging/Muscle/LumbarSpine/ManualSegmentations/'
dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/Reburnt/'# Base data path.
outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/'# Base data path.
#dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/1stPhase/C00011/'# Base data path.
#outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/C00011/'# Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

correctNames = True # If names don't have a C at the start fo the ID, add it.

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = ''
postDflt = 'reburnt'#'reburnt' #''
postSuffix = 'post'
t3Suffix = 'T3'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images

# Check manual segmentations available:
data = os.listdir(dataPath)
data = sorted(data)

folderIndex = []
tagArray = []
#data = data[20:]
for filename in data:
    name, extension = os.path.splitext(filename)




