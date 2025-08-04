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
studyPath = '/home/martin/data_imaging/Muscle/data_sherpas/'
mhdData = studyPath + '/MHDs/'
mhdOutputData = studyPath + '/MHDsCompressed/'
niftiOutputData = studyPath + '/Nifti/'
filenameParticipants = studyPath + 'ParticipantsDemographics.xlsx'

if not os.path.exists(mhdOutputData):
    os.makedirs(mhdOutputData)

if not os.path.exists(niftiOutputData):
    os.makedirs(niftiOutputData)

# Read partcipants demographics:
import pandas as pd
participants = pd.read_excel(filenameParticipants, sheet_name='Sheet1')

# Look for the folders or shortcuts:
data = os.listdir(mhdData)
data = sorted(data)

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
extensionNiftiImages = '.nii'
tagInPhase = ''

folderIndex = []
tagArray = []
for filename in data:
    name, extension = os.path.splitext(filename)
    row = participants[participants['Filenames'] == name]
    if not row.empty:
        code = row.iloc[0]['Code']
        mhdOutputDataThisSubject = mhdOutputData + code + os.path.sep
        if not os.path.exists(mhdOutputDataThisSubject):
            os.makedirs(mhdOutputDataThisSubject)
        niftiOutputDataThisSubject = niftiOutputData + code + os.path.sep
        if not os.path.exists(niftiOutputDataThisSubject):
            os.makedirs(niftiOutputDataThisSubject)
        
    else:
        print(f"Filename '{name}' not found in participants.")
        continue
    if os.path.isdir(mhdData + name):
        dataInSubdir = os.listdir(mhdData + name)
        for filenameInSubdir in dataInSubdir:
            nameInSubdir, extension = os.path.splitext(filenameInSubdir)
            if extension == extensionImages:
                # Read image and write it again in the output folder with the new name and in compressed format:
                image = sitk.ReadImage(mhdData + name + os.path.sep + filenameInSubdir)
                # Replace the name with code but leaving the suffix:
                outputMhdFilenameInSubdir = mhdOutputDataThisSubject + nameInSubdir.replace(name, code) + extensionImages
                outputNiftiFilenameInSubdir = niftiOutputDataThisSubject + nameInSubdir.replace(name, code) + extensionNiftiImages
                sitk.WriteImage(image, outputMhdFilenameInSubdir, True)
                sitk.WriteImage(image, outputNiftiFilenameInSubdir, True)
                print("Written image: {0}".format(outputMhdFilenameInSubdir))
    else:
        print("File not in a subfolder: {0}".format(mhdData + filename))

