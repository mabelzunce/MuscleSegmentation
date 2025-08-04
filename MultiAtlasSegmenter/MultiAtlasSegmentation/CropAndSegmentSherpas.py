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
mhdData = studyPath + '/MHDsCompressed/'
niftiData = studyPath + '/Nifti/'
mhdCroppedData = studyPath + '/MHDsCropped/'
niftiOCroppedData = studyPath + '/NiftiCropped/'
filenameParticipants = studyPath + 'ParticipantsDemographics.xlsx'

# Export formats:
exportFormatMhd = True
exportFormatNifti = True

if not os.path.exists(mhdCroppedData):
    os.makedirs(mhdCroppedData)

if not os.path.exists(niftiOCroppedData):
    os.makedirs(niftiOCroppedData)

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
tagInPhase = '_I'
tagOutOfPhase = '_O'
tagWater = '_W'
tagFat = '_F'  
tagsDixon = [tagInPhase, tagOutOfPhase, tagWater, tagFat]

folderIndex = []
tagArray = []
for filename in data:
    name, extension = os.path.splitext(filename)
    if os.path.isdir(mhdData + name):
        row = participants[participants['Code'] == name]
        if not row.empty:
            print(f"Processing '{name}' participant.")       
        else:
            print(f"Filename '{name}' not found in participants.")
            continue
        code = row.iloc[0]['Code']
        topIliacCrest = row.iloc[0]['Top Iliac Crest']
        lesserTrochanter = row.iloc[0]['Lesser Trochanter']
        mhdCroppedDataThisSubject = mhdCroppedData + code + os.path.sep
        if not os.path.exists(mhdCroppedDataThisSubject):
            os.makedirs(mhdCroppedDataThisSubject)
        niftiOCroppedDataThisSubject = niftiOCroppedData + code + os.path.sep
        if not os.path.exists(niftiOCroppedDataThisSubject):
            os.makedirs(niftiOCroppedDataThisSubject)

        dataInSubdir = os.listdir(mhdData + name)
        filenameDixon = list()
        dixonImages = list()
        i = 0
        for tag in tagsDixon:
            filenameDixon.append(name + tag + extensionImages)
            if filenameDixon[i] in dataInSubdir:
                filenameInSubdir = filenameDixon[i]
                # Read exch Dixon image:
                dixonImages.append(sitk.ReadImage(mhdData + name + os.path.sep + filenameDixon[i]))
            else:
                dixonImages.append([])
                print(f"File '{filenameDixon[i]}' not found for {name}.")
            i = i + 1

        # Check if we have F and W images:
        if dixonImages[2] != [] and dixonImages[3] != []:
            waterfatImage = sitk.Cast(sitk.Add(dixonImages[3], dixonImages[2]),
                sitk.sitkFloat32)
            fatfractionImage = sitk.Divide(sitk.Cast(dixonImages[3],
                sitk.sitkFloat32), waterfatImage)
            fatfractionImage = sitk.Cast(
                sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                sitk.sitkFloat32)
            # Save the resulting image:
            if exportFormatMhd:
                output_filename = os.path.join(mhdData, name + '_ff' + extensionImages)
                sitk.WriteImage(fatfractionImage, output_filename, True)
                print(f"FF Image saved in: {output_filename}")
            if exportFormatNifti:
                output_filename = os.path.join(niftiData, name + '_ff' + extensionNiftiImages)
                sitk.WriteImage(fatfractionImage, output_filename, True)
                print(f"FF Image saved in: {output_filename}")
        else:
            print(f"Missing W and/or F images for {name}.")


        for i in range(0, len(dixonImages)):
            # Crop the image based on the top iliac crest and lesser trochanter:
            imageCropped = dixonImages[i][:,:, int(lesserTrochanter):int(topIliacCrest)]
            # Replace the name with code but leaving the suffix:
            tag = tagsDixon[i]
            if exportFormatMhd:
                outputMhdFilenameInSubdir = mhdCroppedDataThisSubject + code + tag + extensionImages
                sitk.WriteImage(imageCropped, outputMhdFilenameInSubdir, True)
                print("Written image: {0}".format(outputMhdFilenameInSubdir))
            if exportFormatNifti:
                outputNiftiFilenameInSubdir = niftiOCroppedDataThisSubject + code + tag + extensionNiftiImages
                sitk.WriteImage(imageCropped, outputNiftiFilenameInSubdir, True)
                print("Written image: {0}".format(outputNiftiFilenameInSubdir))

            # Now segment the gluteal muscles if its in phase:
            if tag is tagInPhase:
                # segment
                print("Segmenting gluteal muscles for {0}...".format(name))
        # Crop the fat fraction image:
        fatfractionImageCropped = fatfractionImage[:,:, int(lesserTrochanter):int(topIliacCrest)]
        outputFatFractionFilename = niftiOCroppedDataThisSubject + code + '_ff' + extensionNiftiImages
        sitk.WriteImage(fatfractionImageCropped, outputFatFractionFilename, True)
        print("Written image: {0}".format(outputFatFractionFilename))
    else:
        print("File not in a subfolder: {0}".format(mhdData + filename))

