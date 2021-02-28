#! python3
# This scripts registers pre marathon cases, which had been previously manually segmented, to the respective post
# marathon scan.


from __future__ import print_function

import SimpleITK as sitk
import winshell
import numpy as np
import sys
import os
import csv

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### PATHS AND CASES TO SEGMENT #######################################
postMarathonPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\' # Base data path.
preMarathonPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\' # Base data path.
subFolderPost = 'Segmented\\'
subFolderPre = 'Segmented\\'
cropAtLesserTrochanter = False # Flag to indicate if a cropping at the level of the lesser trochanter is done to
                                # homogeneize the field of view.

# Get the atlases names and files:
# Look for the folders or shortcuts:
files = os.listdir(postMarathonPath + subFolderPost)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagSequence = '_I'
tagSequenceO = '_O'
tagSequenceF = '_F'
tagSequenceW = '_W'
tagSequenceSTIR = '_STIR'
tagLabels = '_labels'
postCases = [] # Names of the atlases
postCasesImageFilenames = [] # Filenames of the intensity images
postCasesImageOFilenames = []
postCasesImageFFilenames = []
postCasesImageWFilenames = []
postCasesImageSTIRFilenames = []
postCasesLabelsFilenames = [] # Filenames of the label images
postCasesOutputhPaths = []
preCases = [] # Names of the atlases
preCasesImageFilenames = [] # Filenames of the intensity images
preCasesLabelsFilenames = [] # Filenames of the label images
for filename in files:
    name, extension = os.path.splitext(filename)
    # if name is a lnk, get the path:
    if str(extension).endswith(extensionShortcuts):
        # This is a shortcut:
        shortcut = winshell.shortcut(postMarathonPath + subFolderPost + filename)
        indexStart = shortcut.as_string().find(strForShortcut)
        dataPathThisAtlas = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
    else:
        dataPathThisAtlas = postMarathonPath + subFolderPost + filename + '\\'
    # Check if the images are available:
    filename = dataPathThisAtlas + 'ForLibrary\\' + name + tagSequence + '.' + extensionImages
    filenameLabels = dataPathThisAtlas + 'ForLibrary\\' + name + tagLabels + '.' + extensionImages
    if os.path.exists(filename):
        # Atlas name:
        postCases.append(name)
        # Intensity image:
        postCasesImageFilenames.append(filename)
        # Add the other phases:
        postCasesImageOFilenames.append(dataPathThisAtlas + 'ForLibrary\\' + name + tagSequenceO + '.' + extensionImages)
        postCasesImageFFilenames.append(
            dataPathThisAtlas + 'ForLibrary\\' + name + tagSequenceF + '.' + extensionImages)
        postCasesImageWFilenames.append(
            dataPathThisAtlas + 'ForLibrary\\' + name + tagSequenceW + '.' + extensionImages)
        postCasesImageSTIRFilenames.append(
            dataPathThisAtlas + 'ForLibrary\\' + name + tagSequenceSTIR + '.' + extensionImages)
        # Labels image:
        postCasesLabelsFilenames.append(filenameLabels)
        # Output path:
        postCasesOutputhPaths.append(dataPathThisAtlas + 'ForLibrary\\Registered2Pre\\')
        if not os.path.exists(postCasesOutputhPaths[-1]):
            os.makedirs(postCasesOutputhPaths[-1])
        # Check if pre exists and add it to the pre list:
        dataPathThisAtlasPre = preMarathonPath + subFolderPre + name + '\\'
        filenamePre = dataPathThisAtlasPre + 'ForLibrary\\' + name + tagSequence + '.' + extensionImages
        filenameLabelsPre = dataPathThisAtlasPre + 'ForLibrary\\' + name + tagLabels + '.' + extensionImages
        if os.path.exists(filename):
            # Atlas name:
            preCases.append(name)
            # Intensity image:
            preCasesImageFilenames.append(filenamePre)
            # Labels image:
            preCasesLabelsFilenames.append(filenameLabelsPre)
print("Number of post marathon cases: {0}".format(len(postCases)))
print("List of cases: {0}\n".format(postCases))




######################################## LANDMARKS FOR CROPPING AT THE LESSER TROCHANTER #############################
# Get landmarks for lesser trochanter for postop cases:
landmarksFilename = 'LandmarksPost.csv'
tagsLandmarks = ('Cases', 'LT-L', 'LT-R', 'ASIS-L', 'ASIS-R')
lesserTrochanterForPost = np.zeros((len(postCases),2)) # Two columns for left and right lesser trochanters.
# Read the csv file with the landmarks and store them:
atlasNamesInLandmarksFile = list()
lesserTrochLeftInLandmarksFile = list()
lesserTrochRighttInLandmarksFile = list()
with open(postMarathonPath+landmarksFilename, newline='\n') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        atlasNamesInLandmarksFile.append(row[tagsLandmarks[0]])
        lesserTrochLeftInLandmarksFile.append(row[tagsLandmarks[1]])
        lesserTrochRighttInLandmarksFile.append(row[tagsLandmarks[2]])
    # find the lesse trochanter for each atlas in the list of post and pre cases:
for i in range(0, len(postCases)):
    ind = atlasNamesInLandmarksFile.index(
        postCases[i])  # This will throw an exception if the landmark is not available:
    # save the landmarks for left and right lesser trochanter:
    lesserTrochanterForPost[i, 0] = int(lesserTrochLeftInLandmarksFile[ind])
    lesserTrochanterForPost[i, 1] = int(lesserTrochRighttInLandmarksFile[ind])
    # Get the minimum of left and right:
lesserTrochanterForPost = lesserTrochanterForPost.min(axis=1).astype(int)

# Get landmarks for lesser trochanter for preop cases:
landmarksFilename = 'Landmarks.csv'
tagsLandmarks = ('Cases', 'LT-L', 'LT-R', 'ASIS-L', 'ASIS-R')
lesserTrochanterForPre = np.zeros((len(postCases),2)) # Two columns for left and right lesser trochanters.
# Read the csv file with the landmarks and store them:
atlasNamesInLandmarksFile = list()
lesserTrochLeftInLandmarksFile = list()
lesserTrochRighttInLandmarksFile = list()
with open(preMarathonPath+landmarksFilename, newline='\n') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        atlasNamesInLandmarksFile.append(row[tagsLandmarks[0]])
        lesserTrochLeftInLandmarksFile.append(row[tagsLandmarks[1]])
        lesserTrochRighttInLandmarksFile.append(row[tagsLandmarks[2]])
# find the lesse trochanter for each atlas in the list of post and pre cases:
for i in range(0, len(postCases)):
    ind = atlasNamesInLandmarksFile.index(postCases[i]) # This will throw an exception if the landmark is not available:
    # save the landmarks for left and right lesser trochanter:
    lesserTrochanterForPre[i,0] = int(lesserTrochLeftInLandmarksFile[ind])
    lesserTrochanterForPre[i,1] = int(lesserTrochRighttInLandmarksFile[ind])
# Get the minimum of left and right:
lesserTrochanterForPre = lesserTrochanterForPre.min(axis=1).astype(int)


########################################  REGISTRATION ###############################################
# For every post cases, register the images to the pre case and transfer the labels:
for i in range(0, len(postCases)):
    # Read images:
    postImage = sitk.ReadImage(postCasesImageFilenames[i])
    postLabels = sitk.ReadImage(postCasesLabelsFilenames[i])
    preImage = sitk.ReadImage(preCasesImageFilenames[i])

    # Parameter files, only rigid registration:
    parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
    paramFileRigid = 'Parameters_Rigid_NCC'
    paramFileNonRigid = 'Parameters_BSpline_NCC_4000iters_8192samples'#{,'Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(preImage)
    elastixImageFilter.SetMovingImage(postImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get the images:
    post2preImage = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    # Transfer the labels:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(postLabels)
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    post2preLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Transfer the other sequences:
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "2")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
    # Out
    postImageO = sitk.ReadImage(postCasesImageOFilenames[i])
    transformixImageFilter.SetMovingImage(postImageO)
    transformixImageFilter.Execute()
    postImageO = transformixImageFilter.GetResultImage()
    # Water
    postImageW = sitk.ReadImage(postCasesImageWFilenames[i])
    transformixImageFilter.SetMovingImage(postImageW)
    transformixImageFilter.Execute()
    postImageW = transformixImageFilter.GetResultImage()
    # In-phase
    postImageF = sitk.ReadImage(postCasesImageFFilenames[i])
    transformixImageFilter.SetMovingImage(postImageF)
    transformixImageFilter.Execute()
    postImageF = transformixImageFilter.GetResultImage()
    # STIR
    postImageSTIR = sitk.ReadImage(postCasesImageSTIRFilenames[i])
    transformixImageFilter.SetMovingImage(postImageSTIR)
    transformixImageFilter.Execute()
    postImageSTIR = transformixImageFilter.GetResultImage()

    # Write registered image and labels in the output directory, keeping the same filename
    sitk.WriteImage(post2preImage, postCasesOutputhPaths[i] + postCases[i] + tagSequence + '.' + extensionImages)
    sitk.WriteImage(post2preLabels, postCasesOutputhPaths[i] + postCases[i] + tagLabels + '.' + extensionImages)
    sitk.WriteImage(postImageO, postCasesOutputhPaths[i] + postCases[i] + tagSequenceO + '.' + extensionImages)
    sitk.WriteImage(postImageW, postCasesOutputhPaths[i] + postCases[i] + tagSequenceW + '.' + extensionImages)
    sitk.WriteImage(postImageF, postCasesOutputhPaths[i] + postCases[i] + tagSequenceF + '.' + extensionImages)
    sitk.WriteImage(postImageSTIR, postCasesOutputhPaths[i] + postCases[i] + tagSequenceSTIR + '.' + extensionImages)
