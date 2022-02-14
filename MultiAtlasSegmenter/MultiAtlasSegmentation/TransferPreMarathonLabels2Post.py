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
preMarathonAutomatedPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\V1.3\\NonrigidBSplineStandardGradDesc_NMI_2000iters_3000samples_15mm_RndSparseMask_N5_MaxProb_Mask\\' # path with automated segmentations for those cases where there are not automated.
subFolderPost = 'NotSegmented\\'
subFolderPre = 'AllWithLinks\\'
subFolderPreSegmented = 'Segmented\\'
subFolderPreNotSegmented = 'NotSegmented\\'
cropAtLesserTrochanter = False # Flag to indicate if a cropping at the level of the lesser trochanter is done to
                                # homogeneize the field of view.
# Image registration parameter files:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid_NCC'
paramFileNonRigid = 'Parameters_BSpline_NCC_4000iters_8192samples'#'Parameters_BSpline_NCC_4000iters_8192samples'#{,'Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}


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
# For automated segmentations:
outputType = '' # '', 'Mask', 'LWV', 'GWV', 'STAPLES',
automatedLabelsName = 'segmentedImage' + outputType + '.mhd'
automatedCorrectedLabelsName = 'segmentedImage_ManuallyCorrected' + outputType + '.mhd'
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
    if os.path.exists(filename) and not os.path.exists(filenameLabels): # Only process this scan if has the in phase iamge but not the labels image.
        # Output path:
        postCasesOutputhPaths.append(dataPathThisAtlas + 'ForLibrary\\PropagatedPreLabels\\' + paramFileNonRigid + '\\')
        if not os.path.exists(postCasesOutputhPaths[-1]):
            os.makedirs(postCasesOutputhPaths[-1])
        # Check if pre exists and add it to the pre list:
        if os.path.exists(preMarathonPath + subFolderPreSegmented + name):
            dataPathThisAtlasPre = preMarathonPath + subFolderPreSegmented + name + '\\'
        else:
            dataPathThisAtlasPre = preMarathonPath + subFolderPreNotSegmented + name + '\\'
        filenamePre = dataPathThisAtlasPre + 'ForLibrary\\' + name + tagSequence + '.' + extensionImages
        filenameLabelsPre = dataPathThisAtlasPre + 'ForLibrary\\' + name + tagLabels + '.' + extensionImages
        filenameAutomatedCorrectedLabelsPre = preMarathonAutomatedPath + name + '\\' + automatedCorrectedLabelsName
        filenameAutomatedLabelsPre = preMarathonAutomatedPath + name + '\\' + automatedLabelsName
        # Only add images to do the list if both images pre and post exist.
        if os.path.exists(filenamePre) and os.path.exists(filenameLabelsPre): # Check if the labels exist.
            # Atlas name:
            postCases.append(name)
            # Intensity image:
            postCasesImageFilenames.append(filename)
            # Atlas name:
            preCases.append(name)
            # Intensity image:
            preCasesImageFilenames.append(filenamePre)
            # Labels image:
            preCasesLabelsFilenames.append(filenameLabelsPre)
        elif os.path.exists(filenamePre) and os.path.exists(filenameAutomatedCorrectedLabelsPre): # We need to use the automated labels, the first option would be automated but post-corrected.
            # Atlas name:
            postCases.append(name)
            # Intensity image:
            postCasesImageFilenames.append(filename)
            # Atlas name:
            preCases.append(name)
            # Intensity image:
            preCasesImageFilenames.append(filenamePre)
            # Labels image:
            preCasesLabelsFilenames.append(filenameAutomatedCorrectedLabelsPre)
        elif os.path.exists(filenamePre) and os.path.exists(filenameAutomatedLabelsPre): # If there is not a corrected version, use just the automated.
            # Atlas name:
            postCases.append(name)
            # Intensity image:
            postCasesImageFilenames.append(filename)
            # Atlas name:
            preCases.append(name)
            # Intensity image:
            preCasesImageFilenames.append(filenamePre)
            # Labels image:
            preCasesLabelsFilenames.append(filenameAutomatedLabelsPre)

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
# For every post case, register the pre in-phase iamge to the post-op image and transferred the labels:
for i in range(0, len(postCases)):
    # Read images:
    postImage = sitk.ReadImage(postCasesImageFilenames[i])
    preImage = sitk.ReadImage(preCasesImageFilenames[i])
    preLabels = sitk.ReadImage(preCasesLabelsFilenames[i])

    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileNonRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(postImage)
    elastixImageFilter.SetMovingImage(preImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get the images:
    pre2postImage = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    # Transfer the labels:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(preLabels)
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    pre2postLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Transfer the other sequences:
#    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "2")
#    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "float")
    # In-phase
#    postImageF = sitk.ReadImage(postCasesImageFFilenames[i])
#    transformixImageFilter.SetMovingImage(postImageF)
#    transformixImageFilter.Execute()
#    postImageF = transformixImageFilter.GetResultImage()

    # Write registered image and labels in the output directory, keeping the same filename
    sitk.WriteImage(pre2postImage, postCasesOutputhPaths[i] + postCases[i] + 'preI2pos.' + extensionImages)
    sitk.WriteImage(pre2postLabels, postCasesOutputhPaths[i] + postCases[i] + tagLabels + '.' + extensionImages)
