#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile

import SimpleITK as sitk
import numpy as np
import SitkImageManipulation as sitkExtra
import sys
import os

# Optional parameters:
#   - numSelectedAtlases: number of selected atlas after majority voting.
#   - segmentationType: segmentation type that mainly defines the similarity metric NCC and MRI
def MultiAtlasSegmentation(targetImage, softTissueMask, libraryPath, outputPath, debug, numSelectedAtlases = 5, segmentationType = 'NCC'):
    ############################### CONFIGURATION #####################################
    # Temp path:
    tempPath = outputPath + 'temp' + '\\'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    # Create a log file:
    logFilename = outputPath + 'log.txt'
    log = open(logFilename, 'w')
    ###################################################################################

    ############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
    # Parameter files:
    parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
    if segmentationType == 'NCC':
        paramFileRigid = 'Parameters_Rigid_NCC'
        paramFileBspline = 'Parameters_BSpline_NCC_4000iters_8192samples'
    elif segmentationType == 'NMI':
        paramFileRigid = 'Parameters_Rigid_NMI'
        paramFileBspline = 'Parameters_BSpline_NMI_4000iters_8192samples'
    #paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

    # Exponential gain to enhance smaller differences:
    gain = 4
    # Labels:
    numLabels = 11 # 10 for muscles and bone, and 11 for undecided
    ##########################################################################################

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # If debugging, write image:
    if debug:
        sitk.WriteImage(targetImage, outputPath + "input_registration.mhd")
    ############################################

    ############# 1) ATLAS LIBRARY ##############################
    # Look for the raw files in the library:
    files = os.listdir(libraryPath)
    extensionImages = 'mhd'
    atlasImagesNames = []
    atlasLabelsNames = []
    for filename in files:
        name, extension = os.path.splitext(filename)
    #    # Use only the marathon study
    #    if str(name).startswith("ID"):
        if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
            # Intensity image:
            atlasImagesNames.append(name + '.' + extensionImages)
            # Label image:
            atlasLabelsNames.append(name + '_labels.' + extensionImages)

    log.write("Number of atlases: {0}\n".format(len(atlasImagesNames)))
    log.write("List of files: {0}\n".format(atlasImagesNames))
    ######################################

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

    #################### 3) ATLAS SELECTION #################################
    # convert similarity value in an array:
    similarityValue = np.asarray(similarityValue)
    indicesSorted = np.argsort(similarityValue)
    similarityValuesSorted = similarityValue[indicesSorted]
    # Write similarity values, first with sitk and then with elastix that are included just for debugging purposes:
    log.write('Similarity metric values: {0}\n'.format(similarityValue))
    log.write('Similarity metric values with exponential gain: {0}\n'.format(np.power(similarityValue,2)))
    log.write('Similarity metric values from Elastix: {0}\n'.format(similarityValueElastix))
    # Selected atlases:
    indicesSelected = indicesSorted[0:numSelectedAtlases]
    log.write('Indices of selected atlases: {0}\n'.format(indicesSelected))
    ############################################

    #################### 4) LABEL PROPAGATION #######################
    propagatedLabels = []
    for i in range(0, len(indicesSelected)):
        # Read labels:
        filenameAtlas = atlasLabelsNames[indicesSelected[i]]
        labelsImage = sitk.ReadImage(libraryPath + filenameAtlas)
        nameMoving, extension = os.path.splitext(filenameAtlas)
        # Apply its transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(labelsImage)
        transformixImageFilter.SetTransformParameterMap(transformParameterMaps[indicesSelected[i]])
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        propagatedLabels.append(sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8))
        # If debugging, write label image:
        if debug:
            outputFilename = outputPath + '\\' + nameMoving + '_propagated.mhd'
            sitk.WriteImage(propagatedLabels[i], outputFilename)
    ###############################################

    ##################### 5) LABEL FUSION #################################
    outputLabels = sitk.LabelVoting(propagatedLabels, numLabels)
    multilabelStaple = sitk.MultiLabelSTAPLEImageFilter()
    multilabelStaple.SetTerminationUpdateThreshold(1e-4)
    multilabelStaple.SetMaximumNumberOfIterations(30)
    multilabelStaple.SetLabelForUndecidedPixels(numLabels)
    outputLabelsSTAPLES = multilabelStaple.Execute(propagatedLabels)
    ##############################

    ###################### 6) APPLY MASK ###############
    outputLabelsMask = sitk.Multiply(outputLabels, softTissueMask)
    outputLabelsSTAPLESmask = sitk.Multiply(outputLabelsSTAPLES, softTissueMask)

    ##################### 6) OUTPUT ############################
    # Reset the origin and direction to defaults. As I'm doing that in the plugin.
    #outputLabels.SetOrigin((0,0,0))
    #outputLabels.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    #outputLabelsSTAPLES.SetOrigin((0, 0, 0))
    #outputLabelsSTAPLES.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    #outputLabelsMask.SetOrigin((0, 0, 0))
    #outputLabelsMask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    #outputLabelsSTAPLESmask.SetOrigin((0, 0, 0))
    #outputLabelsSTAPLESmask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    # Write output:
    sitk.WriteImage(outputLabels, outputPath + "segmentedImage.mhd")
    sitk.WriteImage(outputLabelsSTAPLES, outputPath + "segmentedImageSTAPLES.mhd")
    sitk.WriteImage(outputLabelsMask, outputPath + "segmentedImageMask.mhd")
    sitk.WriteImage(outputLabelsSTAPLESmask, outputPath + "segmentedImageSTAPLESmask.mhd")
    # Dictionary with results:
    dictResults = {'segmentedImage': outputLabels, 'segmentedImageSTAPLES': outputLabelsSTAPLES,
                   'segmentedImageMask': outputLabelsMask, 'segmentedImageSTAPLESmask': outputLabelsSTAPLESmask}

    # Close log file:
    log.close()

    return dictResults
