#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile

import SimpleITK as sitk
import numpy as np
import sys
import os

def MultiAtlasSegmentation(targetImage, softTissueMask, libraryPath, outputPath, debug):
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
    paramFileRigid = 'Parameters_Rigid'
    paramFileBspline = 'Parameters_BSpline_NCC'
    #paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

    # Number of Atlases to select:
    numSelectedAtlases = 5

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
        logFilename = 'reg_log_{0}'.format(i) + '.txt' # iT DOESN'T WORK WITH DIFFERENT LOG NAMES
        #logFilename = 'reg_log' + '.txt'
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
        #similarityValue.append(GetFinalMetricFromElastixLogFile(fullLogFilename))
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
    log.write('Similarity metric values: {0}\n'.format(similarityValue))
    # Selected atlases:
    indicesSelected = indicesSorted[0:numSelectedAtlases-1]
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

    ##################### 6) OUTPUT ############################
    # Write output:
    sitk.WriteImage(outputLabels, outputPath + "segmentedImage.mhd")
    sitk.WriteImage(outputLabelsSTAPLES, outputPath + "segmentedImageSTAPLES.mhd")
    dictResults = {'segmentedImage': outputLabels, 'segmentedImageSTAPLES': outputLabelsSTAPLES}

    outputLabels = sitk.Multiply(outputLabels, softTissueMask)
    outputLabelsSTAPLES = sitk.Multiply(outputLabelsSTAPLES, softTissueMask)
    # Write output:
    sitk.WriteImage(outputLabels, outputPath + "segmentedImageMask.mhd")
    sitk.WriteImage(outputLabelsSTAPLES, outputPath + "segmentedImageSTAPLESmask.mhd")
    # Close log file:
    log.close()

    return dictResults