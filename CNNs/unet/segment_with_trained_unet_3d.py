import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import writeMhd
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions

################ CONFIG ##############################
# Needs registration
preRegistration = True
registrationReferenceFilename = "/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/ID00001.mhd"
############################ DATA PATHS ##############################################
dataPath = '../../Data/LumbarSpine3D/InputImages/'
outputPath = '../../Data/LumbarSpine3D/InputImages/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/'
outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/LumbarSpineSegmentations/'
dataPath = '/home/martin/data/UNSAM/Muscle/Nepal//'
outputPath = '/home/martin/data/UNSAM/Muscle/Nepal//'
outputResampledPath = outputPath + '/Resampled/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
dataInSubdirPerSubject = True
################################### REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION #######################
referenceImageFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'
referenceImage = sitk.ReadImage(referenceImageFilename)

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
########## IMAGE FORMATS AND EXTENSION ############
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_in'#'_I'
tagAutLabels = '_aut'
tagManLabels = '_labels'

# OUTPUT PATHS:
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

if not os.path.exists(outputResampledPath):
    os.makedirs(outputResampledPath)

# MODEL
modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cuda')
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

######################### MODEL INIT ######################
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage.GetSize()))

# Look for the folders of the in phase images:
#files = os.listdir(dataPath)
if dataInSubdirPerSubject:
    files = list()
    subdirs = os.listdir(dataPath)
    for subjectDir in subdirs:
        name, extension = os.path.splitext(subjectDir)
        if os.path.isdir(dataPath + name):
            dataInSubdir = os.listdir(dataPath + name)
            for filenameInSubdir in dataInSubdir:
                nameInSubdir, extension = os.path.splitext(filenameInSubdir)
                if (extension == extensionImages and nameInSubdir.endswith(tagInPhase)
                        and not nameInSubdir.endswith(tagManLabels)):
                    files.append(dataPath + name + os.path.sep + filenameInSubdir)
else:
    files = [fn for fn in os.listdir(dataPath)
              if fn.endswith(tagInPhase + extensionImages) and not fn.endswith(tagManLabels + extensionImages)]
if preRegistration:
    refImage = sitk.ReadImage(registrationReferenceFilename)
#files = sorted(files)
imageNames = []
imageFilenames = []

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
i = 0
#files = files[0:2]
for fullFilename in files:
    fileSplit = os.path.split(fullFilename)
    pathSubject = fileSplit[0]
    filename = fileSplit[1]
    name, extension = os.path.splitext(filename)
    subject = name[:-len(tagInPhase)]
    print(subject)

    sitkImage = sitk.ReadImage(fullFilename)
    if preRegistration:
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter()
        # Register image to reference data
        elastixImageFilter.SetFixedImage(referenceImage)
        elastixImageFilter.SetMovingImage(sitkImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute()
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampled = elastixImageFilter.GetResultImage()
        # Write transformed image:
        sitk.WriteImage(sitkImageResampled, outputResampledPath + name + extensionImages, True)
        elastixImageFilter.WriteParameterFile(transform[0], 'transform.txt')
    else:
        sitkImageResampled = sitkImage
    # Convert to float and register it:
    image = sitk.GetArrayFromImage(sitkImageResampled).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    # Run the segmentation through the model:
    torch.cuda.empty_cache()
    with torch.no_grad():
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImageResampled)  # Herramienta de filtrado de imagenes

    if preRegistration:
        # Write
        sitk.WriteImage(output, outputResampledPath + subject + '_resampled_segmentation' + extension, True)

        # Resample to original space:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetInitialTransformParameterFileName('TransformParameters.0.txt')
        elastixImageFilter.SetFixedImage(sitkImageResampled) # sitkImage
        elastixImageFilter.SetMovingImage(sitkImageResampled) # sitkImage
        elastixImageFilter.LogToConsoleOff()
        # rigid_pm = affine_parameter_map()
        rigid_pm = sitk.GetDefaultParameterMap("affine")
        rigid_pm['MaximumNumberOfIterations'] = ("1000",) # By default 256, but it's not enough
        # rigid_pm["AutomaticTransformInitialization"] = "true"
        # rigid_pm["AutomaticTransformInitializationMethod"] = ["Origins"]
        elastixImageFilter.SetParameterMap(rigid_pm)
        elastixImageFilter.SetParameter('HowToCombineTransforms', 'Compose')
        elastixImageFilter.SetParameter('Metric', 'DisplacementMagnitudePenalty')

        elastixImageFilter.Execute()

        Tx = elastixImageFilter.GetTransformParameterMap()
        Tx[0]['InitialTransformParametersFileName'] = ('NoInitialTransform',)
        Tx[0]['Origin'] = tuple(map(str, sitkImage.GetOrigin()))
        Tx[0]['Spacing'] = tuple(map(str, sitkImage.GetSpacing()))
        Tx[0]['Size'] = tuple(map(str, sitkImage.GetSize()))
        Tx[0]['Direction'] = tuple(map(str, sitkImage.GetDirection()))
        #resample = sitk.ResampleImageFilter()
        #resample.SetReferenceImage(sitkImage)
        #resampledMovingIm = resample.Execute(sitkImageResampled)
        #resampleTx = resample.GetTransform()

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        #transformixImageFilter.AddTransformParameterMap(resampleTx)
        transformixImageFilter.SetMovingImage(output)
        transformixImageFilter.SetLogToConsole(False)
        # transformixImageFilter.SetFixedImage(output)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)


    sitk.WriteImage(output, outputPath + subject + '_segmentation' + extensionImages, True)



        # # Downsample it:
        # original_spacing = referenceImage.GetSpacing()
        # original_size = referenceImage.GetSize()
        # origin = referenceImage.GetOrigin()
        # direction = referenceImage.GetDirection()
        # new_spacing = [spc * 2 for spc in original_spacing]
        # new_spacing[2] = original_spacing[2]
        # new_size = [int(sz / 2) for sz in original_size]
        # new_size[2] = original_size[2]
        # resampler = sitk.ResampleImageFilter()
        # resampler.SetSize(new_size)
        # resampler.SetOutputSpacing(new_spacing)
        # resampler.SetOutputOrigin(origin)
        # referenceImage = resampler.Execute(referenceImage)
        # sitk.WriteImage(referenceImage, outputPath + 'referenceResampled.mhd')




