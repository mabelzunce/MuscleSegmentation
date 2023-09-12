import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from unet_2d import Unet
import torch
from utils import writeMhd
from utils import filtered_multilabel
from utils import maxProb

############################ DATA PATHS ##############################################
dataPath = '../../Data/LumbarSpine2D/TestSubjects/'
outputPath = '../../Data/LumbarSpine2D/model/'
modelLocation = '../../Data/LumbarSpine2D/PretrainedModel/'
# Image format extension:
extensionImages = 'mhd'

modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn.
# We will get it from the training set.
# imageSize_voxels = (256,256)

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

######################### MODEL INIT ######################
multilabelNum = 6
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)
###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
folder = os.listdir(dataPath)
folder = sorted(folder)
imageNames = []
imageFilenames = []
i = 0
for files in folder:
    name, extension = os.path.splitext(files)
    # Check if filename is the in phase header and the labels exists:
    if extension.endswith('raw'):
        continue
    filenameImage = dataPath + files
    sitkImage = sitk.ReadImage(filenameImage)
    image = sitk.GetArrayFromImage(sitkImage).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    with torch.no_grad():
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        output = maxProb(output, multilabelNum)
        output = filtered_multilabel((output > 0.5) * 1)
        output = output.squeeze(0)
    output = sitk.GetImageFromArray(output)
    output.CopyInformation(sitkImage)
    sitk.WriteImage(sitk.Cast(output,sitk.sitkUInt8), outputPath + name.split('_')[0] + '_seg' + extension)




