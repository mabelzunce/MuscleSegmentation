import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from unet_2d import Unet
import torch
from utils import writeMhd
from utils import maxProb
from utils import filtered_multilabel

############################ DATA PATHS ##############################################
dataPath = '..\\..\\Data\\LumbarSpine2D\\TestSubjects\\'
outputPath = '..\\..\\Data\\LumbarSpine2D\\model\\'
modelLocation = '..\\..\\Data\\LumbarSpine2D\\PretrainedModel\\'
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
###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(dataPath)
imageNames = []
imageFilenames = []
i = 0
for filename in files:
    name, extension = os.path.splitext(filename)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + filename
    if extension.endswith(extensionImages):
        # Atlas name:
        imageNames.append(name)
        # Intensity image:
        imageFilenames.append(filenameImages)


# Initialize numpy array and read data:
numImages = len(imageFilenames)

for i in range(0, numImages):
    # Read images and add them in a numpy array:
    image = sitk.ReadImage(imageFilenames[i])
    if i == 0:
        imagesDataSet = np.zeros([numImages, image.GetSize()[1], image.GetSize()[0]], 'float32')
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesDataSet.shape[1:3]
    imagesDataSet[i, :, :] = np.reshape(sitk.GetArrayFromImage(image).astype(np.float32), [1, image.GetSize()[1], image.GetSize()[0]])
    i = i + 1

writeMhd(imagesDataSet.astype(np.float32), outputPath + 'Images.mhd')
print("Number of images: {0}".format(len(imageNames)))
print("List of images: {0}\n".format(imageNames))

outputSegmentation = np.zeros(np.shape(imagesDataSet))
imagesDataSet = np.expand_dims(imagesDataSet, axis=1)
multilabelNum = 6
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)


####################### LOAD MODEL ########################
with torch.no_grad():
    ####################### RUN UNET ##########################
    for i, image in enumerate(imagesDataSet):
        inputs = torch.from_numpy(image).to(device)
        outputs = model(inputs.unsqueeze(0))
        outputs = torch.sigmoid(outputs.cpu().to(torch.float32))
        outputs = (outputs > 0.5) * 1
        outputSegmentation[i] = filtered_multilabel(outputs.detach().numpy(), multilabelNum, False)

writeMhd(outputSegmentation.astype(np.float32), outputPath + 'segmentedImages.mhd')

