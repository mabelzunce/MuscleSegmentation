import nibabel as nb
import SimpleITK as sitk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime
from utils import loss_csv
from random import randint

from utils import imshow_from_torch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader


#from sklearn.model_selection import train_test_split


from unet_2d import Unet
#from utils import imshow
#from utils import MSE
import torch
import torch.nn as nn
import torchvision

from torchvision.utils import make_grid

############################ DATA PATHS ##############################################
trainingSetPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSet\\'
outputPath = '..\\..\\Data\\LumbarSpine2D\\model\\'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagLabels = '_labels'
############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn.
# We will get it from the training set.
# imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.75
devSetRelSize = trainingSetRelSize-0.25

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images

#imagesDataSet = np.empty()
#labelsDataSet = np.empty()
i = 0
for filename in files:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name
    # Check if filename is the in phase header and the labels exists:
    filenameImages = trainingSetPath + filename
    filenameLabels = trainingSetPath + atlasName + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Labels image:
        atlasLabelsFilenames.append(filenameLabels)

# Initialize numpy array and read data:
numImages = 12 #len(atlasImageFilenames)
rng = np.random.default_rng()
rInts = rng.choice(4000, size=numImages, replace=False)
for i in range(0, numImages):
    # Read images and add them in a numpy array:
    atlasImage = sitk.ReadImage(atlasImageFilenames[rInts[i]])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[rInts[i]])
    if i == 0:
        imagesDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        labelsDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesDataSet.shape[1:3]              #obtiene el getsize[1 y 0]
    imagesDataSet[i, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
    labelsDataSet[i, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Iterate and plot random images:
numImagesToShow = numImages # Show all images
cols = 6
rows = int(np.ceil(numImagesToShow/cols))
indicesImages = np.random.choice(numImages, numImagesToShow, replace=False)

# Add the channel dimension for compatibility:
imagesDataSet = np.expand_dims(imagesDataSet, axis=1)
labelsDataSet = np.expand_dims(labelsDataSet, axis=1)
# Cast to float (the model expects a float):
imagesDataSet = imagesDataSet.astype(np.float32)
labelsDataSet = labelsDataSet.astype(np.float32)
labelsDataSet[labelsDataSet != 1] = 0
labelsDataSet[labelsDataSet == 1] = 1
######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
# Get the number of images for the training and test data sets:
sizeFullDataSet = int(imagesDataSet.shape[0])
sizeTrainingSet = int(np.round(sizeFullDataSet*trainingSetRelSize))
sizeDevSet = sizeFullDataSet-sizeTrainingSet
# Get random indices for the training set:
rng = np.random.default_rng()
#indicesTrainingSet = rng.choice(int(sizeFullDataSet), int(sizeTrainingSet), replace=False)
#indicesDevSet = np.delete(range(sizeFullDataSet), indicesTrainingSet)
indicesTrainingSet = range(0,int(sizeTrainingSet))
indicesDevSet = range(int(sizeTrainingSet), sizeFullDataSet)
# Create dictionaries with training sets:
trainingSet = dict([('input', imagesDataSet[indicesTrainingSet, :, :, :]), ('output', labelsDataSet[indicesTrainingSet, :, :, :])])
devSet = dict([('input', imagesDataSet[indicesDevSet,:,:,:]), ('output', labelsDataSet[indicesDevSet,:,:,:])])



print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(1, 1)
inp = torch.rand(1, 1, dataSetImageSize_voxels[0], dataSetImageSize_voxels[1])
out = unet(inp)

##
print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape

##################################### U-NET TRAINING ############################################
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

# Number of  batches:
batchSize = 3
devBatchSize = 1
numBatches = np.round(trainingSet['input'].shape[0]/batchSize).astype(int)
devNumBatches = np.round(devSet['input'].shape[0]/devBatchSize).astype(int)
# Show results every printStep batches:
printStep = 1
figImages, axs = plt.subplots(3, 1, figsize=(20, 20))
figLoss, axLoss = plt.subplots(1, 1, figsize=(5, 5))
# Show dev set loss every showDevLossStep batches:
#showDevLossStep = 1
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
best_vLoss = 1000

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

iterationNumbers = []
iterationDevNumbers = []
epochNumbers = []

lossValuesTrainingSet = []
lossValuesDevSet = []

lossValuesTrainingSetAllEpoch = []
lossValuesDevSetAllEpoch = []

iter = 0
devIter = 0


scaler = torch.cuda.amp.GradScaler()


unet.to(device)
for epoch in range(50):  # loop over the dataset multiple times
    epochNumbers.append(epoch)

    lossValuesTrainingSetEpoch = []
    lossValuesDevSetEpoch = []

    unet.train(True)
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = unet(inputs)

            loss = criterion(outputs, gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # print statistics
        # Save loss values:
        lossValuesTrainingSet.append(loss.item())
        lossValuesTrainingSetEpoch.append(loss.item())
        iterationNumbers.append(iter)
        #Print epoch iteration and loss value:
        print('[%d, %5d] loss: %.3f' % (epoch, i, loss.item()))
        # Update iteration number:
        iter = iter + 1
    lossValuesTrainingSetAllEpoch.append(np.mean(lossValuesTrainingSetEpoch))
    loss_csv(lossValuesTrainingSetAllEpoch, outputPath + 'TestDataEpoch.csv')
    loss_csv(lossValuesTrainingSet, outputPath + 'TestDataIter.csv')


    unet.train(False)
    r = randint(0, len(devSet))
    for i in range(devNumBatches):
        inputs = torch.from_numpy(devSet['input'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
        gt = torch.from_numpy(devSet['output'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)

        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        lossValuesDevSet.append(loss.item())
        lossValuesDevSetEpoch.append(loss.item())

        iterationDevNumbers.append(devIter)
        devIter = devIter + 1
        if i == r:
            auxInputs = inputs.cpu()
            auxGT = gt.cpu()
            auxOutputs = outputs.cpu()

    avg_vLoss = np.mean(lossValuesDevSetEpoch)
    lossValuesDevSetAllEpoch.append(avg_vLoss)
    loss_csv(lossValuesDevSetAllEpoch, outputPath + 'ValidDataEpoch.csv')
    loss_csv(lossValuesDevSet, outputPath + 'ValidDataIter.csv')

    if avg_vLoss < best_vLoss:
        best_vLoss = avg_vLoss
        print('[validation Epoch: %d] best_vLoss: %.3f' % (epoch, best_vLoss))
        modelPath = outputPath + 'unet' + '_{}_{}_best_fit'.format(timestamp, epoch) + '.pt'
        torch.save(unet.state_dict(), modelPath)

    if epoch % printStep == (printStep - 1) and epoch > 4:  # print every printStep mini-batches
        # Show input images:
        plt.figure(figImages)
        plt.axes(axs[0])
        imshow_from_torch(torchvision.utils.make_grid(auxInputs, normalize=True))
        axs[0].set_title('Input dev Batch {0}'.format(r))
        plt.axes(axs[1])
        outputsLabels = torch.sigmoid(auxOutputs)
        outputsLabels = (outputsLabels > 0.5) * 255
        # filter out the weak predictions and convert them to integers
        # outputsLabels = outputsLabels.to(torch.uint8)
        imshow_from_torch(torchvision.utils.make_grid(auxInputs, normalize=True))
        imshow_from_torch(torchvision.utils.make_grid(outputsLabels.to(torch.float), normalize=True), ialpha=0.5,
                          icmap='hot')
        axs[1].set_title('Output dev Epoch {0}'.format(epoch))
        plt.axes(axs[2])
        imshow_from_torch(torchvision.utils.make_grid(auxGT, normalize=True))
        axs[2].set_title('Ground Truth')
        plt.savefig(outputPath + 'images' + '_{}_{}_best_fit'.format(timestamp, epoch))
        # Show loss:
        plt.figure(figLoss)

        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        axLoss.plot(epochNumbers[epoch-5:], lossValuesTrainingSetAllEpoch[epoch-5:], label='TrainingSet')
        axLoss.plot(epochNumbers[epoch-5:], lossValuesDevSetAllEpoch[epoch-5:], label='Devset')
        plt.legend(loc='upper left')
        plt.draw()

        plt.savefig(outputPath + 'loss')




print('Finished Training')
torch.save(unet.state_dict(), outputPath + 'unet.pt')
torch.save(unet, outputPath + 'unetFullModel.pt')

