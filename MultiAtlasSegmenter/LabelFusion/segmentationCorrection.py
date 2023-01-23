import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
sys.path.append('..\\..\\CNNs\\unet')
from utils import loss_csv
from utils import imshow_from_torch
from utils import dice
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from sklearn.model_selection import train_test_split
from datetime import datetime
sys.path.append('..\\..\\CNNs\\unet')
from unet_2d import Unet
#from utils import imshow
#from utils import MSE
import torch
import torch.nn as nn
import torchvision
from utils import imshow_from_torch

from torchvision.utils import make_grid

############################ DATA PATHS ##############################################
trainingSetPath = 'C:\\Users\\ecyt\\Documents\\Flor Sarmiento\\CNN correction\\MuscleSegmentation\\Data\\Pelvis3D\\FusionTrainingSet\\'   #'..\\..\\Data" tengo q cambiar por training set
outputPath = 'C:\\Users\\ecyt\\Documents\\Flor Sarmiento\\CNN correction\\MuscleSegmentation\\MultiAtlasSegmenter\\LabelFusion\\Output Path Segmentation Correction'

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagManualLabels = '_manual'
tagAutLabels = '_aut'
############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn: we will get it from the training set.
# imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.8
devSetRelSize = trainingSetRelSize-0.2

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #############################################

# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasManLabelsFilenames = [] # Filenames of the manual label images
atlasAutLabelsFilenames = [] # Filenames of the automatic label images
#imagesDataSet = np.empty()
#labelsDataSet = np.empty()

i = 0
for filename in files:
    filenameImages = trainingSetPath + filename
    if len(filename)==11 and filename.endswith('.mhd'):   #Entra solo en los archivos "numImage.mhd"

        name, extension = os.path.splitext(filename)

        filenameImages = trainingSetPath + filename
        filenameManLabels = trainingSetPath + name + tagManualLabels + '.' + extensionImages
        filenameAutLabels = trainingSetPath + name + tagAutLabels + '.' + extensionImages
        #if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        # Atlas name:
        atlasNames.append(filename)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels:
        atlasManLabelsFilenames.append(filenameManLabels)
        # Automatic Labels:
        atlasAutLabelsFilenames.append(filenameAutLabels)

# Initialize numpy array and read data:
numImages = len(atlasImageFilenames)
for i in range(0, numImages):
    # Read images and add them in a numpy array:
    print('Reading image: {0}'.format(atlasImageFilenames[i]))
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasManLabels = sitk.ReadImage(atlasManLabelsFilenames[i])
    atlasAutLabels = sitk.ReadImage(atlasAutLabelsFilenames[i])

    numSlices = numImages*atlasImage.GetSize()[2]
    if i == 0:              #Arma un array de ceros con la cantidad total de slices y usando la primera imagen de referencia para el tamaño de cada slice
        imagesDataSet = np.zeros([numSlices,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        manLabelsDataSet = np.zeros([numSlices,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        autLabelsDataSet = np.zeros([numSlices, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesDataSet.shape[1:3]

    imagen = sitk.GetArrayFromImage(atlasImage)
    imagesDataSet[i*atlasImage.GetSize()[2]:(i+1)*atlasImage.GetSize()[2],:,:] = sitk.GetArrayFromImage(atlasImage)
    #np.reshape(sitk.GetArrayFromImage(atlasImage), [1,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
    manLabelsDataSet[i*atlasImage.GetSize()[2]:(i+1)*atlasImage.GetSize()[2],:,:] = sitk.GetArrayFromImage(atlasManLabels)
    autLabelsDataSet[i*atlasImage.GetSize()[2]:(i+1)*atlasImage.GetSize()[2],:,:] = sitk.GetArrayFromImage(atlasAutLabels)
    i = i + 1

numberOfCases = len(atlasNames)
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Iterate and plot random images:
numImagesToShow = 2
#numImagesToShow = numSlices # Show all images
cols = numImagesToShow
rows = 2
indicesImages = np.random.choice(numImages, numImagesToShow, replace=False)
#plt.figure(figsize=(15, 10))
for i in range(numImagesToShow):
    plt.subplot(rows, cols, i + 1)
    #overlay = sitk.LabelOverlay(imagesDataSet[i,:,:],autLabelsDataSet[i,:,:],opacity=0.5, backgroundValue=0)
    #plt.imshow(overlay)
    plt.imshow(imagesDataSet[36+i*atlasImage.GetSize()[2], :, :], cmap='gray', vmin=0, vmax=0.5*np.max(imagesDataSet[36+i*atlasImage.GetSize()[2], :, :]))
    autLabelsDataSet = autLabelsDataSet.astype(int)
    plt.imshow(autLabelsDataSet[36+i*atlasImage.GetSize()[2], :, :], cmap='hot', alpha=0.3)
    plt.subplot(rows, cols, i + 3)
    plt.imshow(imagesDataSet[36 + i * atlasImage.GetSize()[2], :, :], cmap='gray', vmin=0, vmax=0.5 * np.max(imagesDataSet[36 + i * atlasImage.GetSize()[2], :, :]))
    manLabelsDataSet = manLabelsDataSet.astype(int)
    plt.imshow(manLabelsDataSet[36+i*atlasImage.GetSize()[2], :, :], cmap='hot',  alpha=0.3)

plt.savefig(outputPath + '\\dataSet.png')

#mask3 = np.any(imagesDataSet, axis=1)

# Add the channel dimension for compatibility: (1 solo canal)
imagesDataSet = np.expand_dims(imagesDataSet, axis=1)
autLabelsDataSet = np.expand_dims(autLabelsDataSet, axis=1)
manLabelsDataSet = np.expand_dims(manLabelsDataSet, axis=1)

# Cast to float (the model expects a float) and leave only one label to correct:
imagesDataSet = imagesDataSet.astype(np.float32)
autLabelsDataSet = autLabelsDataSet.astype(np.float32)
autLabelsDataSet[autLabelsDataSet!=4] = 0  #dejar solo el label del TFL
autLabelsDataSet[autLabelsDataSet==4] = 1
manLabelsDataSet = manLabelsDataSet.astype(np.float32)
manLabelsDataSet[manLabelsDataSet!=4] = 0
manLabelsDataSet[manLabelsDataSet==4] = 1

######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################

#Eliminar las slices vacias (tomando de referencia la segmentacion automatica)

mask = np.any(autLabelsDataSet, axis=(1,2,3))
mask1 = np.any(manLabelsDataSet, axis=(1,2,3))
slicesToDelete = []
for i in range (numSlices):
    if mask[i]==False:
        slicesToDelete.append(i)

autLabelsDataSet = np.delete(autLabelsDataSet, slicesToDelete, axis=0)
manLabelsDataSet = np.delete(manLabelsDataSet, slicesToDelete, axis=0)
imagesDataSet = np.delete(imagesDataSet, slicesToDelete, axis=0)
mask2 = np.any(autLabelsDataSet, axis=(1,2,3))
mask3 = np.any(manLabelsDataSet, axis=(1,2,3))

# Get the number of images for the training and test data sets:
sizeFullDataSet = int(imagesDataSet.shape[0])
numberOfImagesPerCase = sizeFullDataSet / numberOfCases
numberOfCasesForTrainingSet = round(numberOfCases*0.8)
sizeTrainingSet = numberOfCasesForTrainingSet * numberOfImagesPerCase
indicesTrainingSet = range(0,int(sizeTrainingSet))
sizeDevSet = sizeFullDataSet-sizeTrainingSet
indicesDevSet = range(int(sizeTrainingSet), sizeFullDataSet)

inputDataSet = np.concatenate((imagesDataSet, autLabelsDataSet), axis=1)
# Create dictionaries with training sets:
trainingSet = dict([('input',inputDataSet[indicesTrainingSet,:,:,:]), ('output', manLabelsDataSet[indicesTrainingSet,:,:,:])])   #acá deberia agregar en imput el aut
devSet = dict([('input',inputDataSet[indicesDevSet,:,:,:]), ('output', manLabelsDataSet[indicesDevSet,:,:,:])])

inp = trainingSet['input']
out = trainingSet['output']

#Save input and output images

imageFromArray = sitk.GetImageFromArray(inp[:,0,:,:])
autLabelsFromArray = sitk.GetImageFromArray(inp[:, 1, :, :])
manLabelsFromArray = sitk.GetImageFromArray(np.squeeze(out))
sitk.WriteImage(imageFromArray, outputPath + '\\CheckOutputInputImages\\imageAllSlices.mhd')
sitk.WriteImage(autLabelsFromArray, outputPath + '\\CheckOutputInputImages\\autLabelsAllSlices.mhd')
sitk.WriteImage(manLabelsFromArray, outputPath + '\\CheckOutputInputImages\\manLabelsAllSlices.mhd')

print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(2,1)

##
print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape

#ESCRIBIR EN MHD O ALGO LAS IMAGENES PARA CHEQUEAR
numSlices = int(inp.shape[0])
imagesToSave = random.sample(range(0, numSlices), 10)
for i in imagesToSave:
    imageFromArray = sitk.GetImageFromArray(inp[i,0,:,:])
    autLabelsFromArray = sitk.GetImageFromArray(inp[i, 1, :, :])
    manLabelsFromArray = sitk.GetImageFromArray(out[i, :, :, :])
    sitk.WriteImage(imageFromArray, outputPath + '\\CheckOutputInputImages\\imageSlice' + str(i) + '.mhd')
    sitk.WriteImage(autLabelsFromArray, outputPath + '\\CheckOutputInputImages\\autLabelsSlice' + str(i) + '.mhd')
    sitk.WriteImage(manLabelsFromArray, outputPath + '\\CheckOutputInputImages\\manLabelsSlice' + str(i) + '.mhd')

##################################### U-NET TRAINING ############################################
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()                      #establece el criterio con el que mide el error que usa para optimizar
optimizer = optim.Adam(unet.parameters(), lr=0.0001)    #establece el metodo para optimizar

# Number of  batches:
batchSize = 4                       #pruebo en ppio con batches de size 4
numBatches = np.round(trainingSet['input'].shape[0]/batchSize).astype(int)
devNumBatches = np.round(devSet['input'].shape[0]/batchSize).astype(int)
# Show results every printStep batches:
plotStep_epochs = 1
numImagesPerRow = batchSize
# Show results every printStep batches:
printStep = 1
figImages, axs = plt.subplots(3, 1,figsize=(20,20))
figLoss, axLoss = plt.subplots(1, 1,figsize=(5,5))
# Show dev set loss every showDevLossStep batches:
showDevLossStep = 4
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Train
best_vloss = 1000
lossValuesTrainingSet = []
lossValuesTrainingSetPerEpoch = []    #valor medio de los loss value de cada epoca
lossValuesDevSetEpoch = []
iterationNumbers = []
lossValuesDevSet = []
iterationNumbersForDevSet = []
lossValuesDevSetAllEpoch = []

iter = 0
devIter = 0

for epoch in range(100):  # loop over the dataset multiple times
    lossValuesTrainingSetThisEpoch = []
    iterationNumbersForDevSetThisEpoch = []
    print('epoch: {0}'.format(epoch))
    unet.train(True)
    for i in range(numBatches):
        print('Batch: {0}'.format(i))
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:])
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:])
        #imshow_from_torch(torchvision.utils.make_grid(inputs, normalize=True))
        #imshow_from_torch(torchvision.utils.make_grid(gt, normalize=True))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()

        # print statistics
        print('Batch loss: {0}'.format(loss.item()))
        # Save loss values:
        lossValuesTrainingSet.append(loss.item())
        lossValuesTrainingSetThisEpoch.append(loss.item())
        iterationNumbers.append(iter)
        loss_csv(lossValuesTrainingSet, outputPath + '\\Loss_training_iter.csv')

        # Evaluate dev set if it's the turn to do it:
        #if i % showDevLossStep == (showDevLossStep-1):
        #    outputsDevSet = unet(inputsDevSet)
        #    lossDevSet = criterion(outputsDevSet, gtDevSet)
        #    lossValuesDevSet.append(lossDevSet.item())
        #    iterationNumbersForDevSet.append(iter)

        # Update iteration number:
        iter = iter + 1

    lossValuesTrainingSetPerEpoch.append(np.mean(lossValuesTrainingSetThisEpoch))

    unet.train(False)
    for i in range(devNumBatches):
        # get the inputs
        inputs = torch.from_numpy(devSet['input'][i * batchSize:(i + 1) * batchSize, :, :, :])
        gt = torch.from_numpy(devSet['output'][i * batchSize:(i + 1) * batchSize, :, :, :])
        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        #iterationNumbersForDevSet.append(deviter)
        lossValuesDevSet.append(loss.item())
        lossValuesDevSetEpoch.append(loss.item())
        devIter = devIter + 1
        loss_csv(lossValuesDevSet, outputPath + '\\Loss_valid_iter.csv')
        print('lossValuesDevSet: {0}'.format(lossValuesDevSet))

    avg_vloss = np.mean(lossValuesDevSetEpoch)
    lossValuesDevSetAllEpoch.append(avg_vloss)
    loss_csv(lossValuesDevSetAllEpoch, outputPath + '\\ValidDataEpoch.csv')

    # Show data it printStep
    if i % printStep == (printStep - 1):  # print every printStep mini-batches
        print('%d, loss: %.3f' %
              (epoch + 1, np.mean(lossValuesTrainingSetThisEpoch)))
        running_loss = 0.0

        # Show input images:
        plt.figure(figImages)
        plt.axes(axs[0])
        inputs1 = inputs[0,0,:,:]
        inputs2 = inputs[0,1,:,:]
        imshow_from_torch(torchvision.utils.make_grid((inputs1), normalize=True))
        imshow_from_torch(torchvision.utils.make_grid((inputs2), normalize=True))
        axs[0].set_title('Input Batch {0}'.format(i))
        plt.axes(axs[1])
        plt.show

        # outputs = outputs.squeeze()
        outputsLabels = torch.sigmoid(outputs)
        outputsLabels = (outputsLabels > 0.5) * 255

        # filter out the weak predictions and convert them to integers
        # outputsLabels = outputsLabels.to(torch.uint8)
        #imshow_from_torch(torchvision.utils.make_grid(inputs, normalize=True))         #
        #imshow_from_torch(torchvision.utils.make_grid(outputsLabels.to(torch.float), normalize=True), ialpha=0.5, icmap='hot')    #
        #axs[1].set_title('Output Epoch {0}'.format(epoch))      #
        #plt.axes(axs[2])                #
        #imshow_from_torch(torchvision.utils.make_grid(gt, normalize=True))          #
        #axs[2].set_title('Ground Truth')        #

        # Show loss:
        plt.figure(figLoss)
        axLoss.plot(iterationNumbers, lossValuesTrainingSet)
        plt.title('Loss value training Set - Epoch {0}'.format(epoch))
        #axLoss.plot(iterationNumbersForDevSet, lossValuesDevSet)
        plt.draw()
        plt.pause(0.0001)

    #Save one image per epoch
    outputsLabels = torch.sigmoid(outputs)
    outputsLabels = (outputsLabels > 0.5) * 255

    imageFromArray = sitk.GetImageFromArray(inputs[:, 0, :, :])
    autLabelsFromArray = sitk.GetImageFromArray(inputs[:, 1, :, :])
    outLabelsFromArray = sitk.GetImageFromArray(outputsLabels.detach().numpy())
    outputFromArray = sitk.GetImageFromArray(outputs.detach().numpy())
    sitk.WriteImage(imageFromArray, outputPath + '\\save_one_image_per_epoch\\' + str(epoch) + 'image.mhd')
    sitk.WriteImage(autLabelsFromArray, outputPath + '\\save_one_image_per_epoch\\' + str(epoch) + 'aut.mhd')
    sitk.WriteImage(outLabelsFromArray, outputPath + '\\save_one_image_per_epoch\\' + str(epoch) + 'man.mhd')
    sitk.WriteImage(outputFromArray, outputPath + '\\save_one_image_per_epoch\\' + str(epoch) + 'man.mhd')

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print('[validation Epoch: %d] best_vloss: %.3f' % (epoch, best_vloss))
        modelPath = outputPath + '\\unet' + '_{}_{}_best_fit'.format(timestamp, epoch) + '.pt'
        torch.save(unet.state_dict(), modelPath)

print('Finished Training')
torch.save(unet.state_dict(), outputPath + '\\unet.pt')
torch.save(unet, outputPath + '\\unetFullModel.pt')

# See weights:
#kernels = kernels - kernels.min()
#kernels = kernels / kernels.max()
#img = make_grid(kernels)
#plt.imshow(img.permute(1, 2, 0))