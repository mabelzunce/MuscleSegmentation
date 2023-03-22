import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import os
import math
from datetime import datetime
from utils import create_csv
from utils import imshow_from_torch
from utils import dice2d
from utils import specificity
from utils import sensitivity
from utils import maxProb
from utils import writeMhd
from utils import multilabel
from utils import boxplot
from utils import DiceLoss
from utils import pn_weights
from utils import rel_weights
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

import torch.nn.functional as F
from torchvision.utils import make_grid

class Augment(enumerate):
    NA = 0  # No Augment
    NL = 1  # Non linear
    L = 2  # Linear
    A = 3   #Augmented


saveMhd = True        # Saves a mhd file for the output
saveDataSetMhd = False  # Saves a Mhd file of the images and labels from dataset
Background = False        # Background is considered as label
Boxplot = True           # Boxplot created in every best fit
AugmentedTrainingSet = Augment.NA
# Para correr la prueba corroborar que cantidad de filtros  establecidos  en "unet_2d" es igual a los del modelo

############################ DATA PATHS ##############################################
trainingSetPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSet\\'
outputPath = '..\\..\\Data\\LumbarSpine2D\\model\\'
modelLocation = '..\\..\\Data\\LumbarSpine2D\\PretrainedModel\\'


modelName = os.listdir(modelLocation)[0]
unetFilename = modelLocation + modelName

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
trainingSetRelSize = 0.7
devSetRelSize = 1-trainingSetRelSize

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images

numImagesPerSubject = 0
numRot = 0
datasize = 0
numSubjects = 1
subject = files[0].split('_')[0]
subjects = []
subjects.append(subject)
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

        # Data Set info
        subjectName = filename.split('_')[0]
        subjectAdd = filename.split('_')[1]
        if subject == subjectName:
            numImagesPerSubject += 1
            if not (subjectName.startswith(subjectAdd[0])):
                numRot += 1
        if not (subjects.__contains__(subjectName)):
            numSubjects += 1
            subjects.append(subjectName)
        datasize += 1

tNum = int(np.floor(numSubjects * trainingSetRelSize))
vNum = int(np.ceil(numSubjects * devSetRelSize))
trainingSubjects = subjects[:tNum]
validSubjects = subjects[tNum:]
match AugmentedTrainingSet:
    case 1:
        tNum = (tNum * tNum)
    case 2:
        tNum = (numRot * tNum)
    case 3:
        tNum = (tNum + numRot) * tNum

k = 0
j = 0
# Data set avoids mixing same subject images
for i in range(0, datasize):
    name1, name2 = atlasNames[i].split('_')[:2]
    match AugmentedTrainingSet:
        case 0:
            condition1 = (trainingSubjects.__contains__(name1)) and (name2 == name1)
        case 1:
            condition1 = (trainingSubjects.__contains__(name1)) and (trainingSubjects.__contains__(name2))
        case 2:
            conditionAux = trainingSubjects.__contains__(name2) or validSubjects.__contains__(name2)
            condition1 = ((trainingSubjects.__contains__(name1)) and not conditionAux)
        case 3:
            condition1 = (trainingSubjects.__contains__(name1)) and (not (validSubjects.__contains__(name2)))

    condition2 = (validSubjects.__contains__(name1)) and (name2 == name1)

    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])

    trainingSetShape = [tNum, atlasImage.GetSize()[1], atlasImage.GetSize()[0]]
    validSetShape = [vNum, atlasImage.GetSize()[1], atlasImage.GetSize()[0]]

    if i == 0:
        imagesTrainingSet = np.zeros(trainingSetShape)
        labelsTrainingSet = np.zeros(trainingSetShape)
        imagesValidSet = np.zeros(validSetShape)
        labelsValidSet = np.zeros(validSetShape)
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesTrainingSet.shape[1:3]  # obtiene el getsize[1 y 0]

    if condition1:
        imagesTrainingSet[k, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsTrainingSet[k, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        k += 1

    if condition2:
        imagesValidSet[j, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsValidSet[j, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        j += 1
if saveDataSetMhd:
    writeMhd(imagesTrainingSet.astype(np.float32), outputPath + 'images_training_set.mhd')
    writeMhd(labelsTrainingSet.astype(np.uint8), outputPath + 'labels_training_set.mhd')
    writeMhd(imagesValidSet.astype(np.float32), outputPath + 'images_valid_set.mhd')
    writeMhd(labelsValidSet.astype(np.uint8), outputPath + 'labels_valid_set.mhd')
# Initialize numpy array and read data:
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


# Add the channel dimension for compatibility:
imagesTrainingSet = np.expand_dims(imagesTrainingSet, axis=1)
labelsTrainingSet = np.expand_dims(labelsTrainingSet, axis=1)

imagesValidSet = np.expand_dims(imagesValidSet, axis=1)
labelsValidSet = np.expand_dims(labelsValidSet, axis=1)

# Cast to float (the model expects a float):
imagesTrainingSet = imagesTrainingSet.astype(np.float32)
labelsTrainingSet = labelsTrainingSet.astype(np.float32)


imagesValidSet = imagesValidSet.astype(np.float32)
labelsValidSet = labelsValidSet.astype(np.float32)


######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
trainingSet = dict([('input', imagesTrainingSet[:, :, :, :]), ('output', labelsTrainingSet[:, :, :, :])])
devSet = dict([('input', imagesValidSet[:, :, :, :]), ('output', labelsValidSet[:,:,:,:])])
print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))
labelNames = ('Background', 'Left Multifidus', 'Right Multifidus', 'Left Quadratus', 'Right Quadratus', 'Left Psoas', 'Right Psoas')
####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and multiple output canal.
multilabelNum = 6
if Background:
    multilabelNum += 1
    xLabel = ['BG', 'LM', 'RM', 'LQ', 'RQ', 'LP', 'RP']
    pos_weights = rel_weights(labelsTrainingSet, multilabelNum)
    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
else:
    xLabel = ['LM', 'RM', 'LQ', 'RQ', 'LP', 'RP']
    criterion = nn.BCEWithLogitsLoss()

unet = Unet(1, multilabelNum)
unet.load_state_dict(torch.load(unetFilename, map_location=device))

##
print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape
##################################### U-NET TRAINING ############################################
# Number of  batches:
batchSize = 4
devBatchSize = 4
numBatches = np.ceil(trainingSet['input'].shape[0]/batchSize).astype(int)
devNumBatches = np.ceil(devSet['input'].shape[0]/devBatchSize).astype(int)
# Show results every printStep batches:
plotStep_epochs = 1
numImagesPerRow = batchSize
if plotStep_epochs != math.inf:
    figGraphs, axs_graphs = plt.subplots(1, 8, figsize=(15, 8))
# Show dev set loss every showDevLossStep batches:
#showDevLossStep = 1
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
best_vloss = 1


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

iterationNumbers = []
iterationDevNumbers = []
epochNumbers = []

lossValuesTrainingSet = []
lossValuesDevSet = []

lossValuesTrainingSetAllEpoch = []
lossValuesDevSetAllEpoch = []


torch.cuda.empty_cache()
unet.to(device)
if saveMhd:
    outputTrainingSet = np.zeros(trainingSetShape)
    outputValidSet = np.zeros(validSetShape)

lossValuesTrainingSetEpoch = []
lossValuesDevSetEpoch = []

diceTraining = [[] for n in range(multilabelNum)]
diceValid = [[] for n in range(multilabelNum)]

diceTrainingEpoch = [[] for n in range(multilabelNum)]
diceValidEpoch = [[] for n in range(multilabelNum)]
#### TRAINING ####

unet.train(False)
for i in range(numBatches):
    # get the inputs
    with torch.no_grad():
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize, :, :, :]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize, :, :, :]).to(device)
        gt = F.one_hot(gt.to(torch.int64))
        gt = torch.squeeze(torch.transpose(gt, 1, 4), 4)
        gt = gt.float()
        if not Background:
            gt = gt[:, 1:, :, :]

        outputs = unet(inputs)
        loss = criterion(outputs, gt)

        lossValuesTrainingSet.append(loss.item())
        lossValuesTrainingSetEpoch.append(loss.item())

    #Print epoch iteration and loss value:

    create_csv(lossValuesTrainingSet, outputPath + 'TestLossIter.csv')
    label = gt.cpu().numpy()
    label = label.astype('int32')
    segmentation = torch.sigmoid(outputs.cpu().to(torch.float32))
    segmentation = maxProb(segmentation.detach().numpy(), multilabelNum)
    segmentation = (segmentation > 0.5) * 1
    if saveMhd:
        outputTrainingSet[i*batchSize:(i+1)*batchSize] = multilabel(segmentation, multilabelNum, Background)

    for k in range(label.shape[0]):
        for j in range(multilabelNum):
            lbl = label[k, j, :, :]
            seg = segmentation[k, j, :, :]
            diceScore = dice2d(lbl, seg)
            diceTraining[j].append(diceScore)

for j in range(multilabelNum):
    diceTrainingEpoch[j].append(np.mean(diceTraining[j]))
    print('Training Dice Score: %f ' % np.mean(diceTraining[j]))

avg_tloss = np.mean(lossValuesTrainingSetEpoch)
lossValuesTrainingSetAllEpoch.append(avg_tloss)
print('avg_tloss: %f' % avg_tloss)

for k in range(multilabelNum):
    create_csv(diceTrainingEpoch[k], outputPath + 'TrainingDice_' + labelNames[k] + '.csv')
create_csv(lossValuesTrainingSetAllEpoch, outputPath + 'TestLossEpoch.csv')



#### VALIDATION ####
unet.train(False)
torch.cuda.empty_cache()
for i in range(devNumBatches):
    with torch.no_grad():
        inputs = torch.from_numpy(devSet['input'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
        gt = torch.from_numpy(devSet['output'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
        gt = F.one_hot(gt.to(torch.int64))
        gt = torch.squeeze(torch.transpose(gt, 1, 4), 4)
        gt = gt.float()
        if not Background:
            gt = gt[:, 1:, :, :]

        outputs = unet(inputs)
        loss = criterion(outputs, gt)

        lossValuesDevSet.append(loss.item())
        lossValuesDevSetEpoch.append(loss.item())

        create_csv(lossValuesDevSet, outputPath + 'ValidLossIter.csv')

    label = gt.cpu().numpy()
    label = label.astype('int32')
    segmentation = torch.sigmoid(outputs.cpu().to(torch.float32))
    segmentation = maxProb(segmentation.detach().numpy(), multilabelNum)
    segmentation = (segmentation > 0.5) * 1
    if saveMhd:
        outputValidSet[i * devBatchSize:(i + 1) * devBatchSize] = multilabel(segmentation, multilabelNum, Background)

    for k in range(label.shape[0]):
        for j in range(multilabelNum):
            lbl = label[k, j, :, :]
            seg = segmentation[k, j, :, :]
            diceScore = dice2d(lbl, seg)
            specScore = specificity(lbl, seg)
            sensScore = sensitivity(lbl, seg)
            diceValid[j].append(diceScore)
for j in range(multilabelNum):
    diceValidEpoch[j].append(np.mean(diceValid[j]))
    print('Valid Dice Score:  %f ' % np.mean(diceValid[j]))

avg_vloss = np.mean(lossValuesDevSetEpoch)
lossValuesDevSetAllEpoch.append(avg_vloss)
print('avg_vloss: %f' % avg_vloss)

for k in range(multilabelNum):
    create_csv(diceValidEpoch[k], outputPath + 'ValidDice_' + labelNames[k] + '.csv')
create_csv(lossValuesDevSetAllEpoch, outputPath + 'ValidLossEpoch.csv')


#boxplot:
if Boxplot:
    boxplot(diceTraining[1:], xlabel=xLabel[1:],
            outpath=(outputPath + 'trainingBoxplot.png'), yscale=[0, 1], title='Training Dice Scores')
    boxplot(diceTraining[1:], xlabel=xLabel[1:],
            outpath=(outputPath + 'trainingBoxplot_shortScale.png'), yscale=[0.7, 1.0], title='Training Dice Scores')
    boxplot(diceValid[1:], xlabel=xLabel[1:],
            outpath=(outputPath + 'validBoxplot.png'), yscale=[0, 1], title='Validation Dice Scores')
    boxplot(diceValid[1:], xlabel=xLabel[1:],
            outpath=(outputPath + 'validBoxplot_shortScale.png'), yscale=[0.7, 1.0], title='Validation Dice Scores')
    for k in range(multilabelNum):
        boxplot(data=(diceTraining[k], diceValid[k]),
                xlabel=['Training Set', 'Valid Set'], outpath=(outputPath + labelNames[k] + '_boxplot.png'),
                yscale=[0.7, 1.0], title=labelNames[k]+' Dice Scores')
if saveMhd:
    writeMhd(outputTrainingSet.astype(np.uint8), outputPath + 'outputTrainingSet.mhd')
    writeMhd(outputValidSet.astype(np.uint8), outputPath + 'outputValidSet.mhd')
print('Test Finished')






