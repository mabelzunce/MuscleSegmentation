import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import os
import math
from datetime import datetime

from sympy import false

from utils import create_csv
from utils import dice2d
from utils import specificity
from utils import sensitivity
from utils import precision
from utils import haus_distance
from utils import maxProb
from utils import writeMhd
from utils import multilabel
from utils import filtered_multilabel
from utils import boxplot

from unet_3d import Unet

import torch
import torch.nn as nn
import torch.nn.functional as F

class Augment(enumerate):
    NA = 0  # No Augment
    NL = 1  # Non linear
    L = 2  # Linear
    A = 3   #Augmented


saveMhd = False        # Saves a mhd file for the output
saveDataSetMhd = False  # Saves a Mhd file of the images and labels from dataset
Background = False       # Background is considered as label
Boxplot = True           # Boxplot created in every best fit
AugmentedTrainingSet = Augment.NA
############################ DATA PATHS ##############################################
trainingSetPath = '../../Data/LumbarSpine3D/ResampledData/'
outputPath = '../../Data/LumbarSpine3D/modelTestResults/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
trainingSubjects = '../../Data/LumbarSpine3D/trainingSubjects/'
validSubjects = '../../Data/LumbarSpine3D/validSubjects/'

trainingTxt = trainingSubjects + os.listdir(trainingSubjects)[0]
validTxt = validSubjects + os.listdir(validSubjects)[0]

modelName = os.listdir(modelLocation)[0]
unetFilename = modelLocation + modelName

if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(trainingSetPath):
    os.makedirs(trainingSetPath)
if not os.path.exists(modelLocation):
    os.makedirs(modelLocation)

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
devSetRelSize = 1 - trainingSetRelSize

######################### CHECK DEVICE ######################
device = torch.device('cuda:1')
print(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
files = sorted(files)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images

numImagesPerSubject = 0
numRot = 0
datasize = 0
numSubjects = 0
subject = files[0].split('_')[0]
subjects = []
for filename in files:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name.split('_')[0]
    # Check if filename is the in phase header and the labels exists:
    filenameImages = trainingSetPath + atlasName + '.' + extensionImages
    filenameLabels = trainingSetPath + atlasName + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and name.endswith(tagLabels):
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
            if not (subjectName.startswith(subjectAdd)):
                numRot += 1
        if not (subjects.__contains__(subjectName)):
            numSubjects += 1
            subjects.append(subjectName)
        datasize += 1

tNum = int(np.floor(numSubjects * trainingSetRelSize))
vNum = int(np.ceil(numSubjects * devSetRelSize))

textFile = open(trainingTxt,'r')
trainingSubjects = textFile.readlines()
trainingSubjects = [s.replace('\n', '') for s in trainingSubjects]
textFile.close()

textFile = open(validTxt,'r')
validSubjects = textFile.readlines()
validSubjects = [s.replace('\n', '') for s in validSubjects]
textFile.close()


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
flag = True
for i in range(0, datasize):
    #name1, name2 = atlasNames[i].split('_')[:1]
    condition1 = (trainingSubjects.__contains__(atlasNames[i]))
    condition2 = (validSubjects.__contains__(atlasNames[i]))

    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabel = sitk.ReadImage(atlasLabelsFilenames[i])

    trainingSetShape = [tNum, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]]
    validSetShape = [vNum, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]]

    if i == 0:
        imagesTrainingSet = np.zeros(trainingSetShape)
        labelsTrainingSet = np.zeros(trainingSetShape)
        imagesValidSet = np.zeros(validSetShape)
        labelsValidSet = np.zeros(validSetShape)
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesTrainingSet.shape[1:4]  # obtiene el getsize[1 y 0]

    if condition1:
        imagesTrainingSet[k, :, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage),[1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsTrainingSet[k, :, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabel), [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        k += 1

    if condition2:
        imagesValidSet[j, :, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsValidSet[j, :, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabel), [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        j += 1

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
labelNames = ('Background', 'Psoas izquierdo', 'Iliaco izquierdo', 'Cuadrado izquierdo', 'Paraespinal Posterior izquierdo', 'Psoas derecho', 'Iliaco Derecho', 'Cuadrado derecho', 'Paraespinal Posterior derecho')
####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and multiple output canal.
multilabelNum = 8
if Background:
    multilabelNum += 1
    xLabel = ['BG', 'LP', 'LI', 'LQ', 'LM', 'RP', 'RI', 'RQ', 'RM']
    criterion = nn.BCEWithLogitsLoss()
else:
    labelNames = labelNames[1:]
    xLabel = ['$P_i$', '$I_i$', '$CL_i$', '$ES+M_i$', '$P_d$', '$I_d$', '$CL_d$', '$ES+M_d$']
    criterion = nn.BCEWithLogitsLoss()

unet = Unet(1, multilabelNum)
unet.load_state_dict(torch.load(unetFilename, map_location=device,weights_only=True))
total_params = sum(p.numel() for p in unet.parameters())
print("Total Parameters: " + str(total_params))
#tensorGroundTruth.shape
##################################### U-NET TRAINING ############################################
# Number of  batches:
batchSize = 1
devBatchSize = 1
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

lossValuesTrainingSetEpoch = []
lossValuesDevSetEpoch = []

diceTraining = [[] for n in range(multilabelNum)]
diceValid = [[] for n in range(multilabelNum)]

diceTrainingEpoch = [[] for n in range(multilabelNum)]
diceValidEpoch = [[] for n in range(multilabelNum)]

sensTraining = [[] for n in range(multilabelNum)]
sensValid = [[] for n in range(multilabelNum)]

sensTrainingEpoch = [[] for n in range(multilabelNum)]
sensValidEpoch = [[] for n in range(multilabelNum)]

specTraining = [[] for n in range(multilabelNum)]
specValid = [[] for n in range(multilabelNum)]

specTrainingEpoch = [[] for n in range(multilabelNum)]
specValidEpoch = [[] for n in range(multilabelNum)]

precTraining = [[] for n in range(multilabelNum)]
precValid = [[] for n in range(multilabelNum)]

precTrainingEpoch = [[] for n in range(multilabelNum)]
precValidEpoch = [[] for n in range(multilabelNum)]

hausTraining = [[] for n in range(multilabelNum)]
hausValid= [[] for n in range(multilabelNum)]

hausTrainingEpoch = [[] for n in range(multilabelNum)]
hausValidEpoch = [[] for n in range(multilabelNum)]

#### TRAINING ####

unet.train(False)
for i in range(numBatches):
    print(i)
    # get the inputs
    with torch.no_grad():
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize, :, :, :]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize, :, :, :]).to(device)
        gt = F.one_hot(gt.to(torch.int64))
        gt = torch.squeeze(torch.transpose(gt, 1, 5), 5)
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
            if i==0:
                outputTrainingSet = multilabel(segmentation,Background)
            else:
                outputTrainingSet = np.append(outputTrainingSet, multilabel(segmentation, Background), axis=0)

        for k in range(label.shape[0]):
            for j in range(multilabelNum):
                lbl = label[k, j, :, :, :]
                seg = segmentation[k, j, :, :, :]
                diceScore = dice2d(lbl, seg)
                specScore = specificity(lbl, seg)
                sensScore = sensitivity(lbl, seg)
                precScore = precision(lbl, seg)
                #hausScore = haus_distance(lbl,seg)
                diceTraining[j].append(diceScore)
                specTraining[j].append(specScore)
                sensTraining[j].append(sensScore)
                precTraining[j].append(precScore)
                #hausTraining[j].append(hausScore)

for j in range(multilabelNum):
    diceTrainingEpoch[j].append(np.mean(diceTraining[j]))
    print('Training Dice Score: %f ' % np.mean(diceTraining[j]))


for k in range(multilabelNum):
    create_csv(diceTraining[k], outputPath + 'TrainingDice_' + labelNames[k] + '.csv')




    #### VALIDATION ####
for i in range(devNumBatches):
    print(i)
    with torch.no_grad():
        inputs = torch.from_numpy(devSet['input'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
        gt = torch.from_numpy(devSet['output'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
        gt = F.one_hot(gt.to(torch.int64), num_classes=multilabelNum + 1)
        gt = torch.squeeze(torch.transpose(gt, 1, 5), 5)
        gt = gt.float()
        if not Background:
            gt = gt[:, 1:, :, :]

        outputs = unet(inputs)
        loss = criterion(outputs, gt)

        lossValuesDevSet.append(loss.item())
        lossValuesDevSetEpoch.append(loss.item())

        create_csv(lossValuesDevSet, outputPath + 'TestValidLossIter.csv')

    label = gt.cpu().numpy()
    label = label.astype('int32')
    segmentation = torch.sigmoid(outputs.cpu().to(torch.float32))
    segmentation = maxProb(segmentation.detach().numpy(), multilabelNum)
    segmentation = (segmentation > 0.5) * 1
    if saveMhd:
        if i == 0:
            outputValidSet = multilabel(segmentation, Background)
        else:
            outputValidSet = np.append(outputValidSet, multilabel(segmentation, Background), axis=0)

    for k in range(label.shape[0]):
        for j in range(multilabelNum):
            lbl = label[k, j, :, :, :]
            seg = segmentation[k, j, :, :, :]
            diceScore = dice2d(lbl, seg)
            specScore = specificity(lbl, seg)
            sensScore = sensitivity(lbl, seg)
            precScore = precision(lbl, seg)
            hausScore = haus_distance(lbl,seg)
            diceValid[j].append(diceScore)
            specValid[j].append(specScore)
            sensValid[j].append(sensScore)
            precValid[j].append(precScore)
            hausValid[j].append(hausScore)
for j in range(multilabelNum):
    diceValidEpoch[j].append(np.mean(diceValid[j]))
    specValidEpoch[j].append(np.mean(specValid[j]))
    sensValidEpoch[j].append(np.mean(sensValid[j]))
    precValidEpoch[j].append(np.mean(precValid[j]))
    hausValidEpoch[j].append(np.mean(hausValid[j]))
    print('Valid Dice Score:  %f ' % np.mean(diceValid[j]))

avg_vloss = np.mean(lossValuesDevSetEpoch)
lossValuesDevSetAllEpoch.append(avg_vloss)
print('avg_vloss: %f' % avg_vloss)

for k in range(multilabelNum):
    create_csv(diceValid[k], outputPath + 'Test_ValidDice_' + labelNames[k] + '.csv')
    create_csv(sensValid[k], outputPath + 'Test_ValidSensitivity_' + labelNames[k] + '.csv')
    create_csv(specValid[k], outputPath + 'Test_ValidSpecificity_' + labelNames[k] + '.csv')
    create_csv(precValid[k], outputPath + 'Test_ValidPrecision_' + labelNames[k] + '.csv')
    create_csv(hausValid[k], outputPath + 'Test_ValidHaus_' + labelNames[k] + '.csv')



#boxplot:
if Boxplot:
    boxplot(diceTraining[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_trainingBoxplot.tif'), yscale=[0, 1], title='Puntaje Dice en Set de Entrenamiento')
    boxplot(diceTraining, xlabel=xLabel,
            outpath=(outputPath + 'Test_trainingBoxplot_shortScale.tif'), yscale=[0.7, 1.0], title='Puntaje Dice en Set de Entrenamiento')
    boxplot(diceValid[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_validBoxplot.tif'), yscale=[0, 1], title='Validation Set Dice Scores')
    boxplot(diceValid, xlabel=xLabel,
            outpath=(outputPath + 'Test_validBoxplot_shortScale.tif'), yscale=[0.7, 1.0], title='Puntaje Dice en Set de Validación')
    boxplot(sensTraining[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_trainingSensitivityBoxplot.tif'), yscale=[0, 1], title='Sensibilidad en set de entrenamiento')
    boxplot(sensValid[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_validSensitivityBoxplot.tif'), yscale=[0, 1], title='Validation Sensitivity Scores')
    boxplot(specTraining[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_trainingSpecificityBoxplot.tif'), yscale=[0, 1], title='Especificidad en set de entrenamiento')
    boxplot(specValid[:], xlabel=xLabel[:],
            outpath=(outputPath + 'Test_validSpecificityBoxplot.tif'), yscale=[0, 1], title='Especificidad en set de Validacion')
    boxplot(precValid[:], xlabel=xLabel,
        outpath=(outputPath + 'Test_validPrecisionBoxplot.tif'), yscale=[0.7, 1], title='Precisión en set de validación')
    for k in range(multilabelNum):
        boxplot(data=(diceTraining[k], diceValid[k]),
                xlabel=['Entrenamiento', 'Validación'], outpath=(outputPath + labelNames[k] + '_boxplot.tif'),
                yscale=[0.7, 1.0], title=labelNames[k]+' Puntaje Dice')
if saveMhd:
    writeMhd(outputTrainingSet.astype(np.uint8), outputPath + 'outputTrainingSet.mhd')
    writeMhd(outputValidSet.astype(np.uint8), outputPath + 'outputValidSet.mhd')
print('Test Finished')






