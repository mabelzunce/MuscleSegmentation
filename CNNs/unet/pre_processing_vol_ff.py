import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions
from skimage.morphology import convex_hull_image
import imageio
from PIL import Image
import csv
import pandas as pd

#Functions:

# BIAS FIELD CORRECTION
def ApplyBiasCorrection(inputImage, shrinkFactor = (1,1,1)):
    # Bias correction filter:
    biasFieldCorrFilter = sitk.N4BiasFieldCorrectionImageFilter()
    mask = sitk.OtsuThreshold( inputImage, 0, 1, 100)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    # Parameter for the bias corredtion filter:
    biasFieldCorrFilter.SetSplineOrder(3)
    biasFieldCorrFilter.SetConvergenceThreshold(0.0001)
    biasFieldCorrFilter.SetMaximumNumberOfIterations((50, 40, 30))

    if shrinkFactor != (1,1,1):
        # Shrink image and mask to accelerate:
        shrinkedInput = sitk.Shrink(inputImage, shrinkFactor)
        mask = sitk.Shrink(mask, shrinkFactor)


        #biasFieldCorrFilter.SetNumberOfThreads()
        #biasFieldCorrFilter.UseMaskLabelOff() # Because I'm having problems with the mask.
        # Run the filter:
        output = biasFieldCorrFilter.Execute(shrinkedInput, mask)
        # Get the field by dividing the output by the input:
        outputArray = sitk.GetArrayFromImage(output)
        shrinkedInputArray = sitk.GetArrayFromImage(shrinkedInput)
        biasFieldArray = np.ones(np.shape(outputArray), 'float32')
        biasFieldArray[shrinkedInputArray != 0] = outputArray[shrinkedInputArray != 0]/shrinkedInputArray[shrinkedInputArray != 0]
        biasFieldArray[shrinkedInputArray == 0] = 0
        # Generate bias field image:
        biasField = sitk.GetImageFromArray(biasFieldArray)
        biasField.SetSpacing(shrinkedInput.GetSpacing())
        biasField.SetOrigin(shrinkedInput.GetOrigin())
        biasField.SetDirection(shrinkedInput.GetDirection())

        # Now expand
        biasField = sitk.Resample(biasField, inputImage)

        # Apply to the image:
        output = sitk.Multiply(inputImage, biasField)
    else:
        #output = biasFieldCorrFilter.Execute(inputImage, mask)
        output = biasFieldCorrFilter.Execute(inputImage)
    # return the output:
    return output

#VOLUMES TO .CSV
def write_volumes_to_csv(output_csv_path, volumes, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'Vol {label}' for label in sorted(volumes.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())]
        writer.writerow(row)

#FFS TO .CSV
def write_ff_to_csv(output_csv_path, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

def write_vol_ff_to_csv(output_csv_path, volumes, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

#BINARY SEGMENTATIONS:
# Function to load, binarize and save segmentations
#Also calculates the total volume of the binary segmentation
def process_and_save_segmentations(folder, output_folder):
    volume_results = {}
    # CCreate output directory
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")

            segmentation_image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(segmentation_image)  # Array 3D: (Depth, Height, Width)

            # Binarize
            binary_array = (array > 0).astype("uint8")

            # Convert again to image
            binary_image = sitk.GetImageFromArray(binary_array)

            # Copy spatial information
            binary_image.CopyInformation(segmentation_image)

            # Save the binary image in the output directory
            output_path = os.path.join(output_folder, f"{key}_seg_binary.mhd")
            sitk.WriteImage(binary_image, output_path)

            print(f"Binary segmentation saved in: {output_path}")

            # Volume calculation:
            spacing = segmentation_image.GetSpacing()  # (z_spacing, y_spacing, x_spacing)

            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volumen de un voxel

            num_segmented_voxels = np.sum(binary_array)

            total_volume = num_segmented_voxels * voxel_volume

            volume_results[key] = total_volume

            print(f"Total volume for {key}: {total_volume} mm^3")

    return volume_results

#APPLY MASK AND CALCULATE MEAN FF:
def apply_mask_and_calculate_ff(folder):
    ff_results = {}
    for file in os.listdir(folder):
        if file.endswith("_seg_binary.mhd"):
            # Load the binary mask
            mask_path = os.path.join(folder, file)
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)  # Array 3D: (Depth, Height, Width)

            # Find the '_ff' image for every subject
            key = file.replace("_seg_binary.mhd", "")
            ff_file = f"{key}_ff.mhd"
            ff_path = os.path.join(folder, ff_file)

            if os.path.exists(ff_path):
                # Load the '_ff' image
                ff_image = sitk.ReadImage(ff_path)
                ff_array = sitk.GetArrayFromImage(ff_image)  # Array 3D: (Depth, Height, Width)

                # Check the dimensions
                if mask_array.shape != ff_array.shape:
                    print(f"Dimensiones no coinciden entre máscara y '_ff' para: {key}")
                    continue

                # Apply the mask
                masked_values = ff_array[mask_array > 0]

                # Calculate the mean ff
                mean_ff = np.mean(masked_values)
                ff_results[key] = mean_ff

                print(f"Mean FF for {key}: {mean_ff}")

    return ff_results

#SAVE TOTAL VOLUME AND FF ON THE SAME .CSV FILE
def save_results_to_csv(volume_results, ff_results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        #writer.writerow(["Sujeto", "Volumen (mm^3)", "FF Promedio"])

        for key in volume_results:
            volume = volume_results.get(key, "N/A")
            ff = ff_results.get(key, "N/A")
            writer.writerow([key, volume, ff])

    print(f"Information saved in: {output_csv}")


# CONFIGURATION:
# Needs registration
preRegistration = True #TRUE: Pre-register using the next image
registrationReferenceFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'

#DATA PATHS:
dataPath = '/home/facundo/data/Nepal/DIXON/' #INPUT FOLDER THAT CONTAINS ALL THE SUBDIRECTORIES
outputPath = '/home/facundo/data/Nepal/NewSegmentation//' #OUPUT FOLDER TO SAVE THE SEGMENTATIONS
outputResampledPath = outputPath + '/Resampled/' #RESAMPLED SEGMENTATIONS PATH
#outputBiasCorrectedPath = outputPath + '/BiasFieldCorrection/'
outputBiasCorrectedPath = dataPath #I save the images in the same folder because its easier
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
dataInSubdirPerSubject = True

#PATHS FOR THE BINARY SEGMENTATION:
#In that case input and output paths are the same
# Where the original segmentations are saved
inputSeg = outputPath
# Where will be the binary segmentations
outputBinSeg = outputPath
#Folder that contains the binary segmentation and 'ff' images
binarySegAndFFPath = outputPath

# REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION:
referenceImageFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'
referenceImage = sitk.ReadImage(referenceImageFilename)

# REGISTRATION PARAMETER FILES:
similarityMetricForReg = 'NMI' #NMI Metric
parameterFilesPath = '../../Data/Elastix/' #Parameters path
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg

#IMAGE FORMATS AND EXTENSION:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_I'#
# outOfPhaseSuffix
waterSuffix = '_W'
fatSuffix = '_F'
tagAutLabels = '_aut'
tagManLabels = '_labels'

# OUTPUT PATHS:
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

if not os.path.exists(outputResampledPath):
    os.makedirs(outputResampledPath)

if not os.path.exists(outputBiasCorrectedPath):
    os.makedirs(outputBiasCorrectedPath)

vol_csv_name = 'volumes.csv'
ff_csv_name = 'ffs.csv'
vol_and_ff_name = 'volumes_and_ffs.csv'
totalvol_and_meanff_name = 'TotalVol_MeanFF.csv'

#Clear the .csv files if they've got information
file_path_v = os.path.join(outputPath, ff_csv_name)
if os.path.exists(file_path_v):
    with open(file_path_v, 'w') as file:
        pass  # Clear

file_path_ff = os.path.join(outputPath, vol_csv_name)
if os.path.exists(file_path_ff):
    with open(file_path_ff, 'w') as file:
        pass  # Clear

file_path_vol_ff = os.path.join(outputPath, vol_and_ff_name)
if os.path.exists(file_path_vol_ff):
    with open(file_path_vol_ff, 'w') as file:
        pass  # Clear

# CSV THAT CONTAINS TOTAL VOLUME AND MEAN FF PATH
TotalVol_MeanFF_csv = os.path.join(outputPath, totalvol_and_meanff_name)
if os.path.exists(TotalVol_MeanFF_csv):
    with open(TotalVol_MeanFF_csv, 'w') as file:
        pass  # Clear

# MODEL:
modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

#CHECK DEVICE:
device = torch.device('cpu') #'cuda' uses the graphic board
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

# MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage.GetSize()))

# LOOK FOR THE FOLDERS OF THE IN-PHASE IMAGES:
#files = os.listdir(dataPath)
if dataInSubdirPerSubject: # True: info in subdirectories
    files = list()
    subdirs = os.listdir(dataPath) #Folders
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

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
i = 0
#files = files[0:2]
for fullFilename in files:
    fileSplit = os.path.split(fullFilename) #Divide the path in directory-name
    pathSubject = fileSplit[0]
    filename = fileSplit[1]
    name, extension = os.path.splitext(filename)
    subject = name[:-len(tagInPhase)] #Name of the subject without the tag
    print(subject) #Flag to check the actual subject

    #Read the image
    sitkImage = sitk.ReadImage(fullFilename)

    # Apply Bias Field Correction
    shrinkFactor = (2, 2, 1)
    sitkImage = ApplyBiasCorrection(sitkImage, shrinkFactor=shrinkFactor)

    # Obtains the name of the file (without the complete path and divide name and extension)
    filename_no_ext, file_extension = os.path.splitext(filename)

    # Generates the new name with the "_biasFieldCorrection"
    new_filename = f"{filename_no_ext}_biasFieldCorrection{file_extension}"

    # Builds the path to save the corrected image
    outputBiasFilename = os.path.join(outputBiasCorrectedPath, new_filename)

    sitk.WriteImage(sitkImage, outputBiasFilename)

    if preRegistration:
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter() #Create the object
        # Register image to reference data
        elastixImageFilter.SetFixedImage(referenceImage) #Defines reference image
        elastixImageFilter.SetMovingImage(sitkImage) #Defines moving image
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute()
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampled = elastixImageFilter.GetResultImage() #Result image from the register
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
    with torch.no_grad(): #SEGMENTATION:
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImageResampled)

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

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        transformixImageFilter.SetMovingImage(output)
        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    sitk.WriteImage(output, outputPath + subject + '_segmentation' + extensionImages, True)

    #VOLUME CALCULATION:

    # Get the spacial dimensions
    spacing = sitkImageResampled.GetSpacing()  #Tuple (spacing_x, spacing_y, spacing_z)
    print(spacing)

    #Segmentation to array
    segmentation_array = sitk.GetArrayFromImage(output)

    # Number of labels of the segmentation
    num_labels = multilabelNum

    # Voxel volume
    voxel_volume = np.prod(spacing) #volume (X.Y.Z)

    #Path of the volume csv file
    output_csv_path = os.path.join(outputPath, 'volumes.csv')

    # Volume dictionary to save the label volumes
    volumes = {}

    # Iterate over labels
    for label in range(num_labels+1):
        # Creates a mask for the actual label (1 or 0)
        label_mask = (segmentation_array == label).astype(np.uint8)

        # Counts the number of voxels
        label_voxels = np.sum(label_mask)

        # Calculates the volume of that class (quantity of voxels * individual voxel volume)
        label_volume = label_voxels * voxel_volume

        # Save the data on the dictionary
        volumes[label] = label_volume

    #Write on the .csv
    write_volumes_to_csv(output_csv_path, volumes, subject)

    # Print the volume of all the labels:
    print("\nVolumes:")
    for label, volume in volumes.items():
        print(f"Muscle {label}: {volume} mm³")

    #GENERATE FAT FRACTION IMAGE (FF = F/F+W):

    # Folder lists of the main directory
    subdirectories = sorted([d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))])

    auxName = None
    for folder in subdirectories:
        folder_path = os.path.join(dataPath, folder)
        fat_file = os.path.join(folder_path, folder + fatSuffix + extensionImages)
        water_file = os.path.join(folder_path, folder + waterSuffix + extensionImages)

        # Check if the files exists:
        if os.path.exists(fat_file) and os.path.exists(water_file):
            fatImage = sitk.Cast(sitk.ReadImage(fat_file), sitk.sitkFloat32)
            waterImage = sitk.Cast(sitk.ReadImage(water_file), sitk.sitkFloat32)

            # Calculate the FF image and apply the mask on:
            waterfatImage = sitk.Add(fatImage, waterImage)
            fatfractionImage = sitk.Divide(fatImage, waterfatImage)
            fatfractionImage = sitk.Cast(
                sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                sitk.sitkFloat32)

            # Save the resulting image:
            output_filename = os.path.join(outputPath, folder + '_ff' + extensionImages)
            sitk.WriteImage(fatfractionImage, output_filename)
            #print(f"FF Image saved in: {output_filename}")
        else:
            print(
                f"Archivo faltante en {folder_path}. Ver que los archivos {folder}_F.mhd y {folder}_W.mhd estén presentes.")

    #FF CALCULATION:

    # FUNCTION TO RESAMPLE IMAGES:

    #Resample the images:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(output)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(fatfractionImage.GetPixelID())
    fatfractionImage_resampled = resampler.Execute(fatfractionImage)

    #def resample_to_match(image, reference):

    #    resampler = sitk.ResampleImageFilter()
    #    resampler.SetReferenceImage(reference)
    #    resampler.SetInterpolator(sitk.sitkLinear)
    #    resampler.SetOutputPixelType(image.GetPixelID())
    #    return resampler.Execute(image)

    # Remuestrear fatfractionImage para que coincida con output
    #fatfractionImage_resampled = resample_to_match(fatfractionImage, output)

    # Check the dimensions
    if fatfractionImage_resampled.GetSize() != output.GetSize():
        print(f"Dimensiones incompatibles: "
              f"fatfractionImage_resampled {fatfractionImage_resampled.GetSize()} vs output {output.GetSize()}")
        raise ValueError("Las dimensiones de las imágenes no coinciden después del remuestreo.")

    # Images to numpy arrays
    fatfraction_array = sitk.GetArrayFromImage(fatfractionImage_resampled)
    segmentation_array = sitk.GetArrayFromImage(output)

    #FF .CSV PATH:
    output_csv_path = os.path.join(outputPath, 'ffs.csv')

    # Mean FF for every single label
    fat_fraction_means = {}

    for label in range(multilabelNum+1):
        # Mask for actual label
        label_mask = (segmentation_array == label)

        # Check for true values
        if np.any(label_mask):
            # Get the FF values linked to the mask
            fat_values = fatfraction_array[label_mask]

            # Calculate mean value for the label
            fat_fraction_means[label] = np.mean(fat_values)
        else:
            fat_fraction_means[label] = None  # No values for that label

    # Write on the csv file
    write_ff_to_csv(output_csv_path, fat_fraction_means, subject)

    # Print the results
    print("\nFFs:")
    #for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Muscle {label}: {fat_mean:.4f}")
        else:
            print(f"Muscle {label}: Sin valores válidos")

    #SAVE THE VOLUME AND FF VALUES FOR EVERY SUBJECT ON THE SAME .CSV FILE:
    # VOL & FF .CSV PATH:
    output_csv_path = os.path.join(outputPath, 'volumes_and_ffs.csv')
    #CSV FOR ALL THE VOLUME AND FF LABELS:
    write_vol_ff_to_csv(output_csv_path, volumes, fat_fraction_means, subject)


#BINARY SEGMENTATIONS:
#Run the methods that calculate the binary segmentation, calculate the total volume and the mean FF
volume_results = process_and_save_segmentations(inputSeg, outputBinSeg)
ff_results = apply_mask_and_calculate_ff(binarySegAndFFPath)

# Save the results
save_results_to_csv(volume_results, ff_results, TotalVol_MeanFF_csv)


#Generate the complete .csv
# Load the files without the header
df1 = pd.read_csv(os.path.join(outputPath, vol_csv_name), header=None)
df2 = pd.read_csv(os.path.join(outputPath,ff_csv_name), header=None)
df3 = pd.read_csv(os.path.join(outputPath, totalvol_and_meanff_name), header=None)

#New csv that contains the volume and ff information (for every label)
#df4 must be = to df1+df2
df4 = pd.read_csv(os.path.join(outputPath,vol_and_ff_name), header=None)

# Concatenate the DFs based on first column info (índex 0)
#df_concat = pd.merge(df1, df2, left_on=0, right_on=0, how='outer')
df_concat = pd.merge(df4, df3, left_on=0, right_on=0, how='outer')

# Save the DF in a new file (all_vols_and_ffs)
df_concat.to_csv(os.path.join(outputPath,'all_vols_and_ffs.csv'), index=False, header=False)

#Generate individual .csvs and save those files in each subject folder
#Use the concatenate file
df = pd.read_csv(os.path.join(outputPath,'all_vols_and_ffs.csv'), header=None)

# Every row is a different subject
for _, row in df.iterrows():
    subject_id = row[0]  # Index in 1st column

    # Subject's folder path
    subject_folder = os.path.join(dataPath, str(subject_id))

    # Check if the subject exists:
    if os.path.exists(subject_folder):
        # Create a DF for that subject
        subject_data = df[df[0] == subject_id]

        # Save the data inside of his/her folder
        subject_data.to_csv(f'{subject_folder}/{subject_id}.csv', index=False, header=False)
    else:
        print(f"The folder for the subject {subject_id} does not exists.")

"""
#GIFS
#Código para crear GIFS de las segmentaciones

# Ruta de la carpeta donde están los archivos
input_folder = outputPath
output_folderSG = outputPath + 'SegmentationGifs/'

#Si no existen, los crea
if not os.path.exists(output_folderSG):
    os.makedirs(output_folderSG)

# Función para cargar archivos .mhd
def load_segmentations_mhd(folder):
    segmentations = {}
    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")
            # Leer la imagen con SimpleITK
            image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(image)  # Array 3D: (Depth, Height, Width)
            segmentations[key] = array
    return segmentations

# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animation.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

# Ejecutar el flujo
segmentations = load_segmentations_mhd(input_folder)
create_gifs(segmentations, output_folderSG)


#CREAR IMAGEN CON SEGMENTACIÓN POR TEJIDO:

#PARAMETROS
targetPath = '/home/facundo/data/Nepal/DIXON/'

# Cases to process, leave it empty to process all the cases in folder:
casesToSegment = ()
casesToSegment = ()
casesToSegment = list()

# Look for the folders or shortcuts:
files = os.listdir(targetPath)

# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
inPhaseSuffix = '_I_biasFieldCorrection'# Pongo esto para agarrar la corrección
outOfPhaseSuffix = '_O'#
waterSuffix = '_W'#
fatSuffix = '_F'#
suffixSegmentedImages = '_tissue_segmented'
suffixSkinFatImages = '_skin_fat'
suffixBodyImages = '_body'
suffixMuscleImages = '_muscle'
suffixFatFractionImages = '_fat_fraction'
dixonSuffixInOrder = (inPhaseSuffix, outOfPhaseSuffix, waterSuffix, fatSuffix)
#dixonImages = []

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate images are written.
USE_COSINES_AND_ORIGIN = 1
OVERWRITE_EXISTING_SEGMENTATIONS = 1

#FUNCIONES PARA IMAGENES DE TEJIDOS Y DE GRASA SUBCUTANEA:
#FUNCION BODY MASK DESDE INPHASE
# Function that creates a mask for the body from an in-phase dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration.
def GetBodyMaskFromInPhaseDixon(inPhaseImage, vectorRadius = (2,2,2)):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(inPhaseImage, 4, 0, 128, # 4 classes and 128 bins
                                            False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
    # Open the mask to remove connected regions
    background = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 0), vectorRadius, kernel)
    background = sitk.BinaryDilate(background, vectorRadius, kernel)
    bodyMask = sitk.Not(background)
    bodyMask.CopyInformation(inPhaseImage)
    # Fill holes:
    #bodyMask = sitk.BinaryFillhole(bodyMask, False)
    # Fill holes in 2D (to avoid holes coming from bottom and going up):
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    return bodyMask


#FUNCION BODY MASK DESDE FAT
# Function that creates a mask for the body from an fat dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration. Assumes that skin fat
# surround the patient body.
def GetBodyMaskFromFatDixonImage(fatImage, vectorRadius = (2,2,2), minObjectSizeInSkin = 500):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(fatImage, 1, 0, 128, # 1 classes and 128 bins
                                            False)  # 2 Classes, itk, doesn't coun't the background as a class, so we use 1 in the input parameters.
    # Open the mask to remove connected regions, mianly motion artefacts outside the body
    fatMask = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 1), vectorRadius, kernel)
    # Remove small objects:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilter = sitk.RelabelComponentImageFilter()
    relabelComponentFilter.SetMinimumObjectSize(minObjectSizeInSkin)
    sliceFatObjects = relabelComponentFilter.Execute(
        connectedFilter.Execute(fatMask))  # RelabelComponent sort its by size.
    fatMask = sliceFatObjects > 0  # Assumes that can be two large objetcts at most (for each leg)

    fatMask = sitk.BinaryDilate(fatMask, vectorRadius)
    bodyMask = fatMask
    # Go through all the slices:
    for j in range(0, fatMask.GetSize()[2]):
        sliceFat = fatMask[:, :, j]
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFat)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        # Now paste the slice in the output:
        sliceBody = sitk.JoinSeries(sliceFatConvexHull)  # Needs to be a 3D image
        bodyMask = sitk.Paste(bodyMask, sliceBody, sliceBody.GetSize(), destinationIndex=[0, 0, j])
    bodyMask = sitk.BinaryDilate(bodyMask, vectorRadius)
    return bodyMask


#FUNCION CALCULO TEJIDO ADIPOSO SUBCUTANEO (Toma la imagen de la función anterior porque tiene que usar las etiquetas 3)
# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(dixonSegmentedImage, minObjectSizeInMuscle = 500, minObjectSizeInSkin = 500):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    # Create a mask for other tissue:
    notFatMask = sitk.And(bodyMask, (dixonSegmentedImage < 3))
    notFatMask = sitk.BinaryMorphologicalOpening(notFatMask, 3)
    #Filter to process the slices:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilterMuscle = sitk.RelabelComponentImageFilter()
    relabelComponentFilterMuscle.SetMinimumObjectSize(minObjectSizeInMuscle)
    relabelComponentFilterSkin = sitk.RelabelComponentImageFilter()
    relabelComponentFilterSkin.SetMinimumObjectSize(minObjectSizeInSkin)
    # Go through all the slices:
    for j in range(0, skinFat.GetSize()[2]):
        sliceFat = skinFat[:, :, j]
        sliceNotFat = notFatMask[:, :, j]
        # Remove external objects:
        sliceFatEroded = sitk.BinaryMorphologicalOpening(sliceFat, 5)
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFatEroded)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        #sliceNotFat = sitk.BinaryErode(sliceNotFat, 3)

        # Get the largest connected component:
        sliceNotFat = sitk.And(sliceNotFat, sliceFatConvexHull) # To remove fake object in the outer region of the body due to coil artefacts.
        sliceNotFatObjects = relabelComponentFilterMuscle.Execute(
            connectedFilter.Execute(sliceNotFat))  # RelabelComponent sort its by size.
        sliceNotFat = sliceNotFatObjects > 0 # sitk.And(sliceNotFatObjects > 0, sliceNotFatObjects < 3) # Assumes that can be two large objetcts at most (for each leg)
        # Dilate to return to the original size:
        sliceNotFat = sitk.BinaryDilate(sliceNotFat, 3)  # dilate to recover original size

        # Now apply the convex hull:
        ndaNotFatMask = sitk.GetArrayFromImage(sliceNotFat)
        ndaNotFatMask = convex_hull_image(ndaNotFatMask)
        sliceNotFat = sitk.GetImageFromArray(ndaNotFatMask.astype('uint8'))
        sliceNotFat.CopyInformation(sliceFat)
        sliceFat = sitk.And(sliceFat, sitk.Not(sliceNotFat))
        # Leave the objects larger than minSize for the skin fat:
        sliceFat = relabelComponentFilterSkin.Execute(
            connectedFilter.Execute(sliceFat))
        #sliceFat = sitk.Cast(sliceFat, sitk.sitkUInt8)
        sliceFat = sliceFat > 0
        # Now paste the slice in the output:
        sliceFat = sitk.JoinSeries(sliceFat)  # Needs to be a 3D image
        skinFat = sitk.Paste(skinFat, sliceFat, sliceFat.GetSize(), destinationIndex=[0, 0, j])
    skinFat = sitk.BinaryDilate(skinFat, 3)
    return skinFat


#FUNCION RELLENAR AGUJEROS POR CORTE
# Auxiliary function that fill hole in an image but per each slice:
def BinaryFillHolePerSlice(input):
    output = input
    for j in range(0, input.GetSize()[2]):
        slice = input[:,:,j]
        slice = sitk.BinaryFillhole(slice, False)
        # Now paste the slice in the output:
        slice = sitk.JoinSeries(slice) # Needs tobe a 3D image
        output = sitk.Paste(output, slice, slice.GetSize(), destinationIndex=[0, 0, j])
    return output


# FUNCION SEGMENTAR TODOS LOS TEJIDOS
# DixonTissueSegmentation received the four dixon images in the following order: in-phase, out-of-phase, water, fat.
# Returns a labelled image into 4 tissue types: air-background (0), soft-tissue (1), soft-tissue/fat (2), fat (3)
def DixonTissueSegmentation(dixonImages):
    labelAir = 0
    labelFat = 3
    labelSoftTissue = 1
    labelFatWater = 2
    labelBone = 4
    labelUnknown = 5

    # Threshold for background:
    backgroundThreshold = 80
    # Threshold for water fat ratio:
    waterFatThreshold = 2
    # Generate a new image:
    segmentedImage = sitk.Image(dixonImages[0].GetSize(), sitk.sitkUInt8)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # otsuOtuput = sitk.OtsuMultipleThresholds(dixonImages[0], 4, 0, 128, False)
    # voxelsAir = sitk.Equal(otsuOtuput, 0)
    # Faster and simpler version but will depend on intensities:
    voxelsAir = sitk.Less(dixonImages[0], backgroundThreshold)

    # Set air tags for lower values:
    # segmentedImage = sitk.Mask(segmentedImage, voxelsAir, labelUnknown, labelAir)
    ndaSegmented = sitk.GetArrayFromImage(segmentedImage)
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaSegmented.fill(labelUnknown)
    ndaSegmented[ndaInPhase < backgroundThreshold] = labelAir

    # Get arrays for the images:
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaOutOfPhase = sitk.GetArrayFromImage(dixonImages[1])
    ndaWater = sitk.GetArrayFromImage(dixonImages[2])
    ndaFat = sitk.GetArrayFromImage(dixonImages[3])

    # SoftTisue:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaFat != 0)] = ndaWater[(ndaFat != 0)] / ndaFat[(ndaFat != 0)]
    # ndaSegmented[np.isnan(WFratio)] = labelUnknown
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, (ndaSegmented == labelUnknown))] = labelSoftTissue
    # Also include when fat is zero and water is different to zero:
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelSoftTissue

    # For fat use the FW ratio:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaWater != 0)] = ndaFat[(ndaWater != 0)] / ndaWater[(ndaWater != 0)]

    # Fat:
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, ndaSegmented == labelUnknown)] = labelFat
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelFat

    # SoftTissue/Fat:
    ndaSegmented[np.logical_and(WFratio < waterFatThreshold, ndaSegmented == labelUnknown)] = labelFatWater

    # Set the array:
    segmentedImage = sitk.GetImageFromArray(ndaSegmented)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # The fat fraction image can have issues in the edge, for that reason we apply a body mask from the inphase image
    maskBody = GetBodyMaskFromInPhaseDixon(dixonImages[0], vectorRadius=(2, 2, 2))

    # Apply mask:
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    maskFilter.SetOutsideValue(0)
    segmentedImage = maskFilter.Execute(segmentedImage, sitk.Not(maskBody))
    return segmentedImage


#Ahora la tengo que implementar

for filenameInDir in os.listdir(targetPath):
    dixonImages = []
    name, extension = os.path.splitext(filenameInDir)
    # Si hay casos específicos o procesamos todos
    if (len(casesToSegment) == 0) or (name in casesToSegment):
        # Construimos la ruta completa
        dataPath = os.path.join(targetPath, filenameInDir)
        if os.path.isdir(dataPath):
            dataPath += "/"  # Aseguramos el separador final si es directorio

            # Archivos que verificamos (ajustamos las rutas para no usar subFolder)
            filename = os.path.join(dataPath, f"{name}{inPhaseSuffix}.{extensionImages}")
            outFilenameSegmented = os.path.join(dataPath, f"{name}{suffixSegmentedImages}.{extensionImages}")
            outFilenameFatFraction = os.path.join(dataPath, f"{name}{suffixFatFractionImages}.{extensionImages}")

            # Verificamos si procesamos o no
            if OVERWRITE_EXISTING_SEGMENTATIONS or (not os.path.exists(outFilenameSegmented) and not os.path.exists(outFilenameFatFraction)):
                if os.path.exists(filename):
                    print(f'Image to be processed: {name}\n')

                    # Procesamos las imágenes Dixon en orden
                    for suffix in dixonSuffixInOrder:
                        dixonFile = os.path.join(dataPath, f"{name}{suffix}.{extensionImages}")
                        if os.path.exists(dixonFile):
                            dixonImages.append(sitk.Cast(sitk.ReadImage(dixonFile), sitk.sitkFloat32))

                    # Segmentación de tejidos
                    segmentedImage = DixonTissueSegmentation(dixonImages)
                    sitk.WriteImage(segmentedImage, outFilenameSegmented, True)

                    # Máscara del cuerpo
                    bodyMask = GetBodyMaskFromFatDixonImage(dixonImages[3], vectorRadius=(2, 2, 1))
                    bodyMaskFile = os.path.join(dataPath, f"{name}{suffixBodyImages}.{extensionImages}")
                    sitk.WriteImage(bodyMask, bodyMaskFile, True)

                    # Máscara de grasa cutánea
                    skinFat = GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(segmentedImage)
                    skinFat = sitk.And(skinFat, bodyMask)
                    skinFatFile = os.path.join(dataPath, f"{name}{suffixSkinFatImages}.{extensionImages}")
                    sitk.WriteImage(skinFat, skinFatFile, True)

#Código para crear GIFS de las segmentaciones de tejidos

# Ruta de entrada y salida
input_folder = '/home/facundo/data/Nepal/DIXON/'
output_folderTSG = outputPath + 'TissueSegmentationGifs/'

#Si no existen, los crea
if not os.path.exists(output_folderTSG):
    os.makedirs(output_folderTSG)

# Función para encontrar y cargar segmentaciones de tejidos
def load_tissue_segmentations(folder):
    segmentations = {}
    for root, _, files in os.walk(folder):  # Recorrer todas las subcarpetas
        print(f"Explorando carpeta: {root}")  # Para depuración
        for file in files:
            if file.endswith("_tissue_segmented.mhd"):  # Corrige el sufijo según los archivos
                filepath = os.path.join(root, file)
                print(f"Archivo encontrado: {filepath}")  # Para depuración
                try:
                    # Leer la imagen con SimpleITK
                    image = sitk.ReadImage(filepath)
                    array = sitk.GetArrayFromImage(image)  # Array 3D
                    # Usar la carpeta del sujeto como parte del identificador
                    subject_name = os.path.basename(root)
                    key = f"{subject_name}_{file.replace('_tissue_segmented.mhd', '')}"
                    key = key.lstrip('_')  # Elimina el guion bajo al principio si existe
                    segmentations[key] = array
                except Exception as e:
                    print(f"Error al leer {filepath}: {e}")
    print(f"Total de archivos encontrados: {len(segmentations)}")
    return segmentations

# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animationTS.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

# Ejecutar el flujo
segmentations = load_tissue_segmentations(input_folder)
create_gifs(segmentations, output_folderTSG)

"""