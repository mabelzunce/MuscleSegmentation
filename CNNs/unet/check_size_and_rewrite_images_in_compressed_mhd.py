#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
# Optional to write the images, in case they are already compressed:
write_images = True

dataPath = '/home/martin/data_imaging/Muscle/LumbarSpine/ManualSegmentations/MhdsResampled/'# Base data path.
outputPath = '/home/martin/data_imaging/Muscle/LumbarSpine/ManualSegmentations/MhdCompressed/'# Base data path.
#dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/Raw/1stPhase/C00011/'# Base data path.
#outputPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/C00011/'# Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

correctNames = True # If in phase names have an _I, remove it

# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_I'
tagLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images

# Write a CSV
csv_file = os.path.join(outputPath, "image_sizes.csv")
f=open(csv_file, mode='a', newline='')
writer = csv.writer(f)
writer.writerow(["filename", "voxel_size_x", "voxel_size_y", "voxel_size_z",
            "size_voxels_x", "size_voxels_y", "size_voxels_z",
            "size_mm_x", "size_mm_y", "size_mm_z"])
            
folderIndex = []
tagArray = []
#data = data[20:]
max_fov_x = 0
max_fov_x_filename = ""
for filename in data:
    name, extension = os.path.splitext(filename)
    if extension == extensionImages:
        # Read image and write it again:
        image = sitk.ReadImage(dataPath + filename)
        size_voxels = image.GetSize()
        spacing_mm = image.GetSpacing()
        size_mm = tuple(s * sp for s, sp in zip(size_voxels, spacing_mm))
        print(f"{filename}: voxel size={spacing_mm}, size_voxels={size_voxels}, size_mm={size_mm}")

        # Write to CSV

        writer.writerow([
            filename,
            spacing_mm[0], spacing_mm[1], spacing_mm[2],
            size_voxels[0], size_voxels[1], size_voxels[2],
            size_mm[0], size_mm[1], size_mm[2]
            ])
        if name.endswith(tagLabels):
            outputName = name
        elif name.endswith(tagInPhase):
            outputName = name[:-len(tagInPhase)]
        else:
            outputName = name # in phase
        if write_images:
            sitk.WriteImage(image, outputPath + outputName + extensionImages, True)
        # print("Written image: {0}".format(outputPath + filename))

        if size_mm[0] > max_fov_x:
            max_fov_x = size_mm[0]
            max_fov_x_filename = filename
print(f"Filename with maximum FOV in x: {max_fov_x_filename}: max_fov_x={max_fov_x}")
f.close()


