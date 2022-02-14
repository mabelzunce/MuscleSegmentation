#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.
from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SimpleITK as sitk
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import SitkImageManipulation as sitkIm
import winshell
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# DATA FOLDERS:
case = "107"
basePath = "D:\Martin\ImplantMigrationStudy\\" + case + "\\"
postopImageNames = basePath + case + '_Migration_ContralateralPostopHemiPelvis.mhd'
followupImageNames = basePath + case + '_Migration_ContralateralFollowupHemiPelvis.mhd'

#postopImageNames = basePath + case + '_Migration_PostopPelvis.mhd'
#followupImageNames = basePath + case + '_Migration_FollowupPelvis.mhd'

#postopImageNames = basePath + case + '_Migration_PostopBone.mhd'
#followupImageNames = basePath + case + '_Migration_FollowupBone.mhd'

# READ DATA
postopImage = sitk.ReadImage(postopImageNames) # This will be the reference
followupImage = sitk.ReadImage(followupImageNames) # This will be the segmented

# BINARIZE THE IMAGES:
postopImage = sitk.Greater(postopImage, 0)
followupImage = sitk.Greater(followupImage, 0)

# HOW OVERLAP IMAGES
slice_number = round(postopImage.GetSize()[1]/2)
#DisplayWithOverlay(image, segmented, slice_number, window_min, window_max)
sitkIm.DisplayWithOverlay(postopImage[:,slice_number,:], followupImage[:,slice_number,:], 0, 1)
#interact(sitkIm.DisplayWithOverlay, slice_number = (5), image = fixed(postopImage), segmented = fixed(followupImage),
#          window_min = fixed(0), window_max=fixed(1));

# Get the image constrained by both bounding boxes:
#labelStatisticFilter = sitk.LabelShapeStatisticsImageFilter()
#labelStatisticFilter.Execute(postopImage)
#postopBoundingBox = np.array(labelStatisticFilter.GetBoundingBox(1))
#labelStatisticFilter.Execute(followupImage)
#followupBoundingBox = np.array(labelStatisticFilter.GetBoundingBox(1))
#minimumStart = np.minimum(postopBoundingBox[0:3], followupBoundingBox[0:3]+ 20)  # 50 is to give an extra margin
#minimumStop = np.minimum(postopBoundingBox[0:3]+postopBoundingBox[3:6], followupBoundingBox[0:3]+followupBoundingBox[3:6]- 20)
#minimumBoxSize = minimumStop - minimumStart
#postopImage = postopImage[minimumStart[0]:minimumStop[0], minimumStart[1]:minimumStop[1], minimumStart[2]:minimumStop[2]]
#followupImage = followupImage[minimumStart[0]:minimumStop[0], minimumStart[1]:minimumStop[1], minimumStart[2]:minimumStop[2]]

# Another approach is to get the bounding box of the intersection:
postopAndFollowupImage = sitk.And(postopImage, followupImage)
labelStatisticFilter = sitk.LabelShapeStatisticsImageFilter()
labelStatisticFilter.Execute(postopAndFollowupImage)
bothBoundingBox = np.array(labelStatisticFilter.GetBoundingBox(1))
postopImage = postopImage[bothBoundingBox[0]:bothBoundingBox[0]+bothBoundingBox[3],
              bothBoundingBox[1]:bothBoundingBox[1]+bothBoundingBox[4],
              bothBoundingBox[2]+20:bothBoundingBox[2]++bothBoundingBox[5]-20]
followupImage = followupImage[bothBoundingBox[0]:bothBoundingBox[0]+bothBoundingBox[3],
                bothBoundingBox[1]:bothBoundingBox[1]+bothBoundingBox[4],
                bothBoundingBox[2]+20:bothBoundingBox[2]+bothBoundingBox[5]-20]

#Display reduced image:
slice_number = round(postopImage.GetSize()[1]*1/3)
sitkIm.DisplayWithOverlay(postopImage[:,slice_number,:], followupImage[:,slice_number,:], 0, 1)
#sitk.Get
#postopZ = permute(sum(sum(postopImage))>0, [3 1 2]);
#followupZ = permute(sum(sum(followupImage))>0, [3 1 2]);
#bothZ = find(postopZ&followupZ > 0);
#% Remove 10 slices each side:
#bothZ(1:10) = []; bothZ(end-10:end) = [];

# GET SEGMENTATION PERFORMANCE BASED ON SURFACES:
# init signed mauerer distance as reference metrics
reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(postopImage, squaredDistance=False, useImageSpacing=True))

# Get the reference surface:
reference_surface = sitk.LabelContour(postopImage)
statistics_image_filter = sitk.StatisticsImageFilter()
# Get the number of pixels in the reference surface by counting all pixels that are 1.
statistics_image_filter.Execute(reference_surface)
num_reference_surface_pixels = int(statistics_image_filter.GetSum())

# Get the surface (contour) of the segmented image:
segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(followupImage, squaredDistance=False, useImageSpacing=True))
segmented_surface = sitk.LabelContour(followupImage)
# Get the number of pixels in the reference surface by counting all pixels that are 1.
statistics_image_filter.Execute(segmented_surface)
num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)

# Hausdorff distance:
hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
hausdorff_distance_filter.Execute(postopImage, followupImage)

#All the other metrics:
# Multiply the binary surface segmentations with the distance maps. The resulting distance
# maps contain non-zero values only on the surface (they can also contain zero on the surface)
seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)
# Get all non-zero distances and then add zero distances if required.
seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
seg2ref_distances = seg2ref_distances + \
                    list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
ref2seg_distances = ref2seg_distances + \
                    list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
all_surface_distances = seg2ref_distances + ref2seg_distances

# The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
# general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
# segmentations, though in our case it is. More on this below.
#hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()
#max_surface_distance = label_intensity_statistics_filter.GetMaximum(1)
#avg_surface_distance = label_intensity_statistics_filter.GetMean(1)
#median_surface_distance = label_intensity_statistics_filter.GetMedian(1)
#std_surface_distance = label_intensity_statistics_filter.GetStandardDeviation(1)
hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()
avg_surface_distance = np.mean(all_surface_distances)
max_surface_distance = np.max(all_surface_distances)
median_surface_distance = np.median(all_surface_distances)
std_surface_distance = np.std(all_surface_distances)
# Now in mm:
hausdorff_distance_mm = hausdorff_distance * postopImage.GetSpacing()[0]
avg_surface_distance_mm = avg_surface_distance * postopImage.GetSpacing()[0]
max_surface_distance_mm = max_surface_distance * postopImage.GetSpacing()[0]
median_surface_distance_mm = median_surface_distance * postopImage.GetSpacing()[0]
std_surface_distance_mm = std_surface_distance * postopImage.GetSpacing()[0]

print("Surface based metrics [voxels]: MEAN_SD={0}, STDSD={1}, MEDIAN_SD={2}, HD={3}, MAX_SD={4}\n".format(avg_surface_distance, std_surface_distance, median_surface_distance, hausdorff_distance, max_surface_distance))
print("Surface based metrics [mm]: MEAN_SD={0}, STDSD={1}, MEDIAN_SD={2}, HD={3}, MAX_SD={4}\n".format(avg_surface_distance_mm, std_surface_distance_mm, median_surface_distance_mm, hausdorff_distance_mm, max_surface_distance_mm))
# GET SEGMENTATION PERFORMANCE BASED ON OVERLAP METRICS:
overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
overlap_measures_filter.Execute(postopImage, followupImage)
jaccard_value = overlap_measures_filter.GetJaccardCoefficient()
dice_value = overlap_measures_filter.GetDiceCoefficient()
volume_similarity_value = overlap_measures_filter.GetVolumeSimilarity()
false_negative_value = overlap_measures_filter.GetFalseNegativeError()
false_positive_value = overlap_measures_filter.GetFalsePositiveError()

print("Overlap based metrics: Jaccard={0}, Dice={1}, VolumeSimilarity={2}, FN={3}, FP={4}\n".format(jaccard_value, dice_value, volume_similarity_value, false_negative_value, false_positive_value))

# Create a log file:
logFilename = basePath + 'RegistrationPerformance_python.txt'
log = open(logFilename, 'w')
log.write("Mean Surface Distance, STD Surface Distance, Median Surface Distance, Hausdorff Distance, Max Surface Distance\n")
log.write("{0}, {1}, {2}, {3}, {4}\n".format(avg_surface_distance, std_surface_distance, median_surface_distance, hausdorff_distance, max_surface_distance))
log.write("Mean Surface Distance, STD Surface Distance [mm], Median Surface Distance [mm], Hausdorff Distance [mm], Max Surface Distance [mm]\n")
log.write("{0}, {1}, {2}, {3}, {4}\n".format(avg_surface_distance_mm, std_surface_distance_mm, median_surface_distance_mm, hausdorff_distance_mm, max_surface_distance_mm))
log.write("Jaccard, Dice, Volume Similarity, False Negative, False Positive\n")
log.write("{0}, {1}, {2}, {3}, {4}\n".format(jaccard_value, dice_value, volume_similarity_value, false_negative_value, false_positive_value))
log.close()
plt.show()