#! python3

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Gets all the overlap segmentation metrics between a reference and segmented image.
# It receives both images, and optionally the number of labels in the image. If the number
# of labels is 0, it computes the overall metric considering every voxel greater than zero as the label.
# It returns a dictionary with the metrics.
def GetOverlapMetrics(reference, segmented, numLabels):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)

    # If numLabels 0, get a general metric:
    jaccard = []; dice = []; volumeSimilarity = []; fn = []; fp = []; tn = []; tp = []
    sensitivity = []; specificity = []; precision = []; fallout = []
    if numLabels == 0:
        overlap_measures_filter.Execute(reference, segmented)
        jaccard.append(overlap_measures_filter.GetJaccardCoefficient())
        dice.append(overlap_measures_filter.GetDiceCoefficient())
        volumeSimilarity.append(overlap_measures_filter.GetVolumeSimilarity())
        fn.append(overlap_measures_filter.GetFalseNegativeError())
        fp.append(overlap_measures_filter.GetFalsePositiveError())
        tp.append(overlap_measures_filter.GetMeanOverlap())
        tn.append(np.prod(reference.GetSize()) - overlap_measures_filter.GetUnionOverlap())
        sensitivity.append(tp[0]/(tp[0] + fn[0]))
        specificity.append(tn[0] /(tn[0] + fp[0]))
        precision.append(tp[0] /(tp[0] + fp[0]))
        fallout.append(fp[0] / (fp[0] + tn[0]))
    else:
        for i in range(0,numLabels):
            # First execute the filter:
            overlap_measures_filter.Execute(reference==(i+1), segmented==(i+1))
            jaccard.append(overlap_measures_filter.GetJaccardCoefficient())
            dice.append(overlap_measures_filter.GetDiceCoefficient())
            volumeSimilarity.append(overlap_measures_filter.GetVolumeSimilarity())
            fn.append(overlap_measures_filter.GetFalseNegativeError())
            fp.append(overlap_measures_filter.GetFalsePositiveError())
            tp.append(overlap_measures_filter.GetMeanOverlap())
            tn.append(np.prod(reference.GetSize()) - overlap_measures_filter.GetUnionOverlap())
            sensitivity.append(tp[i] / (tp[i] + fn[i]))
            specificity.append(tn[i] / (tn[i] + fp[i]))
            precision.append(tp[i] / (tp[i] + fp[i]))
            fallout.append(fp[i] / (fp[i] + tn[i]))

    metrics = {'dice':np.array(dice), 'jaccard': np.array(jaccard), 'volumeSimilarity':np.array(volumeSimilarity),
               'fn':np.array(fn), 'fp':np.array(fp), 'tp':np.array(tp), 'tn':np.array(tn), 'sensitivity':np.array(sensitivity), 'specificity':np.array(specificity),
               'precision':np.array(precision), 'fallout':np.array(fallout)}
    return metrics


# Gets surface based metrics between a reference and a segmented image.
# If numLabels = 0, all the labels are treated as one and single value per metric is returned.
# If numLabels > 0, a metric per label is returned.
def GetSurfaceMetrics(reference, segmented, numLabels = 0):

    if numLabels == 0:
        reference = reference > 0
        segmented = segmented > 0
        numLabels = 1

    hausdorff_distance_mm = []
    avg_surface_distance_mm = []
    max_surface_distance_mm = []
    median_surface_distance_mm = []
    std_surface_distance_mm = []

    for i in range(0,numLabels):
        referenceLabel = reference == i+1
        segmentedLabel = segmented == i+1
        # init signed mauerer distance for the reference:
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(referenceLabel, squaredDistance=False, useImageSpacing=True))

        # Get the reference surface:
        reference_surface = sitk.LabelContour(referenceLabel)
        statistics_image_filter = sitk.StatisticsImageFilter()
        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())

        # Get the surface (contour) of the segmented image:
        segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(segmentedLabel, squaredDistance=False, useImageSpacing=True))
        segmented_surface = sitk.LabelContour(segmentedLabel)
        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(segmented_surface)
        num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
        label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
        label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)

        # Hausdorff distance:
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(referenceLabel, segmentedLabel)

        # All the other metrics:
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

        hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()
        avg_surface_distance = np.mean(all_surface_distances)
        max_surface_distance = np.max(all_surface_distances)
        median_surface_distance = np.median(all_surface_distances)
        std_surface_distance = np.std(all_surface_distances)
        # Now in mm:
        hausdorff_distance_mm.append(hausdorff_distance * reference.GetSpacing()[0])
        avg_surface_distance_mm.append(avg_surface_distance * reference.GetSpacing()[0])
        max_surface_distance_mm.append(max_surface_distance * reference.GetSpacing()[0]) #Maximum should be the same as hausdorff.
        median_surface_distance_mm.append(median_surface_distance * reference.GetSpacing()[0])
        std_surface_distance_mm.append(std_surface_distance * reference.GetSpacing()[0])

    metrics = {'hausdorff_distance_mm': hausdorff_distance_mm, 'avg_surface_distance_mm':avg_surface_distance_mm,
               'median_surface_distance_mm': median_surface_distance_mm, 'std_surface_distance_mm': std_surface_distance_mm,
               'max_surface_distance_mm': max_surface_distance_mm}
    return metrics