import numpy as np
import random
import math
from decimal import Decimal, ROUND_HALF_UP
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, label
from . import BCSS, utils
import matplotlib.pyplot as plt
from scipy import ndimage


# Noise Modules


def noise_parameters(num, level, noise_types, seed=2023):
    """
    Generate a list of parameter dictionaries for specified noise types.

    Args:
        num (int): Number of parameter sets to generate.
        level (int): Noise level (1, 2, or 3).
        noise_types (list): List of noise types (e.g., ["dilation", "omission", "shrink", "additive"]).
        seed (int, optional): Random seed. Defaults to 2023.

    Returns:
        list: A list of dictionaries containing parameters for each noise type.
    """
    np.random.seed(seed)
    params_list = []

    if "dilation" in noise_types:
        if level == 1:
            iteration_range = (10, 30)
            epsilon_range = (10, 20)
        elif level == 2:
            iteration_range = (30, 50)
            epsilon_range = (20, 30)
        elif level == 3:
            iteration_range = (50, 150)
            epsilon_range = (30, 50)
        else:
            print("error: level should be 1,2, or 3")

    if "omission" in noise_types:
        if level == 1:
            min_area_threshold_range = (10000, 50000)
        elif level == 2:
            min_area_threshold_range = (50000, 100000)
        elif level == 3:
            min_area_threshold_range = (100000, 200000)
        else:
            print("error: level should be 1,2, or 3")

    if "shrink" in noise_types:
        if level == 1:
            repeat_range = (1, 5)
        elif level == 2:
            repeat_range = (5, 9)
        elif level == 3:
            repeat_range = (9, 13)
        else:
            print("error: level should be 1,2, or 3")

    if "additive" in noise_types:
        if level == 1:
            num_region_range = (3, 8)
            areas_range = (20000, 60000)
        elif level == 2:
            num_region_range = (3, 8)
            areas_range = (60000, 120000)
        elif level == 3:
            num_region_range = (5, 8)
            areas_range = (120000, 160000)
        else:
            print("error: level should be 1,2, or 3")
        shape_choices = ["circle", "ellipse", "polygon"]

    for i in range(num):
        para = {}
        if "dilation" in noise_types:
            para["iteration"] = np.random.randint(*iteration_range)
            para["epsilon"] = np.random.randint(*epsilon_range)
        if "omission" in noise_types:
            para["min_area_threshold"] = np.random.randint(*min_area_threshold_range)
        if "shrink" in noise_types:
            para["repeat"] = np.random.randint(*repeat_range)
        if "additive" in noise_types:
            n = np.random.randint(*num_region_range)
            para["num_regions"] = n
            para["areas"] = [np.random.randint(*areas_range) for _ in range(n)]
            random_indices = np.random.choice(len(shape_choices), n, replace=True)
            para["shapes"] = [shape_choices[idx] for idx in random_indices]
        params_list.append(para)

    return params_list


def dilate_regions(mask, iterations):
    """
    Dilate the given binary mask.

    Args:
        mask (PIL.Image): Binary mask.
        iterations (int): Number of dilation iterations.

    Returns:
        PIL.Image: Dilated binary mask.
    """
    mask_array = np.array(mask)
    binary_mask = np.where(mask_array > 0, 1, 0)
    dilated_mask = binary_dilation(binary_mask, iterations=iterations)
    dilated_pil = Image.fromarray((dilated_mask * 255).astype(np.uint8))
    return dilated_pil


def smooth_contour(mask, epsilon):
    """
    Smooth the contour of the given binary mask using OpenCV.

    Args:
        mask (PIL.Image): Binary mask.
        epsilon (float): Smoothing parameter (larger values result in coarser smoothing).

    Returns:
        PIL.Image: Smoothed binary mask.
    """
    mask_array = np.array(mask)
    mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
    _, binary_mask = cv2.threshold(mask_cv, 1, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    smoothed_mask = np.zeros_like(binary_mask)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(smoothed_mask, [approx], 0, 255, thickness=cv2.FILLED)

    smoothed_pil = Image.fromarray(smoothed_mask)
    return smoothed_pil


def shrink_mask(mask, repeat):
    """
    Iteratively shrink the binary mask.

    Args:
        mask (PIL.Image): Binary mask.
        repeat (int): Number of times to repeat the shrink process.

    Returns:
        PIL.Image: Shrunk binary mask.
    """
    iterations = 4
    min_area = 10000
    shrunk_pil = mask

    # First minimal shrink to avoid issues in small connected regions
    mask_array = np.asarray(shrunk_pil)
    binary_shrunk_mask = np.where(mask_array > 0, 1, 0)
    shrunk_mask = binary_erosion(binary_shrunk_mask, iterations=1)
    shrunk_pil = Image.fromarray((shrunk_mask * 255).astype(np.uint8))

    # Repeat shrink based on the given 'repeat' value
    for i in range(repeat):
        mask_array = np.asarray(shrunk_pil)
        mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
        _, binary_mask = cv2.threshold(mask_cv, 1, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

        # Remove regions below the specified min_area
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        for label_val in range(1, labels.max() + 1):
            area = stats[label_val, cv2.CC_STAT_AREA]
            if area <= min_area:
                binary_mask[labels == label_val] = 0

        binary_mask = Image.fromarray(binary_mask)
        binary_shrunk_mask = np.where(mask_array > 0, 1, 0)
        shrunk_mask = binary_erosion(binary_shrunk_mask, iterations=iterations, mask=np.array(binary_mask))
        shrunk_pil = Image.fromarray((shrunk_mask * 255).astype(np.uint8))

    return shrunk_pil


def remove_small_regions(mask, min_area):
    """
    Remove white regions in a mask smaller than the specified area.

    Args:
        mask (PIL.Image): Binary mask.
        min_area (int): Maximum area threshold below which regions are removed.

    Returns:
        PIL.Image: Binary mask with small regions removed.
    """
    mask_array = np.array(mask)
    mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
    _, binary_mask = cv2.threshold(mask_cv, 1, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    for label_val in range(1, labels.max() + 1):
        area = stats[label_val, cv2.CC_STAT_AREA]
        if area <= min_area:
            binary_mask[labels == label_val] = 0

    binary_mask = Image.fromarray(binary_mask)
    return binary_mask


def additive_mask(mask, mask_with_out, num_region, areas, shapes):
    """
    Insert artificial positive regions into a binary mask in areas that are initially negative.

    Args:
        mask (PIL.Image): Original binary mask.
        mask_with_out (PIL.Image): Binary mask used to confirm negative regions.
        num_region (int): Number of new regions to add.
        areas (list): List of areas for each region.
        shapes (list): List of shape types (e.g., "circle", "ellipse", "polygon").

    Returns:
        PIL.Image: Modified mask with added regions.
    """
    width, height = mask.size
    mask_copy = mask.copy()
    draw = ImageDraw.Draw(mask_copy)

    count = 0
    try_error = 0

    while count < num_region:
        current_shape = shapes[count]
        if current_shape == "circle":
            try:
                size_0 = math.sqrt(areas[count])
                size_0 = Decimal(str(size_0))
                size_0 = size_1 = int(size_0.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                x = random.randint(0, width - size_0)
                y = random.randint(0, height - size_1)

                # Confirm that the region is negative
                is_negative_region = all(
                    mask_copy.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                ) and all(
                    mask_with_out.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                )

                if is_negative_region:
                    draw.ellipse([x, y, x + size_0, y + size_1], fill=255)
                    try_error = 0
                    count += 1
                else:
                    try_error += 1
            except Exception:
                try_error += 1

            if try_error >= 500:
                print("try_error: More than 500 errors for circle insertion, skipping.")
                try_error = 0
                count += 1

        elif current_shape == "ellipse":
            try:
                size_0 = math.sqrt(areas[count])
                size_0 = Decimal(str(size_0))
                size_0 = int(size_0.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                size_0 = random.randint(max(0, size_0 - 200), min(width, size_0 + 200))
                size_1 = Decimal(str(areas[count] / size_0))
                size_1 = int(size_1.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                x = random.randint(0, width - size_0)
                y = random.randint(0, height - size_1)

                # Confirm that the region is negative
                is_negative_region = all(
                    mask_copy.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                ) and all(
                    mask_with_out.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                )

                if is_negative_region:
                    draw.ellipse([x, y, x + size_0, y + size_1], fill=255)
                    count += 1
                    try_error = 0
                else:
                    try_error += 1
            except Exception:
                try_error += 1

            if try_error >= 500:
                print("try_error: More than 500 errors for ellipse insertion, skipping.")
                try_error = 0
                count += 1

        elif current_shape == "polygon":
            try:
                size_0 = math.sqrt(areas[count])
                size_0 = Decimal(str(size_0))
                size_0 = size_1 = int(size_0.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                x = random.randint(0, width - size_0)
                y = random.randint(0, height - size_1)

                # Confirm that the region is negative
                is_negative_region = all(
                    mask_copy.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                ) and all(
                    mask_with_out.getpixel((i, j)) == 0 for i in range(x, x + size_0) for j in range(y, y + size_1)
                )

                if is_negative_region:
                    # Draw a random polygon
                    sides = random.randint(5, 8)
                    angle = 360 / sides
                    points = []
                    for i_side in range(sides):
                        x_point = x + size_0 // 2 + size_0 // 2 * math.cos(math.radians(i_side * angle))
                        y_point = y + size_1 // 2 + size_1 // 2 * math.sin(math.radians(i_side * angle))
                        points.append((x_point, y_point))

                    draw.polygon(points, fill=255)
                    count += 1
                    try_error = 0
                else:
                    try_error += 1
            except Exception:
                try_error += 1

            if try_error >= 500:
                print("try_error: More than 500 errors for polygon insertion, skipping.")
                try_error = 0
                count += 1

    return mask_copy


def omission_mask(mask, min_area):
    """
    Remove white regions in a mask smaller than the specified area.

    Args:
        mask (PIL.Image): Binary mask.
        min_area (int): Maximum area threshold below which regions are removed.

    Returns:
        PIL.Image: Binary mask with small regions removed.
    """
    mask_array = np.array(mask)
    mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
    _, binary_mask = cv2.threshold(mask_cv, 1, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    for label_val in range(1, labels.max() + 1):
        area = stats[label_val, cv2.CC_STAT_AREA]
        if area <= min_area:
            binary_mask[labels == label_val] = 0
    binary_mask = Image.fromarray(binary_mask)
    return binary_mask


def restore_one_regions(original_mask, edited_mask):
    """
    Restore pixels that are positive (non-zero) in the original_mask to the edited_mask.

    Args:
        original_mask (PIL.Image): Original binary mask.
        edited_mask (PIL.Image): Edited binary mask.

    Returns:
        PIL.Image: Edited mask with positive regions from the original restored.
    """
    original_array = np.array(original_mask)
    edited_array = np.array(edited_mask)
    one_regions = np.where(original_array > 0, 255, 0)
    restored_array = np.where(one_regions == 255, 255, edited_array)
    restored_mask = Image.fromarray(restored_array.astype(np.uint8))
    return restored_mask


# Noise Model


def PREPROCESS(mask):
    """
    Preprocess function placeholder. Returns the mask as is.
    """
    result = mask
    return result


def MAKE_DILATION_NOISE(edited_mask, list_noise, BCSS_mask=None):
    """
    Apply dilation noise to a mask.

    Args:
        edited_mask (PIL.Image): Binary mask to edit.
        list_noise (dict): Dictionary containing "iteration" and "epsilon".
        BCSS_mask (PIL.Image, optional): Used to restore zero-value regions. Defaults to None.

    Returns:
        PIL.Image: Mask with dilation noise.
    """
    result = dilate_regions(edited_mask, iterations=list_noise["iteration"])
    result = smooth_contour(result, epsilon=list_noise["epsilon"])
    if BCSS_mask is not None:
        result = BCSS.restore_zero_regions(BCSS_mask, result)
    result = restore_one_regions(edited_mask, result)
    return result


def MAKE_OMISSION_NOISE(edited_mask, list_noise, BCSS_mask=None):
    """
    Apply omission noise to a mask (remove small regions).

    Args:
        edited_mask (PIL.Image): Binary mask to edit.
        list_noise (dict): Dictionary containing "min_area_threshold".
        BCSS_mask (PIL.Image, optional): Used to restore zero-value regions. Defaults to None.

    Returns:
        PIL.Image: Mask with omission noise.
    """
    result = remove_small_regions(edited_mask, list_noise["min_area_threshold"])
    if BCSS_mask is not None:
        result = BCSS.restore_zero_regions(BCSS_mask, result)
    return result


def MAKE_SHRINK_NOISE(edited_mask, list_noise, BCSS_mask=None):
    """
    Apply shrink noise to a mask.

    Args:
        edited_mask (PIL.Image): Binary mask to edit.
        list_noise (dict): Dictionary containing "repeat" times of shrink.
        BCSS_mask (PIL.Image, optional): Used to restore zero-value regions. Defaults to None.

    Returns:
        PIL.Image: Mask with shrink noise.
    """
    result = shrink_mask(edited_mask, list_noise["repeat"])
    if BCSS_mask is not None:
        result = BCSS.restore_zero_regions(BCSS_mask, result)
    return result


def MAKE_ADDITIVE_NOISE(edited_mask, list_noise, BCSS_mask=None):
    """
    Apply additive noise by inserting new positive regions.

    Args:
        edited_mask (PIL.Image): Binary mask to edit.
        list_noise (dict): Dictionary with "num_regions", "areas", and "shapes".
        BCSS_mask (PIL.Image, optional): Mask to differentiate between positive and outside ROI. Defaults to None.

    Returns:
        PIL.Image: Mask with additive noise.
    """
    if BCSS_mask is not None:
        positive_and_out = BCSS.convert_mask_a_as_b(BCSS_mask, a=[0, 19, 20], b=1)
        positive_and_out = BCSS.binarize_mask(positive_and_out, target=1)
    else:
        positive_and_out = edited_mask

    result = additive_mask(
        edited_mask,
        positive_and_out,
        num_region=list_noise["num_regions"],
        areas=list_noise["areas"],
        shapes=list_noise["shapes"]
    )

    if BCSS_mask is not None:
        result = BCSS.restore_zero_regions(BCSS_mask, result)
    return result


# Noise Visualization


def convert(gt_mask, noise_mask):
    """
    Create a visualization comparing ground truth and noisy masks.
    Overlapping, missing, and added regions are color-coded.
    """
    gt_mask = np.array(gt_mask)
    noise_mask = np.array(noise_mask)

    gt_mask = np.argmax(gt_mask, axis=0) if len(gt_mask.shape) > 2 else gt_mask
    noise_mask = np.argmax(noise_mask, axis=0) if len(noise_mask.shape) > 2 else noise_mask
    gt_mask_bin = gt_mask > 0
    noise_mask_bin = noise_mask > 0

    display_image = np.ones((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8) * 255

    # Overlapping regions
    agree_regions = gt_mask_bin & noise_mask_bin
    display_image[agree_regions, 0] = 82
    display_image[agree_regions, 1] = 151
    display_image[agree_regions, 2] = 204

    # Missing regions (shrunk/removed)
    missing_regions = gt_mask_bin & (~noise_mask_bin)
    display_image[missing_regions, 0] = 255
    display_image[missing_regions, 1] = 253
    display_image[missing_regions, 2] = 200

    # Added regions (dilated/inserted)
    added_regions = (~gt_mask_bin) & noise_mask_bin
    display_image[added_regions, 0] = 20
    display_image[added_regions, 1] = 124
    display_image[added_regions, 2] = 204

    # Ground-truth edges
    dilated_gt = ndimage.binary_dilation(gt_mask_bin, iterations=5)
    gt_edges = np.logical_xor(dilated_gt, gt_mask_bin)

    # Color edges in red
    display_image[gt_edges, 0] = 255
    display_image[gt_edges, 1] = 40
    display_image[gt_edges, 2] = 40

    return display_image


def see_diff(img_he, img_mask, img_noisymask, figsize=(6, 1.5), border_width=5):
    """
    Display comparisons among the HE image, ground truth mask, noisy mask,
    and a difference map in a single figure.
    """
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    img_mask = np.array(img_mask)
    img_noisymask = np.array(img_noisymask)

    # HE
    ax[0].imshow(img_he)
    ax[0].axis("off")

    # Ground truth mask
    ax[1].imshow(img_mask, cmap="gray")
    ax[1].axis("off")

    # Noisy mask
    ax[2].imshow(img_noisymask, cmap="gray")
    ax[2].axis("off")

    # Difference map
    display_img = convert(img_mask, img_noisymask)
    ax[3].imshow(display_img)
    ax[3].axis("off")

    # Add a border around each subplot
    for i in range(4):
        for spine in ax[i].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(border_width)

    plt.show()
