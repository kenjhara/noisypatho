import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from PIL import Image, ImageColor

# Read metadata
meta = pd.read_csv("./example/meta/gt_codes.tsv", sep="\t", index_col=1)
gt_labels = {x: y for x, y in zip(meta.index, meta["label"])}
gt_colors = {x: [z / 255 for z in list(ImageColor.getcolor(y, "RGB"))] for x, y in zip(meta.index, meta["colors"])}


# Visualization (BCSS only)


def get_color(mask):
    """
    Extract unique labels from the given PIL mask, and retrieve their corresponding
    color values and label names based on 'gt_labels' and 'gt_colors'.

    Args:
        mask (PIL.Image): Mask image.

    Returns:
        tuple: (unique_label_array, list_of_label_names, list_of_colors)
    """
    mask_array = np.array(mask)
    unique_labels = np.unique(mask_array)
    colors_in_mask = [gt_colors[x] for x in unique_labels]
    labels_in_mask = [gt_labels[x] for x in unique_labels]
    return unique_labels, labels_in_mask, colors_in_mask


def mshow(mask, ax):
    """
    Display a PIL mask on the given matplotlib Axes object with a color legend.

    Args:
        mask (PIL.Image): Mask image.
        ax (matplotlib.axes.Axes): Matplotlib Axes to display the mask.

    Returns:
        matplotlib.axes.Axes: The Axes object with the mask displayed.
    """
    c_labels, labels_, colors_ = get_color(mask)
    cmap = ListedColormap(colors_)
    bounds = c_labels
    handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in colors_]
    ax.legend(handles, [gt_labels[x] for x in c_labels], fontsize=8)
    if len(c_labels) != 1:
        norm = BoundaryNorm(bounds, cmap.N - 1)
        ax.imshow(mask, cmap=cmap, norm=norm)
    else:
        ax.imshow(mask, cmap=cmap)
    ax.axis("off")
    return ax


def ishow(img, ax):
    """
    Display a PIL image on the given matplotlib Axes object.

    Args:
        img (PIL.Image): Image to display.
        ax (matplotlib.axes.Axes): Matplotlib Axes to display the image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the image displayed.
    """
    ax.imshow(img)
    ax.axis("off")
    return ax


def overlay(img, mask, ax):
    """
    Overlay a PIL mask on top of a PIL image with semi-transparency.

    Args:
        img (PIL.Image): Base image.
        mask (PIL.Image): Mask image.
        ax (matplotlib.axes.Axes): Matplotlib Axes to display the result.

    Returns:
        matplotlib.axes.Axes: The Axes object with the overlaid image.
    """
    c_labels, labels_, colors_ = get_color(mask)
    cmap = ListedColormap(colors_)
    bounds = c_labels
    norm = BoundaryNorm(bounds, cmap.N - 1)
    handles = [Rectangle((0, 0), 1, 1, color=_c, alpha=0.5) for _c in colors_]
    ax.legend(handles, [gt_labels[x] for x in c_labels], fontsize=8)
    ax.imshow(img)
    ax.imshow(mask, cmap=cmap, norm=norm, alpha=0.5)
    ax.axis("off")
    return ax


# Preprocessing (BCSS only)


def convert_mask_a_as_b(mask, a, b):
    """
    Convert specified label(s) 'a' in a PIL mask to another label 'b'.

    Args:
        mask (PIL.Image): Mask image.
        a (int or list): Label value(s) to convert.
        b (int): Label value to replace 'a'.

    Returns:
        PIL.Image: Mask with label(s) 'a' replaced by 'b'.
    """
    np_mask = np.array(mask)
    if isinstance(a, list):
        for t in a:
            np_mask = np.where(np_mask == t, b, np_mask)
    else:
        np_mask = np.where(np_mask == a, b, np_mask)

    converted_mask = Image.fromarray(np_mask.astype(np.uint8))
    return converted_mask


def binarize_mask(mask, target):
    """
    Binarize a PIL mask by marking the specified 'target' label as white (255)
    and all other labels as black (0).

    Args:
        mask (PIL.Image): Mask image.
        target (int): Label to be set to white.

    Returns:
        PIL.Image: Binarized mask image.
    """
    mask_gray = mask.convert("L")
    np_mask = np.array(mask_gray)
    binary_mask = np.where(np_mask == target, 1, 0)
    binary_mask_img = Image.fromarray((binary_mask.astype(np.uint8) * 255))
    return binary_mask_img


# Noise Utils (BCSS only)


def restore_zero_regions(original_mask, edited_mask):
    """
    Restore zero-valued regions (background) from the original mask to the edited mask.
    Regions that were originally 0 remain 0 in the final output.

    Args:
        original_mask (PIL.Image): The original mask image.
        edited_mask (PIL.Image): The edited mask image (e.g., after dilation or other processing).

    Returns:
        PIL.Image: The mask with original zero (background) regions restored.
    """
    original_array = np.array(original_mask)
    edited_array = np.array(edited_mask)
    zero_regions = np.where(original_array == 0, 1, 0)
    one_regions = np.where(edited_array > 0, 1, 0)
    restored_array = np.where(zero_regions & one_regions, 0, edited_array)
    restored_mask = Image.fromarray(restored_array.astype(np.uint8))
    return restored_mask
