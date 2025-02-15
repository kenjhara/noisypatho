
def mshow(mask, ax):
    """
    Display a PIL mask in grayscale.

    Args:
        mask (PIL.Image): Mask image (0-255).
        ax (matplotlib.axes.Axes): Matplotlib Axes for displaying the image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the mask displayed.
    """
    ax.imshow(mask, vmin=0, vmax=255, cmap='gray')
    ax.axis("off")
    return ax


def ishow(img, ax):
    """
    Display a PIL image.

    Args:
        img (PIL.Image): Image to display.
        ax (matplotlib.axes.Axes): Matplotlib Axes for displaying the image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the image displayed.
    """
    ax.imshow(img)
    ax.axis("off")
    return ax


def overlay(img, mask, ax):
    """
    Overlay a grayscale mask on a PIL image with partial transparency.

    Args:
        img (PIL.Image): Base image.
        mask (PIL.Image): Mask image (0-255).
        ax (matplotlib.axes.Axes): Matplotlib Axes for displaying the result.

    Returns:
        matplotlib.axes.Axes: The Axes object with the overlay displayed.
    """
    ax.imshow(img)
    ax.imshow(mask, cmap='gray', alpha=0.5)
    ax.axis("off")
    return ax
