"""Drawing tools for bounding boxes and segmentation masks."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from .logger import get_logger

logger = get_logger(__name__)


def _get_colormap(
    num_colors: int, colors: np.ndarray | None = None, colormap: str = "tab10"
) -> list[tuple[int, int, int, int]]:
    """Get a list of colors to use for visualization.

    Args:
        num_colors (int): Number of colors to generate.
        colors (np.ndarray | None, optional): NumPy array of mask colors to use with shape (N, 4) or (N, 3). If None,
            colors will be generated based on the `colormap`. Defaults to None.
        colormap (str, optional): Matplotlib colormap name for mask colors to use if `colors` is None.
            Defaults to "tab10".

    Returns:
        list[tuple[int, int, int, int]]: List of colors to use for visualization.

    """
    if colormap in [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ]:
        cmap = plt.get_cmap(colormap)
        if num_colors > cmap.N:
            logger.warning(
                'Colormap "%s" only has %d colors, but %d colors were requested.', colormap, cmap.N, num_colors
            )
        colors = cmap(np.arange(num_colors)) if colors is None else colors
    else:
        cmap = plt.get_cmap(colormap, num_colors + 2)
        colors = cmap(np.arange(num_colors + 1))[1:] if colors is None else colors

    return [tuple(color) for color in ((colors * 255).astype(int) if colors.max() <= 1.0 else colors).tolist()]


def draw_bboxes(
    image: Image.Image, bboxes: np.ndarray | None, colors: np.ndarray | None = None, colormap: str = "tab10"
) -> dict[str, Image.Image | list[Image.Image]]:
    """Draw bounding boxes on image and return bounding box crops.

    Args:
        image (PIL.Image.Image): Input PIL Image with size (W, H)
        bboxes (np.ndarray | None): NumPy array of bounding boxes (x1, y1, x2, y2) with shape (N, 4)
        colors (np.ndarray | None, optional): NumPy array of mask colors to use with shape (N, 4) or (N, 3). If None,
            colors will be generated based on the `colormap`. Defaults to None.
        colormap (str, optional): Matplotlib colormap name for mask colors to use if `colors` is None.
            Defaults to "tab10".

    Returns:
        dict[str, Image.Image | list[Image.Image]]: A dictionary containing the original image with bounding boxes drawn
            and a list of cropped images for each bounding box.

    """
    full_image = image.copy()

    if bboxes is None:
        return image.copy(), []

    # Create a copy of the original image for drawing
    draw = ImageDraw.Draw(full_image)

    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        msg = f"Expected 2D tensor of shape (N, 4) for bboxes, got {bboxes.shape}"
        raise ValueError(msg)

    # Generate colors from colormap
    colors = _get_colormap(len(bboxes), colors, colormap)

    # Get bbox width based on image size
    img_width, img_height = full_image.size
    bbox_width = max(1, min(img_width, img_height) // 250)

    crops = []
    for i, bbox in enumerate(bboxes):
        draw.rectangle(list(map(int, bbox)), outline=colors[i], width=bbox_width)
        crops.append(image.crop(list(map(int, bbox))))

    return {"image": full_image, "cropped_bboxes": crops}


def draw_masks(
    image: Image.Image,
    masks: np.ndarray | None,
    colors: np.ndarray | None = None,
    colormap: str = "tab10",
    transparency: float = 0.5,
) -> dict[str, Image.Image]:
    """Draw masks on image and remove background.

    Args:
        image (PIL.Image.Image): Input PIL Image with size (W, H)
        masks (np.ndarray | None): NumPy array of segmentation masks with shape (N, H, W)
        colors (np.ndarray | None, optional): NumPy array of mask colors to use with shape (N, 4) or (N, 3). If None,
            colors will be generated based on the `colormap`. Defaults to None.
        colormap (str, optional): Matplotlib colormap name for mask colors to use if `colors` is None.
            Defaults to "tab10".
        transparency (float, optional): Transparency level of mask overlay (0-1). Defaults to 0.5.

    Returns:
        dict[str, Image.Image]: A dictionary containing the original image with masks drawn and the image with all
            the regions apart from the masks (background) replaced with transparency.

    """
    full_image = image.copy().convert("RGBA")
    no_background = Image.new("RGBA", full_image.size, (0, 0, 0, 0))

    if masks is None:
        return image.copy(), no_background

    # Create a copy of the original image for drawing
    image_mode = image.mode
    full_image = full_image.convert("RGBA")

    if masks.ndim != 3 or masks.shape[1] != full_image.size[1] or masks.shape[2] != full_image.size[0]:
        msg = f"Expected 3D tensor of shape (N, H, W) for masks, got {masks.shape}"
        raise ValueError(msg)

    # Generate colors from colormap
    colors = _get_colormap(len(masks), colors, colormap)

    for i, mask in enumerate(masks):
        # Create RGBA image with mask as transparency (alpha channel)
        # Convert mask to PIL Image (assuming mask is 0-1 or 0-255)
        seg_arr = (((mask * 255) if mask.max() <= 1.0 else mask) * transparency).astype(np.uint8)

        # Create binary mask image
        seg_mask = Image.fromarray(seg_arr)
        overlay_color = Image.new("RGBA", full_image.size, colors[i])
        full_image = Image.composite(overlay_color, full_image, seg_mask)

        # Add region to no-background image
        no_background = Image.composite(image, no_background, Image.fromarray(seg_arr.astype(bool)))

    return {"image": full_image.convert(image_mode), "no_background_image": no_background}
