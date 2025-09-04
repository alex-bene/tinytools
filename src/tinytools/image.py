"""Image tools."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def img_from_array(img_array: np.ndarray, is_bgr: bool = False) -> Image.Image:
    """Convert a NumPy array representing image(s) to a list of PIL Image objects.

    Args:
        img_array (np.ndarray): A NumPy array representing the image(s). The array can have the following shapes:
            - (H, W, C): A color image, where H is height, W is width, and C is the number of channels.
            - (H, W): A grayscale image, where H is height and W is width.
        is_bgr (bool): A boolean indicating whether the input images are in BGR format.
            If True, the function will convert them to RGB. Defaults to False (assumes RGB or grayscale).

    Returns:
        Image.Image: A PIL Image.

    Raises:
        ValueError: If the number of dimensions in the input array is not 2D or 3D.

    """
    if is_bgr:
        img_array = np.flip(img_array, axis=-1)  # Convert BGR to RGB

    img_array = img_array.squeeze()  # Remove singleton dimensions

    if len(img_array.shape) < 2 or len(img_array.shape) > 3:
        msg = f"Invalid number of dimensions {len(img_array.shape)} for image array."
        raise ValueError(msg)

    # ensure correct type for all cases before doing anything
    img_array = img_array.astype(np.uint8)

    return Image.fromarray(img_array)


def imgs_from_array_batch(img_array_batch: np.ndarray, is_bgr: bool = False) -> list[Image.Image]:
    """Convert a NumPy array representing a batch of image(s) to a list of PIL Image objects.

    Args:
        img_array_batch (np.ndarray): An array representing the batch of images with one of the following shapes:
            - (N, H, W, C): A batch of color images, where N is the batch size, H is height, W is width, and C is the
                number of channels (e.g., 3 for RGB).
            - (N, H, W): A batch of grayscale images, where N is the batch size, H is height, and W is width.
        is_bgr (bool): A boolean indicating whether the input images are in BGR format.
            If True, the function will convert them to RGB. Defaults to False (assumes RGB or grayscale).

    Returns:
        list[Image.Image]: A list of PIL Image objects.

    Raises:
        ValueError: If the number of dimensions in the input array is not 3D or 4D.

    """
    if len(img_array_batch.shape) < 3 or len(img_array_batch.shape) > 4:
        msg = (
            f"Invalid number of dimensions {len(img_array_batch.shape)} for batch image array. "
            "Must be at least 3D (N, H, W) or 4D (N, H, W, C)."
        )
        raise ValueError(msg)

    return [img_from_array(img_array, is_bgr=is_bgr) for img_array in img_array_batch]


def image_grid(
    image_list: list[Image.Image | None],
    max_columns: int = 3,
    padding: int = 0,
    bg_color: tuple = (255, 255, 255),
    resize_to_fit: bool = True,
) -> Image.Image:
    """Create a grid of images from a list of image objects.

    Args:
        image_list (list[Image.Image | None]): A list of PIL Image objects or None. None values will be drawn as empty.
        max_columns (int, optional): The maximum number of columns in the grid. Defaults to 3.
        padding (int, optional): The number of pixels of padding between images. Defaults to 0.
        bg_color (tuple, optional): The background color for the grid and padding. Defaults to white (255, 255, 255).
        resize_to_fit (bool, optional): If True, resizes images to fit the grid cell dimensions without stretching.
            Defaults to True.

    Returns:
        PIL.Image.Image: A new Image object containing the grid of images.

    Raises:
        ValueError: If the image_list is empty.

    """
    if not image_list:
        msg = "image_list is empty."
        raise ValueError(msg)

    # --- Determine the size of each grid cell ---
    # Find the maximum width and height among all images. This will be the cell size.
    max_width = 0
    max_height = 0
    for img in image_list:
        if img is None:
            continue
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

    # --- Determine the number of columns ---
    num_images = len(image_list)
    # Use provided max columns, but ensure it doesn't exceed the number of available images
    cols = min(max_columns, num_images)

    # --- Calculate grid dimensions ---
    # Calculate the number of rows needed
    rows = math.ceil(num_images / cols)

    # Calculate the total width and height of the grid image, using cell dimensions and padding
    grid_width = (cols * max_width) + ((cols - 1) * padding)
    grid_height = (rows * max_height) + ((rows - 1) * padding)

    # Create a new blank image (the canvas for the grid)
    grid = Image.new("RGB", size=(grid_width, grid_height), color=bg_color)

    # --- Paste the images onto the grid ---
    for i, img in enumerate(image_list):
        if img is None:
            continue

        # Determine the row and column for the current image
        row_idx = i // cols
        col_idx = i % cols

        # Calculate the top-left coordinate of the cell where the image will be pasted
        cell_x = col_idx * (max_width + padding)
        cell_y = row_idx * (max_height + padding)

        # Handle resizing if requested
        paste_img = img
        if resize_to_fit and (paste_img.width != max_width or paste_img.height != max_height):
            # Create a thumbnail that fits within the cell dimensions while preserving aspect ratio
            # Calculate scaling factors to fit within cell dimensions while preserving aspect ratio
            scale_x = max_width / img.width
            scale_y = max_height / img.height
            # Use the smaller scale to ensure the image fits entirely within the cell
            scale = min(scale_x, scale_y)
            # Calculate new dimensions
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            # Resize image while preserving aspect ratio
            paste_img = paste_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate the position to center the image
        paste_x = cell_x + (max_width - paste_img.width) // 2
        paste_y = cell_y + (max_height - paste_img.height) // 2

        # Paste the resized image onto the grid canvas
        grid.paste(paste_img, (paste_x, paste_y))

    return grid
