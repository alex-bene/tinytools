"""Camera tools."""

import math


def focal_to_fov(image_size: tuple[int, int], focal_length: float, in_degrees: bool = True) -> dict[str, float]:
    """Calculate horizontal field of view from image width and focal length.

    Args:
        image_size (tuple[int, int]): Image size in pixels (width, height).
        focal_length (float): Focal length in pixels.
        in_degrees (bool, optional): Whether to return the field of view in degrees or radians.

    Returns:
        dict[str, float]: Dictionary with keys "horizontal" and "vertical"
            corresponding to horizontal and vertical FOV.

    """
    fov = {
        "horizontal": 2 * math.atan(image_size[0] / (2 * focal_length)),
        "vertical": 2 * math.atan(image_size[1] / (2 * focal_length)),
    }

    if not in_degrees:
        return fov

    return {key: math.degrees(value) for key, value in fov.items()}


def fov_to_focal(image_size: int, fov: float, is_degrees: bool = True) -> float:
    """Calculate focal length from image size and field of view.

    Args:
        image_size (int): Image size in pixels, either width or height.
        fov (float): Field of view, either horizontal or vertical. Should match the dimension of the image,
            i.e. horizontal for width and vertical for height.
        is_degrees (bool, optional): Whether the field of view is in degrees or radians.

    Returns:
        float: Focal length in pixels

    """
    if is_degrees:
        fov = math.radians(fov)

    return image_size / (2 * math.tan(fov / 2))
