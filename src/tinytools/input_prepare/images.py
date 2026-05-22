"""Image tensor conversion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    get_image_type,
    infer_channel_dimension_format,
    make_list_of_images,
)

from tinytools.imports import optional_module

if TYPE_CHECKING:
    import torch  # pyright: ignore[reportMissingImports]
    import torchvision  # pyright: ignore[reportMissingImports]
else:
    torch = optional_module("torch", extra="torch")
    torchvision = optional_module("torchvision", extra="torch")


def prepare_images(
    images: ImageInput,
    expected_ndims: int = 3,
    do_convert_rgb: bool | None = None,
    input_data_format: str | ChannelDimension | None = None,
    device: torch.device | None = None,
    allow_none: bool = False,
) -> list[torch.Tensor | None]:
    """Convert image inputs to channel-first tensors.

    Args:
        images (ImageInput): Input image or batch of images. Supported layouts include
            grayscale `(H, W)`, channel-first `(C, H, W)`, and channel-last `(H, W, C)`.
        expected_ndims (int, optional): Expected per-image rank used by Hugging Face
            batching utilities. Default: 3.
        do_convert_rgb (bool | None, optional): Whether to convert each image to RGB
            before tensor conversion. If `None` or `False`, no explicit RGB conversion
            is applied. Default: None.
        input_data_format (str | ChannelDimension | None, optional): Channel-dimension
            format of each input image. If `None`, the format is inferred per image.
            Default: None.
        device (torch.device | None, optional): Target device for returned tensors.
            If `None`, tensors remain on their current/default device. Default: None.
        allow_none (bool, optional): Whether to preserve `None` entries in top-level
            Python sequence inputs. When `True`, `None` entries are returned unchanged
            in their original positions. Default: False.

    Returns:
        list[torch.Tensor | None]: Prepared image tensors in channels-first layout, with
            optional `None` placeholders preserved. Shape: [(C, H, W) | None, ...]

    """
    none_locations: list[int] = []
    if allow_none:
        if images is None:
            return [None]
        if isinstance(images, (list, tuple)):
            image_sequence = list(images)
            none_locations = [idx for idx, image in enumerate(image_sequence) if image is None]
            images = [image for image in image_sequence if image is not None]

    images_list = make_list_of_images(images, expected_ndims=expected_ndims)
    prepared_images = [prepare_single_image(image, do_convert_rgb, input_data_format, device) for image in images_list]
    for idx in none_locations:
        prepared_images.insert(idx, None)
    return prepared_images


def prepare_single_image(
    image: ImageInput,
    do_convert_rgb: bool | None = None,
    input_data_format: str | ChannelDimension | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a Hugging Face `ImageInput` object to a torch tensor in CHW layout.

    The conversion mirrors the Transformers fast image processor behavior:
    PIL images are converted with `torchvision` tensor conversion, NumPy inputs
    are wrapped with `torch.from_numpy`, and torch inputs are passed through.
    If the input is channel-last, it is permuted to channel-first.

    Args:
        image (ImageInput): Input image as PIL image, NumPy array, or torch tensor.
            Supported layouts include grayscale `(H, W)`, channel-first `(C, H, W)`,
            and channel-last `(H, W, C)`.
        do_convert_rgb (bool | None, optional): Whether to convert the input to
            RGB before tensor conversion. If `None` or `False`, no explicit RGB
            conversion is applied. Default: None.
        input_data_format (str | ChannelDimension | None, optional): Channel-dimension
            format of the input. If `None`, the format is inferred from `image`.
            Default: None.
        device (torch.device | None): Target device for the returned tensor.
            If `None`, the tensor remains on its current/default device.

    Returns:
        torch.Tensor: Image tensor with channels-first layout. Shape: (C, H, W)

    """
    # from transformers
    image_type = get_image_type(image)
    if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
        msg = f"Unsupported input image type {image_type}"
        raise ValueError(msg)

    if do_convert_rgb:
        image = convert_to_rgb(image)

    if image_type == ImageType.PIL:
        image = torchvision.transforms.v2.functional.pil_to_tensor(image)
    elif image_type == ImageType.NUMPY:
        # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
        image = torch.from_numpy(image).contiguous()

    # If the image is 2D, we need to unsqueeze it to add a channel dimension for processing
    if image.ndim == 2:
        image = image.unsqueeze(0)

    # Infer the channel dimension format if not provided
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    if input_data_format == ChannelDimension.LAST:
        # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
        image = image.permute(2, 0, 1).contiguous()

    # Now that we have torch tensors, we can move them to the right device
    if device is not None:
        image = image.to(device)

    return image
