# TinyTools

TinyTools is a small utility library designed to provide common functionality for use in various projects. It includes tools for handling archives, images, logging, and video processing.

## Features

- **Archive Extraction**: Safe extraction of ZIP and TAR archives with protection against path traversal attacks.
- **Image Processing**: Tools for converting NumPy arrays to PIL Images, creating image grids, and handling batches of images.
- **Logging**: Rich logging configuration with colored output and custom formatting.
- **Video Processing**: Utilities for loading videos as image lists or NumPy arrays, saving video files from image lists, and retrieving video metadata.

## Installation

```bash
pip install git+https://github.com/alex-bene/tinytools.git
# or
uv add git+https://github.com/alex-bene/tinytools.git
```

## Usage Examples

### Archive Extraction

```python
import zipfile
from tinytools.archives import safe_zip_extract_all

with zipfile.ZipFile("example.zip", "r") as zip_ref:
    safe_zip_extract_all(zip_ref, "./extracted_files")
```

### Image Processing

```python
import numpy as np
from PIL import Image
from tinytools.image import img_from_array, imgs_from_array_batch

# Convert a NumPy array to PIL Image
img = img_from_array(np.random.rand(100, 100, 3))

# Convert a batch of images to PIL Images
batch = np.random.rand(5, 100, 100, 3)
imgs = imgs_from_array_batch(batch)

# Create an image grid
grid = image_grid(imgs, rows=2, cols=3)
```

### Logging

```python
from tinytools.logger import get_logger

logger = get_logger("my_app")
logger.info("This is an info message")
```

### Video Processing

```python
from tinytools.video import load_video, save_video

# Load video as list of PIL images
frames = load_video("input.mp4")

# Save list of PIL images as video
save_video(frames, "output.mp4", fps=30.0)
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
