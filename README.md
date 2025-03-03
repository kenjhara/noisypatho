# noisypatho: Create Noisy Annotations for Pathological Semantic Segmentation Data
![](https://github.com/kenjhara/noisypatho/blob/main/noisypatho_logo.png)

## Overview
**noisypatho** is a tool for generating artificially noisy annotations for pathological binarized segmentation tasks.  
It can be used with any type of image (e.g., H&E-stained) and their corresponding segmentation masks.  
In this repository, we demonstrate its functionality using [Breast Cancer Semantic Segmentation (BCSS) data](https://github.com/PathologyDataScience/BCSS) \[1\].

## Requirements
- **Python**: 3.10 or newer
- **Dependencies** (examples below; please confirm versions in your environment):
  - matplotlib
  - numpy
  - pandas
  - Pillow
  - scipy
  - opencv-python

## Installation
Install the library using the following command:
```bash
pip install git+https://github.com/kenjhara/noisypatho.git
```

To see a demonstration and verify a successful installation, open and run `usage.ipynb` in this repository.

## Usage

Below is an example of how to generate noisy labels using **noisypatho**.  
We assume you have at least one H&E-stained image and a corresponding segmentation mask.
To see a demonstration, open and run `usage.ipynb` in this repository.

### Preparation
```python
from PIL import Image
import os

from noisypatho import BCSS
from noisypatho import utils
from noisypatho import noise

# Paths
# If you have multiple images to process, you can use glob or a CSV file to manage all paths.
path_he_list = ["./example/he/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"]
path_mask_list = ["./example/mask/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"]
```

### Noise setting
```python
# noise_level can be 1, 2, or 3 — choose one
noise_level = 3

# noise_types can be ["dilation", "shrink", "additive", "omission"]
# You can select one or combine multiple
noise_types = ["dilation"]

num_img = len(path_he_list)
seed = 3

# Create parameters for noise
params = noise.noise_parameters(num_img, noise_level, noise_types, seed)[0]

```

### Create Noise

```python
for path_he, path_mask in zip(path_he_list, path_mask_list):
    # Open images
    img_he = Image.open(path_he)
    img_mask = Image.open(path_mask)

    # Apply noise to the segmentation mask
    result = img_mask
    if "omission" in noise_types:
        result = noise.MAKE_OMISSION_NOISE(result, params)
    if "dilation" in noise_types:
        result = noise.MAKE_DILATION_NOISE(result, params)
    if "shrink" in noise_types:
        result = noise.MAKE_SHRINK_NOISE(result, params)
    if "additive" in noise_types:
        result = noise.MAKE_ADDITIVE_NOISE(result, params)

    # Save the noisy mask
    os.makedirs("./example/noisylabel/", exist_ok=True)
    noise_types_name = "-".join(noise_types)
    base_name = os.path.basename(path_he)
    result.save(f"./example/noisylabel/Level{noise_level}_{noise_types_name}_{base_name}.jpg")
```

### Additional Information
- You can batch-process multiple images by preparing lists of image paths (using `glob` or a CSV file for path management).
- Different noise levels (1, 2, 3) and combinations of noise types (`["dilation", "shrink", "additive", "omission"]`) can be used to simulate various annotation errors.

## License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- For non-commercial use, please use the code under **CC-BY-NC-SA**.  
- For commercial use, please contact us at [kenjhara@east.ncc.go.jp](mailto:kenjhara@east.ncc.go.jp).

## Citation
If you use **noisypatho** in your research, please cite our paper, and/or provide a link to this GitHub repository.
1. In preparation

## Reference
1. Amgad M, Elfandy H, Hussein H, et al. Structured crowdsourcing enables convolutional segmentation of histology images. *Bioinformatics*. 2019;35(18):3461-3467. doi:[10.1093/bioinformatics/btz083](https://doi.org/10.1093/bioinformatics/btz083)
