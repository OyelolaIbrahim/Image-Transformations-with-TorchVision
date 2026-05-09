# Image Transformations with TorchVision

## Overview

This project demonstrates how to perform common image preprocessing and augmentation techniques using the `torchvision.transforms` module in Python. The project uses the Python Imaging Library (PIL) together with TorchVision to manipulate and transform images for computer vision and deep learning tasks.

The notebook covers:

* Loading and displaying images
* Center cropping
* Horizontal flipping
* Color jittering
* Five-crop augmentation
* Combining transformations using `Compose`
* Functional image transformations

These transformations are widely used in machine learning workflows, especially in image classification, object detection, and data augmentation pipelines.

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* PIL (Python Imaging Library)

---

## Installation

Install the required library before running the notebook:

```python
%pip install torchvision
```



---

## Notebook Workflow

### 1. Import Required Libraries

The notebook starts by importing:

```python
from PIL import Image
import torchvision.transforms as transforms
```

---

### 2. Load an Image

An image is loaded using PIL:

```python
image = Image.open("galina-nelyubova-3ARFkqm3cf0-unsplash.jpg")
```

---

### 3. Center Crop Transformation

The image is cropped from the center using:

```python
centre_crop = transforms.CenterCrop(1000)
centre_crop(image)
```

This transformation extracts the central portion of the image.

---

### 4. Random Horizontal Flip

The notebook demonstrates horizontal flipping:

```python
h_flip = transforms.RandomHorizontalFlip(p=1)
h_flip(image)
```

* `p=1` ensures the image is always flipped.
* Useful for data augmentation in deep learning.

---

### 5. Color Jitter Transformation

Color properties of the image are modified using:

```python
c_jitter = transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.8,
    hue=0.2
)

c_jitter(image)
```

This transformation helps improve model robustness by varying image appearance.

---

### 6. Five Crop Transformation

The notebook generates five crops from the image:

```python
five_crop = transforms.FiveCrop(200)
five_crop(image)
```

This produces:

* Top-left crop
* Top-right crop
* Bottom-left crop
* Bottom-right crop
* Center crop

---

### 7. Compose Multiple Transformations

Multiple transformations are combined using `Compose`:

```python
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.CenterCrop(200),
    transforms.ToTensor()
])

trans(image)
```

This creates a reusable preprocessing pipeline.

---

### 8. Functional Transformations

The notebook also uses TorchVision functional APIs:

```python
import torchvision.transforms.functional as tf

tf.center_crop(image, 1000)
```

Functional transforms provide lower-level control over image operations.

---

## Learning Outcomes

By working through the notebook, you will learn:

* How to preprocess images using TorchVision
* How to apply data augmentation techniques
* How to combine multiple transformations efficiently
* The difference between transform classes and functional APIs

---

## Applications

These techniques are commonly used in:

* Image Classification
* Object Detection
* Computer Vision Projects
* Deep Learning Pipelines
* CNN Model Training

---



