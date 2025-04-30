import numpy as np
import cv2 as cv

__all__ = ["negative",
           "log_transform",
           "gamma_transform",
           "contrast_stretching",
           "intensity_thresholding",
           "intensity_slicing",
           "bit_plane_shift",
           "histogram_visualize",
           "histogram_equalization",
           "local_histogram"
           ]

from networkx.classes import neighbors


def negative(image, **kwargs):
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    image_np = 255 - image_np

    return image_np

def log_transform(image: np.ndarray, **kwargs) -> np.ndarray:
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    if "L" in kwargs:
        L = kwargs["constant_factor"]
    else:
        L: int = 255

    constant = (L - 1.0) / np.log(1.0 * L)
    image_np = constant * np.log(1.0 + image_np)

    # convert back to 8-bit integer to show in scale 0-255
    return np.array(image_np, dtype=np.uint8)

def gamma_transform(image: np.ndarray, **kwargs) -> np.ndarray:
    """Power-law transformations or Gamma Transformations"""
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    if "gamma" in kwargs:
        gamma = float(kwargs["gamma"])
    else:
        gamma = 2.5

    if "L" in kwargs:
        L = kwargs["L"]
    else:
        L = 255

    # compute constant value with respect to gamma value (and L)
    constant = np.power(L - 1.0 , 1.0 - gamma)

    image_np = constant * np.power(image_np, gamma)

    return np.array(image_np, dtype=np.uint8)

def contrast_stretching(image: np.ndarray, **kwargs) -> np.ndarray:
    r_min, r_max = np.min(image), np.max(image)
    new_weight = (255 - 0) / (r_max - r_min)

    return np.array(new_weight * (image - r_min), dtype=np.uint8)

def intensity_thresholding(image: np.ndarray, **kwargs) -> np.ndarray:
    if "threshold" in kwargs:
        threshold = kwargs["threshold"]
    else:
        threshold = np.mean(image)
    return np.array(np.where(image > threshold, 255, 0), dtype=np.uint8)

def intensity_slicing(image: np.ndarray, **kwargs) -> np.ndarray:
    if "mode" in kwargs:
        mode = kwargs["mode"]
    else:
        mode = "linear"
    if "A" in kwargs:
        a = int(kwargs["A"])
    else:
        a = np.mean(image) - 25
    if "B" in kwargs:
        b = int(kwargs["B"])
    else:
        b = np.mean(image) + 25
    if "C" in kwargs:
        c = int(kwargs["C"])
    else:
        c = np.max(image)
    if mode == "linear":
        return np.array(np.where((a < image) & (image < b), c, image), dtype=np.uint8)
    elif mode == "highlights":
        return np.array(np.where((a < image) & (image < b), c, 0), dtype=np.uint8)

def bit_plane_shift(image: np.ndarray, **kwargs):
    if "bits" in kwargs:
        bits = int(kwargs["bits"])
    else:
        bits = 87654321    # MSB
    new_image = np.zeros(shape=image.shape, dtype=np.uint8)
    while bits > 0:
        new_image |= np.uint8((1 << (bits % 10 - 1)) & image)
        bits //= 10
    return new_image

def histogram_visualize(image: np.ndarray, **kwargs):
    height, width = image.shape
    buffer_out = np.zeros(shape=(256, 256), dtype=np.uint8)
    for i in range(256):
        pixel_count = np.count_nonzero(image == i) / (height * width)
        pixel_count *= 255
        cv.line(buffer_out, (i, int(255 - pixel_count)), (i, 255), [255], 1)

    return buffer_out

def histogram_equalization(image: np.ndarray, **kwargs):
    cum_sum = [0 for _ in range(256)]
    height, width = image.shape
    cum_sum[0] = np.sum(image == 0) / (height * width)
    buffer_out = np.array(np.where(image == 0, np.int8(round(255 * cum_sum[0])), np.array(0)), dtype=np.uint8)
    for i in range(1, 256):
        cum_sum[i] = cum_sum[i - 1] + np.sum(image == i) / (height * width)
        buffer_out = np.array(np.where(image == i, np.int8(round(255 * cum_sum[i])), buffer_out), dtype=np.uint8)

    return buffer_out

def local_histogram(image: np.ndarray, **kwargs):
    height, width = image.shape
    buffer_out = np.zeros((height, width), np.uint8)
    if "filter_size" in kwargs:
        filter_size = int(kwargs["filter_size"])
    else:
        filter_size = 3

    for i in range(height)[::filter_size]:
        for j in range(width)[::filter_size]:
            pointer_i = i + filter_size if i + filter_size < height else height - 1
            pointer_j = j + filter_size if j + filter_size < width else width - 1
            buffer_out[i:pointer_i, j:pointer_j] = histogram_equalization(image[i:pointer_i, j:pointer_j])

    return buffer_out

if __name__ == "__main__":
    pass