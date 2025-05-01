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
           "local_histogram",
           "histogram_statistic",
           "sharpening",
           "gradient"
           ]

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

    l = kwargs["L"] if "L" in kwargs else 255

    constant = (l - 1.0) / np.log(1.0 * l)
    image_np = constant * np.log(1.0 + image_np)

    # convert back to 8-bit integer to show in scale 0-255
    return np.array(image_np, dtype=np.uint8)

def gamma_transform(image: np.ndarray, **kwargs) -> np.ndarray:
    """Power-law transformations or Gamma Transformations"""
    # Ensure it's a NumPy array
    image_np = np.asarray(image)

    # check if image is a gray scale image
    assert (image_np.ndim == 2), "Must be an Gray Scale Image."

    # Get params for Transformation function
    gamma = float(kwargs["gamma"]) if "gamma" in kwargs else 2.5
    l = kwargs["L"] if "L" in kwargs else 255

    # compute constant value with respect to gamma value (and L)
    constant = np.power(l - 1.0 , 1.0 - gamma)

    image_np = constant * np.power(image_np, gamma)

    return np.array(image_np, dtype=np.uint8)

def contrast_stretching(image: np.ndarray, **kwargs) -> np.ndarray:
    r_min, r_max = np.min(image), np.max(image)
    new_weight = (255 - 0) / (r_max - r_min)

    return np.array(new_weight * (image - r_min), dtype=np.uint8)

def intensity_thresholding(image: np.ndarray, **kwargs) -> np.ndarray:
    # Get params for Transformation function
    threshold = int(kwargs["threshold"]) if "threshold" in kwargs else np.mean(image)

    return np.array(np.where(image > threshold, 255, 0), dtype=np.uint8)

def intensity_slicing(image: np.ndarray, **kwargs) -> np.ndarray:
    # Get params for Transformation function
    mode = kwargs["mode"] if "mode" in kwargs else "linear"
    a = int(kwargs["A"]) if "A" in kwargs else np.mean(image) - 25
    b = int(kwargs["B"]) if "B" in kwargs else np.mean(image) + 25
    c = int(kwargs["C"]) if "C" in kwargs else np.max(image)

    if mode == "linear":
        return np.array(np.where((a < image) & (image < b), c, image), dtype=np.uint8)
    elif mode == "highlights":
        return np.array(np.where((a < image) & (image < b), c, 0), dtype=np.uint8)

def bit_plane_shift(image: np.ndarray, **kwargs):
    # Get params for Transformation function
    bits = int(kwargs["bits"]) if "bits" in kwargs else 8   # default plane is the MSB plane

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
    cum_sum = np.histogram(image, bins=256, range=(0, 255))[0]
    cum_sum = np.cumsum(cum_sum, dtype=np.float32) / np.multiply(image.shape[0], image.shape[1])
    cum_sum = (np.round(cum_sum * 255)).astype(np.uint8)

    return cum_sum[image]   # using an 2D array as index for 1D array is FANCY

def local_histogram(image: np.ndarray, **kwargs):
    height, width = image.shape
    buffer_out = np.zeros((height, width), np.uint8)

    filter_size = int(kwargs["filter_size"]) if "filter_size" in kwargs else 3

    for i in range(0, height, filter_size):
        for j in range(0, width, filter_size):
            end_i = i + filter_size if i + filter_size < height else height - 1
            end_j = j + filter_size if j + filter_size < width else width - 1
            this_block = image[i:end_i, j:end_j]
            # print(f"\rProcessing: ({i}, {j}) to ({end_j}, {end_j})", end="")
            buffer_out[i:end_i, j:end_j] = histogram_equalization(this_block)

    return buffer_out

def histogram_statistic(image: np.array, **kwargs):
    domain_range = np.linspace(0, 255, 256)

    def compute_mean(x: np.array):
        glob_hist = np.histogram(x, bins=256, range=(0,255))[0] / (x.shape[0] * x.shape[1])
        return np.round(np.sum(domain_range * glob_hist)).astype(np.uint8)

    def compute_std(x: np.array, mean: int = None):
        glob_hist = np.histogram(x, bins=256, range=(0,255))[0] / (x.shape[0] * x.shape[1])
        return np.round(np.sqrt(np.sum(np.power(domain_range - mean, 2) * glob_hist)))

    image_glob_mean = compute_mean(image)
    image_glob_std = compute_std(image, image_glob_mean)
    # get params, default according to book
    filter_size = int(kwargs["filter_size"]) if "filter_size" in kwargs else 3
    k0 = int(kwargs["k0"]) if "ko" in kwargs else 0
    k1 = int(kwargs["k1"]) if "k1" in kwargs else 0.25
    k2 = int(kwargs["k2"]) if "k2" in kwargs else 0
    k3 = int(kwargs["k3"]) if "k3" in kwargs else 0.1
    constant = np.uint8(kwargs["constant"]) if "constant" in kwargs else 22.8
    height, width = image.shape
    buffer_out = np.copy(image).astype(np.uint8)
    for i in range(0, height, filter_size):
        for j in range(0, width, filter_size):
            end_i = i + filter_size if i + filter_size < height else height - 1
            end_j = j + filter_size if j + filter_size < width else width - 1
            this_block = image[i:end_i, j:end_j]

            local_mean = compute_mean(this_block)
            local_std = compute_std(this_block, local_mean)

            # print(f"\rProcessing: ({i}, {j}) to ({end_j}, {end_j})", end="")
            if (k0 * image_glob_mean <= local_mean) and (local_mean <= k1 * image_glob_mean):
                if (k2 * image_glob_std <= local_std) and (local_std <= k3 * image_glob_std):
                    buffer_out[i:end_i, j:end_j] = np.multiply(this_block, constant)

    return buffer_out

def sharpening(image: np.ndarray, **kwargs):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    laplacian_filtered = cv.filter2D(image, cv.CV_32FC1, kernel)
    buffer_out = image - laplacian_filtered
    buffer_out = np.clip(buffer_out, 0, 255).astype(np.uint8)
    return buffer_out


def gradient(image: np.ndarray, **kwargs):
    kernel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gx = cv.filter2D(image, cv.CV_32FC1, kernel_x)
    gy = cv.filter2D(image, cv.CV_32FC1, kernel_y)

    image = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

if __name__ == "__main__":
    pass