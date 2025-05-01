import numpy as np
import cv2 as cv

__all__ = [
    "get_image_spectrum",
    "get_image_phase_angle",
    "remove_moire_simple",
    "remove_moire",
    "remove_interference"
]

def fourier_transform(image: np.ndarray, **kwargs):
    height, width = image.shape
    fourier_height = cv.getOptimalDFTSize(height)
    fourier_width = cv.getOptimalDFTSize(width)
    preprocessed_image = np.zeros((fourier_height, fourier_width), np.float32)
    preprocessed_image[:height, :width] = 1.0 * image / 255

    for x in range(0, height):
        for y in range(0, width):
            if (x + y) % 2 == 1:
                preprocessed_image[x, y] = -preprocessed_image[x, y]

    return cv.dft(preprocessed_image, flags=cv.DFT_COMPLEX_OUTPUT)

def get_image_spectrum(image: np.ndarray, **kwargs):
    fourier = fourier_transform(image)

    spectrum = np.sqrt(fourier[:, :, 0] ** 2 + fourier[:, :, 1] ** 2)
    spectrum = np.clip(spectrum, 0, 255).astype(np.uint8)
    return spectrum

def get_image_phase_angle(image: np.ndarray, **kwargs):
    fourier = fourier_transform(image)

    preprocessed_image = np.arctan2(fourier[:, :, 1], fourier[:, :, 0])
    phase_angle_degrees = np.degrees(preprocessed_image).astype(np.uint8)

    return phase_angle_degrees

def remove_moire_simple(image: np.ndarray, **kwargs):
    height, width = image.shape
    dft_height = cv.getOptimalDFTSize(height)
    dft_width = cv.getOptimalDFTSize(width)
    notch_filter = create_notch_filter(dft_height, dft_width)
    return frequency_filtering(image, notch_filter)

def remove_moire(image: np.ndarray, **kwargs):
    height, width = image.shape
    dft_height = cv.getOptimalDFTSize(height)
    dft_width = cv.getOptimalDFTSize(width)
    butterworth_filter = create_butterworth_notch_reject_filter(dft_height, dft_width)
    return frequency_filtering(image, butterworth_filter)

def remove_interference(image: np.ndarray, **kwargs):
    height, width = image.shape
    dft_height = cv.getOptimalDFTSize(height)
    dft_width = cv.getOptimalDFTSize(width)
    vertical_filter = create_vertical_notch_rejec_filter(dft_height, dft_width)
    return frequency_filtering(image, vertical_filter)

def create_notch_filter(dft_height, dft_width):
    notch_filter = np.ones((dft_height, dft_width, 2), np.float32)
    notch_filter[:, :, 1] = 0.0

    u1, v1 = 45, 59
    u2, v2 = 86, 59
    u3, v3 = 39, 119
    u4, v4 = 83, 119
    D0 = 10
    for u in range(0, dft_height):
        for v in range(0, dft_width):
            # u1, v1
            d_uv = np.sqrt((u - u1) ** 2 + (v - v1) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # u2, v2
            d_uv = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # u3, v3
            d_uv = np.sqrt((u - u3) ** 2 + (v - v3) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # u4, v4
            d_uv = np.sqrt((u - u4) ** 2 + (v - v4) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0

            # Đối xứng của u1, v1
            d_uv = np.sqrt((u - (dft_height - u1)) ** 2 + (v - (dft_width - v1)) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # Đối xứng của u2, v2
            d_uv = np.sqrt((u - (dft_height - u2)) ** 2 + (v - (dft_width - v2)) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # Đối xứng của u3, v3
            d_uv = np.sqrt((u - (dft_height - u3)) ** 2 + (v - (dft_width - v3)) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
            # Đối xứng của u4, v4
            d_uv = np.sqrt((u - (dft_height - u4)) ** 2 + (v - (dft_width - v4)) ** 2)
            if d_uv <= D0:
                notch_filter[u, v, 0] = 0.0
    return notch_filter

def create_butterworth_notch_reject_filter(P, Q):
    H = np.ones((P, Q, 2), np.float32)
    H[:, :, 1] = 0.0

    u1, v1 = 45, 59
    u2, v2 = 86, 59
    u3, v3 = 39, 119
    u4, v4 = 83, 119
    D0 = 10
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            r = 1.0
            # u1, v1
            Duv = np.sqrt((u - u1) ** 2 + (v - v1) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # u2, v2
            Duv = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # u3, v3
            Duv = np.sqrt((u - u3) ** 2 + (v - v3) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # u4, v4
            Duv = np.sqrt((u - u4) ** 2 + (v - v4) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # Đối xứng của u1, v1
            Duv = np.sqrt((u - (P - u1)) ** 2 + (v - (Q - v1)) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # Đối xứng của u2, v2
            Duv = np.sqrt((u - (P - u2)) ** 2 + (v - (Q - v2)) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # Đối xứng của u3, v3
            Duv = np.sqrt((u - (P - u3)) ** 2 + (v - (Q - v3)) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            # Đối xứng của u4, v4
            Duv = np.sqrt((u - (P - u4)) ** 2 + (v - (Q - v4)) ** 2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r = r * 1.0 / (1.0 + np.power(D0 / Duv, n))
                else:
                    r = 0.0

            H[u, v, 0] = r
    return H

def create_vertical_notch_rejec_filter(P, Q):
    H = np.ones((P, Q, 2), np.float32)
    H[:, :, 1] = 0.0
    D0 = 7
    D1 = 10
    for u in range(0, P):
        for v in range(0, Q):
            if not u in range(Q // 2 - D1, Q // 2 + D1):
                D = v - Q // 2
                if abs(D) <= D0:
                    H[u, v, 0] = 0
    return H

def frequency_filtering(image, freq_filter):
    height, width = image.shape

    P = cv.getOptimalDFTSize(height)
    Q = cv.getOptimalDFTSize(width)
    fp = np.zeros((P, Q), np.float32)
    fp[:height, :width] = 1.0 * image

    for x in range(0, height):
        for y in range(0, width):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    F = cv.dft(fp, flags=cv.DFT_COMPLEX_OUTPUT)
    G = cv.mulSpectrums(F, freq_filter, flags=cv.DFT_ROWS)
    g = cv.idft(G, flags=cv.DFT_SCALE)

    gR = g[:height, :width, 0]
    for x in range(0, height):
        for y in range(0, width):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]

    return np.clip(gR, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    pass