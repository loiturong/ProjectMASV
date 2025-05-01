import numpy as np
import cv2 as cv

__all__ = [
    "connected_component",
    "remove_small_rice"
]

def connected_component(image: np.array, thresh = 200, **kwargs):
    if kwargs.get("thresh"):
        thresh = kwargs["thresh"]
    temp = cv.threshold(image, thresh, 255, cv.THRESH_BINARY)[1]
    image = cv.medianBlur(temp, 7)
    n, label = cv.connectedComponents(image, None)

    a = np.zeros((n), np.int32)
    M, N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            if r > 0:
                a[r] = a[r] + 1

    temp = "there are %d connected component" % (n-1)
    cv.putText(image, temp, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    for r in range(1, n):
        s = "%3d %5d" % (r, int(a[r]))
        cv.putText(image, s, (10, 20 + 20 * (r+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    return image

def remove_small_rice(image, **kwargs):
    w = cv.getStructuringElement(cv.MORPH_ELLIPSE, (81, 81))
    temp = cv.morphologyEx(image, cv.MORPH_TOPHAT, w)

    thresh = kwargs.get("thresh", 100)
    temp = cv.threshold(temp, thresh, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    n, label = cv.connectedComponents(temp, None)

    a = np.zeros((n), np.int32)
    M, N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            if r > 0:
                a[r] = a[r] + 1
    max_val = np.max(a)
    image = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = label[x, y]
            if r > 0 and a[r] > 0.7 * max_val:
                image[x, y] = 255

    return image

if __name__ == "__main__":
    pass