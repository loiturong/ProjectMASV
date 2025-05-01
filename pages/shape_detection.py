import streamlit as st
import cv2 as cv
import numpy as np

# SHAPE_DICT = {
#             "circle": [0.000624, 0.000626, "red"],
#             "square": [0.000648, 0.000664, "green"],
#             "triangle": [0.000729, 0.000747, "blue"]
#             }

def threshold_map(hu_feature: float, variant_factor: float = 0.15):
    low_lim = 1 - variant_factor
    high_lim = 1 + variant_factor
    if 0.000624 * low_lim < hu_feature < 0.000626 * high_lim:
        return "circle", (255, 0, 0)
    elif 0.000648 * low_lim < hu_feature < 0.000664 * high_lim:
        return "square", (0, 255, 0)
    elif 0.000729 * low_lim < hu_feature < 0.000747 * high_lim:
        return "triangle", (0, 0, 255)
    else:
        return "unknown", None

def thresholding_image(image: np.array):
    img = np.mean(image, axis=-1, keepdims=False).astype(np.uint8)
    img = np.where((img == 63), np.uint8(255), np.uint8(0)).astype(np.uint8)

    return cv.medianBlur(img, 7)

def compute_hu_feature(threshold_image: np.array):
    m = cv.moments(threshold_image)
    hu = cv.HuMoments(m)
    return np.float64(hu[0, 0])

def run_shape_classification(image: np.array):
    threshold_image = thresholding_image(image)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold_image)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        this_shape = np.where(labels == i, threshold_image, np.uint8(0)).astype(np.uint8)

        hu_feature = compute_hu_feature(this_shape)
        shape, color = threshold_map(hu_feature, variant_factor=0.005)
        # Draw the bounding box and centroid on the original image
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, shape, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

class MultiShapeDetection:
    """
    Shape Detection from Camera IFM
    """
    def __init__(self):
        self.title = ":rainbow[Shape Detection] :gray[Tool]"
        self.welcome_message = "Welcome to our Digital Processing Application! This tool provides capabilities of digital processing."


    def render(self):
        st.title(self.title, anchor=False)
        st.write(self.welcome_message)
        with st.expander("More Details About This Features"):
            st.write("""
                    This application offers state-of-the-art machine vision capabilities:
                    - Multiple shape detections in a single image
                    - Lightweight processing using numpy and OpenCV
                    """)
        from utils import handle_image_upload
        @handle_image_upload(header_text="Upload Image", button_text="Process Image")
        def _process_upload(image):
            # check if the image is np.array
            if not hasattr(image, "shape"):
                image = np.array(image)

            return run_shape_classification(image)

        result, img_in, img_out, img_status = _process_upload()

        if result is not None:
            img_out.image(result, caption="Processed Image", use_container_width=True)
            img_status.success(f"Processed successfully!")

if __name__ == "__main__":
    MultiShapeDetection().render()