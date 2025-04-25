import streamlit as st
import cv2 as cv
import numpy as np


def _load_chapter_3(func_list):
    function = func_list.selectbox("Select Function in Chapter 3:",
                        ["Image Negatives",
                         "Log Transformations",
                         "Power-Law (Gamma) Transformations",
                         "Contrast Stretching",
                         "Intensity-Level Slicing",
                         "Histogram Equalization"
                         ])
    return function

def _load_chapter_4(func_list):
    function = func_list.selectbox("Select Function in Chapter 4:",
                                   [
                                    ])
    return function

def _load_chapter_9(func_list):
    function = func_list.selectbox("Select Function in Chapter 9:",
                                   [
                                    ])
    return function

def process_chapter(chapter, func_list):
    match chapter:
        case "Chapter 3: Intensity Transformations and Spatial Filtering":
            return _load_chapter_3(func_list)
        case "Chapter 4: Filtering in the Frequency Domain":
            return _load_chapter_4(func_list)
        case "Chapter 9: Morphological Image Processing":
            return _load_chapter_9(func_list)
        case _:
            return func_list.failure("Invalid or not Implemented chapter selected.")


def parse_parameters(param_string):
    """Parse a string of parameters into a dictionary.
    Example: 'factor=1, max=12' -> {'factor': 1, 'max': 12}
    """
    params = {}
    if not param_string.strip():
        return params

    # Split by comma and process each key-value pair
    pairs = param_string.split(',')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Try to convert value to the appropriate type (int, float, or keep as string)
            try:
                # First try as int
                params[key] = int(value)
            except ValueError:
                try:
                    # Then try as float
                    params[key] = float(value)
                except ValueError:
                    # Keep as a string if not a number
                    params[key] = value

    return params

class DigitalProcessingPage:
    def __init__(self):
        self.title = ":rainbow[Digital Processing] :gray[Tool]"
        self.welcome_message = "Welcome to our Digital Processing Application! This tool provides capabilities of digital processing."
        self.model = None
        self.func = None

    def render(self):
        st.title(self.title, anchor=False)
        st.write(self.welcome_message)
        with st.expander("More Details About This Features"):
            st.write("""
            This application offers state-of-the-art machine vision capabilities:
            - Multiple processing functions
            - Lightweight processing using numpy and OpenCV
            """)
        chapter = st.selectbox("Select Chapter in Digital Image Processing book:",
                               ["Chapter 3: Intensity Transformations and Spatial Filtering",
                                "Chapter 4: Filtering in the Frequency Domain",
                                "Chapter 9: Morphological Image Processing"])
        # Placeholder for function selection
        func_list = st.empty()
        function = process_chapter(chapter, func_list)

        # Add parameter input field
        param_input = st.text_input("Function Parameters (e.g. 'factor=1, max=12', leave blank if none), lookup in the book:", "")

        # Parse parameters
        params = parse_parameters(param_input)

        status = st.empty()
        st.divider()
        tab1, tab2, tab3 = st.tabs(["Upload Image", "Camera", "Database Images"])
        with tab1:
            from utils import handle_image_upload
            @handle_image_upload(header_text="Upload Image", button_text="run Digital Processing")
            def _process_upload(image):
                # check if the image is np.array
                if not getattr(image, "shape"):
                    image = np.array(image)
                # convert to grayscale
                if len(image.shape) == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                return function(image, **params)
            _process_upload()

        with tab2:
            from utils import handle_camera_input
            @handle_camera_input(header_text="Record camera", fps=24)
            def _process_upload(image):
                # check if the image is np.array
                if not getattr(image, "shape"):
                    image = np.array(image)
                # convert to grayscale
                if len(image.shape) == 3:
                    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                return function(image, **params)
            _process_upload()

        with tab3:
            st.header("Not Implemented", anchor=False)

if __name__ == "__main__":
    DigitalProcessingPage().render()