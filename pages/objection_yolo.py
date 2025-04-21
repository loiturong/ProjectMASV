import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

def _process_image(image):
    """
    Process the image with face recognition.

    Args:
        image: Image array in OpenCV format

    Returns:
        Processed image with recognition results
    """

    # Create a copy to avoid modifying the original
    img_copy = image.copy()

    return img_copy

class ObjectDetectionYoloPage:
    def __init__(self):
        self.title = ":rainbow[Object Detection] :gray[Tool]"
        self.welcome_message = "Welcome to our Object Detection Application! This tool provides capabilities of detect object and fruit (i have no idea) in real-time."
        self.model = None

    def render(self):
        st.title(self.title, anchor=False)
        st.write(self.welcome_message)
        with st.expander("More Details About This Features"):
            st.write("""
            This application offers state-of-the-art machine vision capabilities:
            - Real-time object detection
            - Multiple object detection and tracking
            """)
        model = st.selectbox("Select yolo type model", ["Standard", "Fruit"])
        model_status = st.empty()
        if model == "Standard":
            with model_status.progress(0):
                self._load_standard()
                model_status.success("Standard Model loaded successfully!")
        elif model == "Fruit":
            with model_status.progress(0):
                self._load_fruit()
                model_status.success("Fruit-ed Model loaded successfully!")
        tab1, tab2, tab3 = st.tabs(["Upload Image", "Camera", "Database Images"])

        with tab1:
            from pages.utils import handle_image_upload
            header = "Object Detection through uploaded image"
            @handle_image_upload(header_text=header, button_text="YOLO that")
            def recognize_face(image):
                return self._yolo_that_image(image)
            result, _, img_out, image_status = recognize_face()

            if result is not None:
                img_out.image(result, caption="Processed Image", use_container_width=True)

        with tab2:
            pass
        with tab3:
            st.header("Not Implemented", anchor=False)

    def _yolo_that_image(self, image):
        return image

    def _load_standard(self):
        model_path = "resources/models/yolo11n.onnx"
        self.model = cv2.dnn.readNet(model_path)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _load_fruit(self):
        model_path = "resources/models/yolo_fruit.onnx"
        self.model = cv2.dnn.readNet(model_path)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if __name__ == "__main__":
    ObjectDetectionYoloPage().render()