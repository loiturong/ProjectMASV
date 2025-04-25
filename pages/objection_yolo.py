import streamlit as st
import cv2
import numpy as np
import ultralytics


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
            st.session_state.yolo_model = "Standard"
            with model_status.progress(0):
                self._load_standard()
                model_status.success("Standard Model loaded successfully!")
        elif model == "Fruit":
            st.session_state.yolo_model = "Fruit"
            with model_status.progress(0):
                self._load_fruit()
                model_status.success("Fruit-ed Model loaded successfully!")
        tab1, tab2, tab3 = st.tabs(["Upload Image", "Camera", "Database Images"])

        with tab1:
            from utils import handle_image_upload
            header = "Object Detection through uploaded image"
            @handle_image_upload(header_text=header, button_text="YOLO that")
            def recognize_face(image):
                return self._yolo_that_image(image)
            result, _, img_out, image_status = recognize_face()

            if result is not None:
                img_out.image(result, caption="Processed Image", use_container_width=True)

        with tab2:
            from utils import handle_camera_input
            @handle_camera_input(header_text="Object Detection in real-time", fps=24)
            def recognize_camera(image):
                return self._yolo_that_image(image)
            recognize_camera()

        with tab3:
            st.header("Not Implemented", anchor=False)

    def _yolo_that_image(self, image):
        # Convert uploaded image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Get original image dimensions
        h, w = image.shape[:2]
        
        # Check if image is larger than 640 in any dimension
        if w > 640 or h > 640:
            # Calculate scale factor to maintain aspect ratio
            scale = min(640 / w, 640 / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize the image while maintaining aspect ratio
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create a 640x640 canvas (black padding)
            padded_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Calculate padding offsets to center the image
            x_offset = (640 - new_w) // 2
            y_offset = (640 - new_h) // 2
            
            # Place the resized image on the canvas
            padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
            
            # Use the padded image for inference
            results = self.model(padded_image)
        else:
            # If image is smaller than 640x640, pad it directly
            padded_image = np.zeros((640, 640, 3), dtype=np.uint8)
            x_offset = (640 - w) // 2
            y_offset = (640 - h) // 2
            padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
            
            # Use the padded image for inference
            results = self.model(padded_image)
        
        return results[0].plot()

    def _load_standard(self):
        model_path = "resources/models/yolo11n.onnx"
        self.model = ultralytics.YOLO(model_path)

    def _load_fruit(self):
        model_path = "resources/models/yolo_fruit.onnx"
        self.model = ultralytics.YOLO(model_path)

if __name__ == "__main__":
    ObjectDetectionYoloPage().render()