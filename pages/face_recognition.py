import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from face_recognition_onnx.utils import get_parser
from face_recognition_onnx.utils import draw_rectangle
from utils import handle_image_upload, handle_camera_input


class FaceRecognitionPage:
    """
    Class representing the Face Recognition page of the Machine Vision Application.
    """

    def __init__(self):
        """Initialize the FaceRecognitionPage instance."""
        self.title = ":rainbow[Face Recognition] :gray[Tool]"
        self.welcome_message = "Welcome to our Face Recognition Application! This tool provides capabilities of recognize your ex face."
        self.image_folder = "resources/Images"  # Path to your images directory
        self.recognizer = None
        self.detector = None
        self.svm = None

    def render(self):
        """Render the face recognition page content."""
        # Display the title and welcome message
        st.title(self.title, anchor=False)
        st.write(self.welcome_message)
        with st.expander("More Details About This Features"):
            st.write("""
            This application offers state-of-the-art machine vision capabilities:
            - Real-time face detection and recognition
            - Multiple object detection and tracking
            - Image classification using deep learning
            - Custom model training options
            """)
        model_status = st.empty()
        with model_status.progress(0):
            self._load_model()
            model_status.success("Model loaded successfully!")
        # Create tabs for different input options
        tab1, tab2, tab3 = st.tabs(["Upload Image", "Camera", "Database Images"])

        with tab1:
            header = "Face Recognition through uploaded image"
            @handle_image_upload(header_text=header, button_text="run Face recognition")
            def recognize_face(image):
                return self._process_detect_faces(image)
            result, img_in, img_out, image_status = recognize_face()

            if result is not None:
                processed_image = result

                # Display the processed image with face detections
                img_out.image(processed_image, caption="Processed Image", use_container_width=True)

                # Display the recognition results
                image_status.success(f"Recognized this one")
            else:
                if image_status is not None:
                    image_status.error("Failed to process image. Please try again with a clearer image.")

        with tab2:
            @handle_camera_input(header_text="Face Recognition in real-time")
            def recognize_camera(image):
                return self._process_detect_faces(image)
            recognize_camera()

        with tab3:
            st.header("Not Implemented", anchor=False)

    def _load_model(self):
        args = get_parser()
        detector = cv.FaceDetectorYN.create(
            args.face_detection_model,
            "",
            (320, 320),
            args.score_threshold,
            args.nms_threshold,
            args.top_k
        )
        recognizer = cv.FaceRecognizerSF.create(
            args.face_recognition_model, "")

        self.detector, self.recognizer = detector, recognizer
        self.svm = joblib.load('resources/models/svc.pkl')

    def _process_detect_faces(self, image):
        """
        Process the image with face recognition for multiple faces.

        Args:
            image: Image array from PIL (uploaded via st.file_uploader)

        Returns:
            Tuple of (processed_image, person_name, confidence) or None if failed
        """
        # Convert PIL image array to OpenCV format
        # PIL image from st.file_uploader is RGB, OpenCV expects BGR
        img_copy = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        # Set input size for the detector based on image dimensions
        frameWidth, frameHeight = img_copy.shape[1], img_copy.shape[0]
        self.detector.setInputSize([frameWidth, frameHeight])

        try:
            # Detect faces
            faces = self.detector.detect(img_copy)

            # If no faces detected, return early with error message
            if faces[1] is None:
                st.error("No faces detected in the image!")
                return None

            # Draw rectangles for all faces
            draw_rectangle(img_copy, faces, None)

            # Process each face
            for i, face in enumerate(faces[1]):
                # Align and extract features for this face
                face_align = self.recognizer.alignCrop(img_copy, face)
                face_feature = self.recognizer.feature(face_align)

                # Predict using SVM
                test_predict = self.svm.predict(face_feature)[0] if hasattr(self.svm, 'decision_function') else 0

                # Get raw decision scores for all classes
                decision_scores = self.svm.decision_function(face_feature)[0] if hasattr(self.svm, 'decision_function') else 0
                # Map numeric prediction to name
                mydict = ["Doan", "Hieu", "Dat", "Lap", "Loi"]
                result = mydict[test_predict]

                decision_scores = softmax(decision_scores)[test_predict]

                # Draw result on image - position text above each face
                # Get coordinates for this specific face
                coords_x = int(face[0])  # x coordinate
                coords_y = int(face[1] - 10)  # y coordinate (slightly above the face)
                if coords_y < 0:  # Ensure text is visible
                    coords_y = int(face[1] + face[3] + 20)  # Place below the face instead

                # Draw name and confidence
                cv.putText(img_copy, f"{result} ({decision_scores:.2f})",
                          (coords_x, coords_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert back to RGB for Streamlit display
            rgb_image = cv.cvtColor(img_copy, cv.COLOR_BGR2RGB)

            # Return the processed image and recognition results
            return rgb_image

        except Exception as e:
            # Log the full error for debugging
            st.error(f"Error in face detection/recognition: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
            return None


def _process_train_svm():
    pass

# Function to maintain backward compatibility with Class-Based API
def face_recognition():
    """Legacy function that creates and renders the FaceRecognitionPage."""
    this_page = FaceRecognitionPage()
    this_page.render()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    face_recognition()