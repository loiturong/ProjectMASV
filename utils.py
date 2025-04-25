import numpy as np
import streamlit as st
from functools import wraps
from PIL import Image


def handle_image_upload(header_text:str, button_text:str):
    """
    Decorator factory that creates a decorator for handling image upload and processing.

    Args:
        header_text: Custom text for header
        button_text: Custom text for the action button

    Returns:
        Decorator function that handles the complete upload-process-display flow
    """

    def decorator(recognition_function):
        @wraps(recognition_function)
        def wrapper(*args, **kwargs):
            st.header(header_text, anchor=False)
            input_img, output_img = st.columns(2)
            with input_img:
                input_img = st.empty()
                input_img.image("resources/flork.jpeg", caption="Place Holder Image", use_container_width=True)
            with output_img:
                output_img = st.empty()
                output_img.image("resources/flork.jpeg", caption="Recognized Place Holder Image",
                                 use_container_width=True)
            image_status = st.empty()

            # Create the file uploader widget with customizable text
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "mp4"])

            if uploaded_file is not None:
                # Read and display the uploaded image
                image = Image.open(uploaded_file)
                input_img.image(image, caption="Uploaded Image", use_container_width=True)

                # Process the image when user clicks the button (with custom text)
                if st.button(button_text):
                    with st.spinner("Processing image..."):
                        # Call the recognition function
                        result = recognition_function(image, *args, **kwargs)

                        return result, input_img, output_img, image_status
            return None, input_img, output_img, None

        return wrapper

    return decorator


def handle_camera_input(header_text: str, fps: int = 30, height: int = 640, width: int = 640,):
    """
    Decorator that handles webcam input and real-time processing.
    
    Args:
        header_text: Text displayed as header above the webcam stream
        :param width: Width of the webcam stream
        :param height: Height of the webcam stream
        :param fps: frames per second
        
    Returns:
        Decorator function that sets up the complete camera streaming functionality
    """
    def decorator(process_function):
        @wraps(process_function)
        def wrapper(*args, **kwargs):
            st.header(header_text, anchor=False)
            
            # Import required libraries
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
            import av
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            import threading
            
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    # Store the current script run context
                    self.ctx = threading.current_thread()
                    add_script_run_ctx(self.ctx)
                    # For controlling frame rate
                    self.frame_count = 0
                    self.frames_to_skip = max(1, int(30 / fps)) - 1  # Calculate frames to skip

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = np.flip(img, axis=1)

                    # Process the frame using the provided function
                    processed_result = process_function(img, *args, **kwargs)
                    
                    if processed_result is not None:
                        # Use the processed image
                        return av.VideoFrame.from_ndarray(processed_result, format="bgr24")
                    
                    # Return original frame if processing failed
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Create a webRTC streamer with the custom processor
            webrtc_streamer(
                key="stream-" + header_text.lower().replace(" ", "-"),
                video_processor_factory=lambda: VideoProcessor(),
                media_stream_constraints={"video": {
                        "frameRate": {"ideal": fps},
                        "width": {"ideal": width},
                        "height": {"ideal": height},
                    }, "audio": False},
                async_processing=True,
            )
            
        return wrapper
    return decorator

if __name__ == "__main__":
    pass