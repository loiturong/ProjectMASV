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
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

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

if __name__ == "__main__":
    pass