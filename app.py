import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import io
import os
from ultralytics import YOLO
import platform

# -----------------------------
# Load YOLO Segmentation Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("best(4).pt")  # Use the local file in the GitHub repo
    return model

model = load_model()

# -----------------------------
# Run YOLO Segmentation
# -----------------------------
def run_segmentation(pil_img):
    results = model(pil_img)[0]
    annotated_img = results.plot()  # YOLO's built-in annotated output
    return Image.fromarray(annotated_img[..., ::-1])  # Convert BGR to RGB for PIL

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Rice Disease Segmentation App")

# Check if running on cloud; warn about camera
if platform.system() == "Darwin" or platform.system() == "Linux":
    st.warning("Real-time camera segmentation may not work on Streamlit Cloud.")

mode = st.radio("Choose input method:", ["Upload from Gallery", "Take a Picture", "Real-time Camera Segmentation"])

# --- Upload from Gallery ---
if mode == "Upload from Gallery":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB").resize((640, 640))
        cols = st.columns(2)
        with cols[0]:
            st.image(image, caption="Original Image", use_container_width=True)
        with cols[1]:
            result = run_segmentation(image)
            st.image(result, caption="Segmented Output", use_container_width=True)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Segmented Image", buf.getvalue(), file_name="segmented.png", mime="image/png")

# --- Take a Picture ---
elif mode == "Take a Picture":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert("RGB").resize((640, 640))
        cols = st.columns(2)
        with cols[0]:
            st.image(image, caption="Original Image", use_container_width=True)
        with cols[1]:
            result = run_segmentation(image)
            st.image(result, caption="Segmented Output", use_container_width=True)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Segmented Image", buf.getvalue(), file_name="segmented.png", mime="image/png")

# --- Real-time Camera Segmentation ---
elif mode == "Real-time Camera Segmentation":
    st.warning("This works only in local environment. Press 'q' to stop the webcam.")

    if st.button("Start Real-time Segmentation"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB → PIL → run model
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 640))
            results = model(image_pil)[0]

            # Get YOLO annotated image (BGR)
            annotated = results.plot()

            # Convert BGR to RGB for Streamlit display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
