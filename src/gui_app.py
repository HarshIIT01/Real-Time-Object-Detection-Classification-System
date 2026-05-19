"""
Streamlit GUI Dashboard supporting both Classification and Detection.
"""
import streamlit as st
import numpy as np
import cv2
import os
import json
from PIL import Image
import time

st.set_page_config(page_title="Object Detection Dashboard", layout="wide", page_icon="🔍")

st.title("🔍 Real-Time Object Detection & Classification")
st.markdown("Upload an image for classification or detection inference.")


@st.cache_resource
def load_yolo_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)


@st.cache_resource
def load_tf_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def load_class_names(model_path):
    for d in [os.path.dirname(model_path), "outputs/models"]:
        p = os.path.join(d, "class_names.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return [f"Class_{i}" for i in range(80)]


# ── Sidebar ──
st.sidebar.header("⚙️ Configuration")
task = st.sidebar.selectbox("Task", ["Detection (YOLOv8)", "Classification (TF)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

if task.startswith("Detection"):
    default_model = "outputs/models/yolov8_run/weights/best.pt"
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
else:
    default_model = "outputs/models/best_model.keras"

model_path = st.sidebar.text_input("Model Path", default_model)

# ── Main Content ──
tabs = st.tabs(["📷 Image Upload", "📊 Model Info"])

with tabs[0]:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Handle color channels
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)

        with col2:
            if os.path.exists(model_path):
                t = time.time()

                if task.startswith("Detection"):
                    model = load_yolo_model(model_path)
                    results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
                    latency = (time.time() - t) * 1000

                    annotated = results[0].plot()
                    st.image(annotated[:, :, ::-1], caption="Detections", use_column_width=True)

                    detections = []
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        name = model.names[cls] if cls in model.names else f"Class_{cls}"
                        detections.append({"Class": name, "Confidence": f"{float(box.conf):.1%}"})

                    if detections:
                        st.dataframe(detections, use_container_width=True)
                    st.info(f"Found **{len(detections)} objects** in {latency:.0f}ms")

                else:
                    import tensorflow as tf
                    model = load_tf_model(model_path)
                    class_names = load_class_names(model_path)

                    resized = cv2.resize(img_array, (224, 224))
                    arr = np.expand_dims(resized, 0).astype(np.float32)
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                    preds = model.predict(arr, verbose=0)
                    latency = (time.time() - t) * 1000

                    idx = int(np.argmax(preds[0]))
                    conf = float(preds[0][idx])
                    label = class_names[idx] if idx < len(class_names) else f"Class_{idx}"

                    st.success(f"**{label}** — {conf:.1%} confidence")
                    st.info(f"Latency: {latency:.0f}ms")

                    # Top-5 predictions
                    top5 = np.argsort(preds[0])[-5:][::-1]
                    st.bar_chart({class_names[i] if i < len(class_names) else f"C_{i}": float(preds[0][i]) for i in top5})
            else:
                st.warning(f"Model not found: {model_path}")

with tabs[1]:
    st.header("Model Information")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.metric("Model Size", f"{size_mb:.1f} MB")
        st.text(f"Path: {model_path}")
    else:
        st.warning("Model file not found.")
