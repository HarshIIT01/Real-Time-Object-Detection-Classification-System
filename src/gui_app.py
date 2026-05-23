"""
Real-Time Object Detection & Classification System — Premium Dashboard
Created with rich modern styling, complete features, and full hardware diagnostics.
"""
import sys
import os

# Add the project root to sys.path to resolve 'src' imports when running under Streamlit
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure TensorFlow GPU memory growth immediately to avoid allocating 100% VRAM.
# This prevents CUDA out-of-memory crashes when running YOLOv8 and TensorFlow models together.
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

import streamlit as st
import numpy as np
import cv2
import os
import json
import yaml
import time
from datetime import datetime
from PIL import Image

# Setup Page Configuration
st.set_page_config(
    page_title="RT-Vision Dashboard",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    /* Main Layout Styling */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #e0e0e0 !important;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Card Styles */
    .premium-card {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    
    /* Tabs Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 6px 6px 0px 0px;
        color: #9ca3af;
        border: 1px solid #374151;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
        border-color: #2563eb !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPERS & MODEL LOADING
# -----------------------------------------------------------------------------

@st.cache_resource
def load_yolo_model(model_path):
    """Load and cache YOLO model."""
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load YOLO model: {e}")
        return None

@st.cache_resource
def load_tf_model(model_path):
    """Load and cache TF/Keras model safely."""
    try:
        import tensorflow as tf
        # Import to register WarmupCosineSchedule custom schedule
        try:
            from src.lr_schedules import WarmupCosineSchedule
        except ImportError:
            pass
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.sidebar.error(f"Failed to load TF model: {e}")
        return None

def load_class_names(model_path):
    """Load class names from disk based on model path."""
    for d in [os.path.dirname(model_path), "outputs/models", "outputs/models/classification"]:
        p = os.path.join(d, "class_names.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def get_model_size_mb(path):
    """Return model file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    return 0.0

def load_system_config():
    """Load primary config YAML."""
    cfg_path = "configs/config.yaml"
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    return {}

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

st.sidebar.markdown("### 🛠️ Mode & Model Selection")
task_mode = st.sidebar.selectbox(
    "Active System Task",
    ["Object Detection (YOLOv8)", "Image Classification (TF/Keras)"],
    index=0
)

# Auto-detect default paths and configurations
config = load_system_config()
default_yolo_path = "outputs/models/yolov8_run/weights/best.pt"
default_tf_path = "outputs/models/classification/best_model.keras"

if task_mode.startswith("Object Detection"):
    default_model = default_yolo_path
    st.sidebar.markdown("### 🎯 Detection Hyperparameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("Intersection over Union (IoU)", 0.0, 1.0, 0.45, 0.05)
    use_tta = st.sidebar.checkbox("Enable Test-Time Augmentation (TTA)", value=False)
else:
    default_model = default_tf_path
    st.sidebar.markdown("### 🧪 Classification Options")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
    show_gradcam = st.sidebar.checkbox("Compute Grad-CAM Interpretability", value=True)

model_path = st.sidebar.text_input("Active Model Weights Path", default_model)

# Model Validation in Sidebar
if os.path.exists(model_path):
    st.sidebar.success("Model weights found!")
else:
    st.sidebar.warning("Path does not exist. Please check or input a valid path.")

# Device Selection
st.sidebar.markdown("### 💻 Hardware Device")
device_opt = st.sidebar.selectbox("Inference Device", ["CPU", "GPU (CUDA)", "Auto"], index=2)

# Check GPU availability
import torch
gpu_avail = torch.cuda.is_available()
st.sidebar.info(f"NVIDIA GPU Available: **{gpu_avail}**")

# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------

st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">⚡ RT-Vision Analytics Dashboard</h1>
        <p class="header-subtitle">
            Interact, evaluate, benchmark, and deploy real-time deep learning models. 
            Active Mode: <b>{"Detection (YOLOv8)" if task_mode.startswith("Object Detection") else "Classification (TF/Keras)"}</b>
        </p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Inference Demo",
    "📊 Evaluation Metrics",
    "⚡ Hardware Benchmarking",
    "📦 Export & Deployment",
    "ℹ️ System & Model Info"
])

# -----------------------------------------------------------------------------
# TAB 1: INFERENCE DEMO
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### Perform Live Inference")
    input_source = st.radio("Select Image Source", ["Upload Image File", "Use Webcam / Camera Input"], horizontal=True)
    
    uploaded_image = None
    if input_source == "Upload Image File":
        uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg", "webp"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
    else:
        cam_file = st.camera_input("Capture Image from Webcam")
        if cam_file:
            uploaded_image = Image.open(cam_file)
            
    if uploaded_image:
        st.toast("Image received! Preparing inference pipeline...")
        
        # Convert PIL to Numpy BGR/RGB
        img_np = np.array(uploaded_image)
        # Ensure 3-channel RGB format
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Input Image")
            st.image(uploaded_image, width="stretch")
            
        with col2:
            st.markdown("#### Model Output Predictions")
            
            if not os.path.exists(model_path):
                st.error(f"Cannot run inference: Model not found at '{model_path}'")
            else:
                status_block = st.status("Initializing model and processing...")
                try:
                    # Determine device string
                    if device_opt == "CPU":
                        dev_str = "cpu"
                    elif device_opt == "GPU (CUDA)":
                        dev_str = "0"
                    else:
                        dev_str = "0" if gpu_avail else "cpu"
                        
                    t_start = time.time()
                    
                    if task_mode.startswith("Object Detection"):
                        # YOLOv8 Path
                        status_block.write("Loading YOLO model...")
                        yolo = load_yolo_model(model_path)
                        
                        if yolo:
                            status_block.write("Running YOLO forward pass...")
                            # Predict
                            res = yolo.predict(
                                img_np,
                                conf=conf_threshold,
                                iou=iou_threshold,
                                augment=use_tta,
                                device=dev_str,
                                verbose=False
                            )
                            t_end = time.time()
                            latency_ms = (t_end - t_start) * 1000
                            
                            # Retrieve annotated image
                            annotated_bgr = res[0].plot()
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption="Annotated Bounding Boxes", width="stretch")
                            
                            # Table of detections
                            detections_list = []
                            for box in res[0].boxes:
                                cls_idx = int(box.cls)
                                name = yolo.names[cls_idx] if cls_idx in yolo.names else f"Class_{cls_idx}"
                                conf_val = float(box.conf)
                                xyxy = box.xyxy[0].tolist()
                                detections_list.append({
                                    "Class": name,
                                    "Confidence": f"{conf_val:.2%}",
                                    "Coordinates [x1, y1, x2, y2]": f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]"
                                })
                                
                            status_block.update(label="Inference finished successfully!", state="complete")
                            
                            # Metrics display
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Objects Found", len(detections_list))
                            m_col2.metric("Latency", f"{latency_ms:.1f} ms")
                            
                            if detections_list:
                                st.dataframe(detections_list, width="stretch")
                                
                                # Download annotated image
                                is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
                                if is_success:
                                    st.download_button(
                                        label="💾 Download Annotated Image",
                                        data=buffer.tobytes(),
                                        file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                        mime="image/jpeg"
                                    )
                            else:
                                st.info("No objects detected above the confidence threshold.")
                        else:
                            status_block.update(label="Inference failed.", state="error")
                            
                    else:
                        # TF/Keras Path
                        status_block.write("Loading TensorFlow environment...")
                        import tensorflow as tf
                        model = load_tf_model(model_path)
                        class_names = load_class_names(model_path)
                        
                        if model:
                            status_block.write("Preprocessing input tensor...")
                            # Read input shape from model
                            in_shape = model.input_shape
                            if isinstance(in_shape, list):
                                in_shape = in_shape[0]
                            h = in_shape[1] if (len(in_shape) > 1 and in_shape[1] is not None) else 224
                            w = in_shape[2] if (len(in_shape) > 2 and in_shape[2] is not None) else 224
                            
                            resized = cv2.resize(img_np, (w, h))
                            arr = np.expand_dims(resized, 0).astype(np.float32)
                            
                            # Resolve preprocessing
                            from src.model import get_preprocess_fn
                            backbone_name = config.get("model", {}).get("backbone", "MobileNetV2")
                            try:
                                preprocess_fn = get_preprocess_fn(backbone_name)
                                arr = preprocess_fn(arr)
                            except Exception:
                                arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                                
                            status_block.write("Running forward pass...")
                            with tf.device(f"/{'GPU' if dev_str != 'cpu' else 'CPU'}:0"):
                                preds = model.predict(arr, verbose=0)
                            t_end = time.time()
                            latency_ms = (t_end - t_start) * 1000
                            
                            pred_idx = int(np.argmax(preds[0]))
                            conf_val = float(preds[0][pred_idx])
                            
                            # Show Top Class
                            if class_names and pred_idx < len(class_names):
                                label = class_names[pred_idx]
                            else:
                                label = f"Class {pred_idx}"
                                
                            # Create Top 5 predictions list
                            top5_indices = np.argsort(preds[0])[-5:][::-1]
                            top5_preds = []
                            for idx in top5_indices:
                                c_name = class_names[idx] if (class_names and idx < len(class_names)) else f"Class_{idx}"
                                top5_preds.append({"Class": c_name, "Probability": float(preds[0][idx])})
                                
                            status_block.update(label="Inference finished successfully!", state="complete")
                            
                            # Metrics
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Predicted Class", label)
                            m_col2.metric("Confidence Score & Latency", f"{conf_val:.1%} ({latency_ms:.1f} ms)")
                            
                            # Display top prediction bar chart with Plotly
                            import plotly.express as px
                            fig = px.bar(
                                top5_preds,
                                x="Probability",
                                y="Class",
                                orientation="h",
                                title="Top-5 Classifier Confidences",
                                color="Probability",
                                color_continuous_scale="Blues"
                            )
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
                            st.plotly_chart(fig, width="stretch")
                            
                            # Grad-CAM overlay display
                            if show_gradcam:
                                status_block.write("Generating Grad-CAM overlays...")
                                try:
                                    from src.evaluate import make_gradcam_heatmap
                                    # Find last convolutional layer dynamically
                                    base_model = None
                                    for layer in model.layers:
                                        if isinstance(layer, tf.keras.Model):
                                            base_model = layer
                                            break
                                    target_model = base_model if base_model else model
                                    
                                    conv_layers = [l.name for l in target_model.layers if len(l.output_shape) == 4]
                                    if conv_layers:
                                        last_conv_name = conv_layers[-1]
                                        
                                        # Compute heatmap
                                        # If nested model, we need to extract feature maps from base model
                                        if base_model:
                                            # We need to construct nested Grad-CAM
                                            # For simplicity, load_model with compile=False and build a sub-grad model
                                            heatmap = make_gradcam_heatmap(arr, base_model, last_conv_name)
                                        else:
                                            heatmap = make_gradcam_heatmap(arr, model, last_conv_name, pred_index=pred_idx)
                                            
                                        # Compute overlay
                                        img_resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                                        heatmap_resized = cv2.resize(heatmap, (w, h))
                                        heatmap_norm = np.uint8(255 * heatmap_resized)
                                        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                                        superimposed = heatmap_color * 0.4 + img_resized_bgr
                                        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                                        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                                        
                                        st.image(superimposed_rgb, caption=f"Grad-CAM Heatmap Overlay (Layer: {last_conv_name})", width="stretch")
                                    else:
                                        st.warning("No 4D convolutional layers found to generate Grad-CAM.")
                                except Exception as gc_err:
                                    st.warning(f"Grad-CAM generation failed: {gc_err}")
                                    
                            # Download prediction data
                            st.download_button(
                                label="📊 Download Predictions JSON",
                                data=json.dumps(top5_preds, indent=2),
                                file_name=f"preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        else:
                            status_block.update(label="Inference failed.", state="error")
                            
                except Exception as ex:
                    st.exception(ex)
                    status_block.update(label="Error occurred during execution.", state="error")

# -----------------------------------------------------------------------------
# TAB 2: EVALUATION METRICS
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### System Accuracy & Metrics Reports")
    
    # Path settings
    eval_dir = "outputs/evaluation"
    
    if task_mode.startswith("Object Detection"):
        metrics_json_path = os.path.join(eval_dir, "detection_metrics.json")
        if os.path.exists(metrics_json_path):
            try:
                with open(metrics_json_path) as f:
                    metrics_data = json.load(f)
                
                st.success("Trained Object Detection validation metrics found!")
                
                # Metric Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("mAP@50 Accuracy", f"{metrics_data.get('mAP50', 0.0):.4f}")
                m2.metric("mAP@50-95 (COCO)", f"{metrics_data.get('mAP50_95', 0.0):.4f}")
                m3.metric("Validated Img Size", f"{metrics_data.get('imgsz', 640)} px")
                
                st.markdown("#### Full Valuation Parameters")
                st.json(metrics_data)
                
                # Check for ultralytics val plots
                val_run_dir = metrics_data.get("val_run_dir", "")
                if val_run_dir and os.path.exists(val_run_dir):
                    st.markdown("#### Ultralytics Evaluation Plots")
                    # List directory for png images
                    files = [f for f in os.listdir(val_run_dir) if f.endswith(".png") or f.endswith(".jpg")]
                    if files:
                        selected_plot = st.selectbox("Select Plot to View", files)
                        plot_path = os.path.join(val_run_dir, selected_plot)
                        st.image(plot_path, width="stretch")
                    else:
                        st.info("No plots found in the validation runs directory.")
            except Exception as e:
                st.error(f"Error reading validation metrics: {e}")
        else:
            st.warning("No detection metrics file found. Run validation using the evaluate script:")
            st.code("python -m src.evaluate --model outputs/models/yolov8_run/weights/best.pt --task detection")
            
    else:
        # Classification report & confusion matrix
        clf_report_path = os.path.join(eval_dir, "classification_report.json")
        cm_path = os.path.join(eval_dir, "confusion_matrix.png")
        
        if os.path.exists(clf_report_path):
            try:
                with open(clf_report_path) as f:
                    clf_data = json.load(f)
                    
                st.success("Trained Classification validation metrics found!")
                
                # Summary metrics
                acc_val = clf_data.get("accuracy", 0.0)
                m1, m2 = st.columns(2)
                m1.metric("Global Validation Accuracy", f"{acc_val:.2%}")
                m2.metric("Classes Evaluated", len([k for k in clf_data.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]))
                
                # Display classification report dataframe
                table_data = []
                for k, v in clf_data.items():
                    if isinstance(v, dict):
                        table_data.append({
                            "Class/Metric": k,
                            "Precision": f"{v.get('precision', 0.0):.4f}",
                            "Recall": f"{v.get('recall', 0.0):.4f}",
                            "F1-Score": f"{v.get('f1-score', 0.0):.4f}",
                            "Support": int(v.get('support', 0))
                        })
                st.dataframe(table_data, width="stretch")
                
            except Exception as e:
                st.error(f"Error reading classification report: {e}")
        else:
            st.warning("No classification report file found. Run validation using the evaluate script:")
            st.code("python -m src.evaluate --model outputs/models/classification/best_model.keras --task classification")
            
        if os.path.exists(cm_path):
            st.markdown("#### Confusion Matrix Plot")
            st.image(cm_path, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: HARDWARE BENCHMARKING
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### Run Hardware Performance Benchmarks")
    st.markdown("Compute exact throughput, average latency, and 95th-percentile (P95) tail latencies.")
    
    bench_model_path = st.text_input("Model Path to Benchmark", model_path, key="bench_model_path")
    bench_runs = st.slider("Number of Benchmark Warmups/Runs", 10, 200, 50, 10)
    
    # Initialize benchmark history in session state
    if "bench_history" not in st.session_state:
        st.session_state["bench_history"] = []
        
    if st.button("⚡ Run Latency & FPS Benchmark"):
        if not os.path.exists(bench_model_path):
            st.error(f"Cannot run benchmark: File does not exist at '{bench_model_path}'")
        else:
            status = st.status("Starting hardware benchmark...")
            try:
                ext = os.path.splitext(bench_model_path)[1].lower()
                
                # Run YOLO Benchmark
                if ext in ('.pt', '.pth', '.engine', '.onnx'):
                    status.write("Initializing YOLOv8 model for benchmarking...")
                    from ultralytics import YOLO
                    yolo = YOLO(bench_model_path)
                    
                    imgsz = config.get("data", {}).get("detection_imgsz", 640)
                    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                    
                    status.write(f"Warming up ({bench_runs // 5} runs)...")
                    for _ in range(max(2, bench_runs // 5)):
                        yolo.predict(dummy, verbose=False)
                        
                    status.write(f"Benchmarking ({bench_runs} runs)...")
                    latencies = []
                    for _ in range(bench_runs):
                        t0 = time.time()
                        yolo.predict(dummy, verbose=False)
                        latencies.append((time.time() - t0) * 1000)
                        
                # Run TF/Keras Benchmark
                else:
                    status.write("Initializing TensorFlow environment and model...")
                    import tensorflow as tf
                    from src.lr_schedules import WarmupCosineSchedule  # Import to register
                    model = tf.keras.models.load_model(bench_model_path, compile=False)
                    
                    in_shape = model.input_shape
                    if isinstance(in_shape, list):
                        in_shape = in_shape[0]
                    h = in_shape[1] if (len(in_shape) > 1 and in_shape[1] is not None) else 224
                    w = in_shape[2] if (len(in_shape) > 2 and in_shape[2] is not None) else 224
                    
                    dummy = np.random.random((1, h, w, 3)).astype(np.float32)
                    
                    status.write(f"Warming up ({bench_runs // 5} runs)...")
                    for _ in range(max(2, bench_runs // 5)):
                        model.predict(dummy, verbose=0)
                        
                    status.write(f"Benchmarking ({bench_runs} runs)...")
                    latencies = []
                    for _ in range(bench_runs):
                        t0 = time.time()
                        model.predict(dummy, verbose=0)
                        latencies.append((time.time() - t0) * 1000)
                
                # Compute Stats
                latencies = np.array(latencies)
                avg_l = np.mean(latencies)
                med_l = np.percentile(latencies, 50)
                p95_l = np.percentile(latencies, 95)
                fps = 1000 / avg_l
                sz_mb = get_model_size_mb(bench_model_path)
                
                status.update(label="Benchmark complete!", state="complete")
                
                # Display Metrics
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Average Latency", f"{avg_l:.2f} ms")
                c2.metric("Median Latency", f"{med_l:.2f} ms")
                c3.metric("P95 Tail Latency", f"{p95_l:.2f} ms")
                c4.metric("Throughput (FPS)", f"{fps:.1f}")
                c5.metric("Model Size", f"{sz_mb:.1f} MB")
                
                # Save to session history
                res_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model": os.path.basename(bench_model_path),
                    "Avg Latency (ms)": f"{avg_l:.2f}",
                    "P95 (ms)": f"{p95_l:.2f}",
                    "FPS": f"{fps:.1f}",
                    "Size (MB)": f"{sz_mb:.1f}"
                }
                st.session_state["bench_history"].append(res_entry)
                
            except Exception as e:
                st.error(f"Benchmark run failed: {e}")
                status.update(label="Benchmark failed.", state="error")
                
    if st.session_state["bench_history"]:
        st.markdown("#### Performance Run History")
        st.dataframe(st.session_state["bench_history"], width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: EXPORT & DEPLOYMENT
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### Export Models for Optimization & Edge Devices")
    
    exp_model_path = st.text_input("Model Weights Path to Export", model_path, key="exp_model_path")
    
    ext = os.path.splitext(exp_model_path)[1].lower()
    
    if ext in ('.pt', '.pth'):
        st.info("Detected YOLO PyTorch Model Weights.")
        yolo_fmt = st.selectbox("Export Format", ["onnx", "tflite", "engine", "torchscript"])
        
        if st.button("📦 Export YOLO Model"):
            if not os.path.exists(exp_model_path):
                st.error("Weights file does not exist.")
            else:
                status = st.status(f"Exporting YOLO to {yolo_fmt}...")
                try:
                    from src.export import export_yolo
                    # Extract args
                    imgsz = config.get("data", {}).get("detection_imgsz", 640)
                    export_yolo(exp_model_path, fmt=yolo_fmt, imgsz=imgsz)
                    status.update(label=f"YOLO successfully exported! Saved in weights parent folder.", state="complete")
                except Exception as e:
                    st.error(str(e))
                    status.update(label="Export failed.", state="error")
                    
    else:
        st.info("Detected TF/Keras Model Weights.")
        tf_fmt = st.selectbox("Export Format", ["TFLite (FP16 Quantized)", "TFLite (INT8 Quantized)", "ONNX Format"])
        
        if st.button("📦 Export Keras Model"):
            if not os.path.exists(exp_model_path):
                st.error("Weights file does not exist.")
            else:
                out_dir = os.path.join(os.path.dirname(exp_model_path), "exports")
                os.makedirs(out_dir, exist_ok=True)
                status = st.status(f"Converting classification model to {tf_fmt}...")
                
                try:
                    if tf_fmt.startswith("TFLite (FP16"):
                        from src.export import export_tflite
                        export_tflite(exp_model_path, out_dir, quantization="fp16")
                        status.update(label=f"Exported successfully! Saved in {out_dir}/model_fp16.tflite", state="complete")
                        
                    elif tf_fmt.startswith("TFLite (INT8"):
                        from src.export import export_tflite
                        from src.data_loader import DataLoader
                        # Require calibration dataset generator
                        status.write("Building calibration dataset generator...")
                        dl = DataLoader(config)
                        rep_gen = dl.get_representative_dataset()
                        
                        export_tflite(exp_model_path, out_dir, quantization="int8", representative_gen=rep_gen)
                        status.update(label=f"Exported successfully! Saved in {out_dir}/model_int8.tflite", state="complete")
                        
                    else:
                        from src.export import export_onnx
                        export_onnx(exp_model_path, out_dir)
                        status.update(label=f"Exported successfully! Saved in {out_dir}/model.onnx", state="complete")
                        
                except Exception as e:
                    st.error(str(e))
                    status.update(label="Export failed.", state="error")

# -----------------------------------------------------------------------------
# TAB 5: SYSTEM & MODEL INFO
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### System Environment & Active Model Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Hardware Diagnostics")
        
        import platform
        import psutil
        
        diag_data = {
            "OS Platform": f"{platform.system()} ({platform.release()})",
            "Python Version": platform.python_version(),
            "CPU Processor": platform.processor(),
            "CPU Cores (Logical)": psutil.cpu_count(logical=True),
            "System RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "CUDA Available": str(gpu_avail)
        }
        if gpu_avail:
            diag_data["GPU Device Name"] = torch.cuda.get_device_name(0)
            diag_data["GPU Memory Alloc"] = f"{torch.cuda.memory_allocated(0) / (1024**2):.1f} MB"
            
        st.json(diag_data)
        
        # System Config file
        st.markdown("#### System Configuration (config.yaml)")
        st.json(config)
        
    with col2:
        st.markdown("#### Active Model Metadata")
        
        if not os.path.exists(model_path):
            st.warning("No active model weights file found at the selected path.")
        else:
            model_info = {
                "Filename": os.path.basename(model_path),
                "Size (MB)": f"{get_model_size_mb(model_path):.2f} MB",
                "Full Path": os.path.abspath(model_path),
                "Last Modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            ext = os.path.splitext(model_path)[1].lower()
            if ext in ('.pt', '.pth'):
                model_info["Type"] = "YOLOv8 Object Detection (PyTorch)"
                # Extract classes
                yolo = load_yolo_model(model_path)
                if yolo:
                    model_info["Registered Classes Count"] = len(yolo.names)
                    model_info["Classes List"] = list(yolo.names.values())
            else:
                model_info["Type"] = "Classification (TensorFlow/Keras)"
                import tensorflow as tf
                tf_model = load_tf_model(model_path)
                if tf_model:
                    model_info["Input Shape"] = str(tf_model.input_shape)
                    model_info["Output Shape"] = str(tf_model.output_shape)
                    class_names = load_class_names(model_path)
                    if class_names:
                        model_info["Registered Classes Count"] = len(class_names)
                        model_info["Classes List"] = class_names
                        
            st.json(model_info)
