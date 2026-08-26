# ============================================================
# AI VISION CAPSTONE - COMPLETE STREAMLIT APPLICATION
# YOLO11 Object Detection + Live Camera + Dataset Analytics
# PYNQ / FPGA Capstone Dashboard
# ============================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from collections import Counter
import pandas as pd
import cv2
import av
import threading
import time
import io

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
DATASET_PATH = BASE_DIR / "dataset" / "test"
RESULTS_PATH = BASE_DIR / "results"

RESULTS_PATH.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Vision Capstone",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(40, 90, 160, 0.12),
                transparent 35%
            ),
            #0b0f17;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #0b1220 100%
        );
        border-right: 1px solid #263244;
    }

    /* Main title */
    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Section title */
    .section-title {
        font-size: 30px;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #9ca3af;
        margin-bottom: 25px;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(
            145deg,
            #151d2b,
            #101722
        );
        border: 1px solid #273449;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .metric-number {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 14px;
    }

    /* Information cards */
    .info-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 15px;
        padding: 22px;
        margin-bottom: 15px;
    }

    /* Status */
    .status-online {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.25);
        font-size: 13px;
        font-weight: 700;
    }

    .status-ready {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(59,130,246,0.12);
        color: #60a5fa;
        border: 1px solid rgba(59,130,246,0.25);
        font-size: 13px;
        font-weight: 700;
    }

    /* Detection result */
    .object-row {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 35px 0 15px 0;
        font-size: 13px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"YOLO model not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "dataset_results" not in st.session_state:
    st.session_state.dataset_results = []

if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_image(image, confidence=0.25):
    """
    Run YOLO detection on a PIL image.
    """

    start = time.perf_counter()

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    elapsed = time.perf_counter() - start

    result = results[0]

    counts = Counter()
    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[class_id]

            counts[class_name] += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append(
                {
                    "object": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                }
            )

    annotated = result.plot()

    return (
        result,
        counts,
        detections,
        annotated,
        elapsed
    )


def create_history_record(
    filename,
    counts,
    detections,
    elapsed
):

    total = sum(counts.values())

    avg_confidence = 0

    if detections:
        avg_confidence = (
            sum(
                d["confidence"]
                for d in detections
            )
            / len(detections)
        )

    return {
        "Image": filename,
        "Objects": total,
        "Types": len(counts),
        "Average Confidence": avg_confidence,
        "Inference Time": elapsed
    }


def metric_card(label, value, icon="📊"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {icon} &nbsp; {label}
            </div>
            <div class="metric-number">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def analyze_dataset(confidence=0.25):

    if not DATASET_PATH.exists():
        return []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    image_files = [
        p for p in DATASET_PATH.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    results_list = []

    progress = st.progress(0)

    status = st.empty()

    total = len(image_files)

    if total == 0:
        return []

    for index, image_path in enumerate(image_files):

        status.write(
            f"Analyzing **{image_path.name}** "
            f"({index + 1}/{total})"
        )

        try:

            image = Image.open(image_path).convert("RGB")

            (
                result,
                counts,
                detections,
                annotated,
                elapsed
            ) = detect_image(
                image,
                confidence
            )

            record = {
                "Image": image_path.name,
                "Total Objects": sum(counts.values()),
                "Inference Time (sec)": round(
                    elapsed,
                    4
                ),
                "Average Confidence": round(
                    (
                        sum(
                            d["confidence"]
                            for d in detections
                        ) / len(detections)
                    )
                    if detections
                    else 0,
                    3
                )
            }

            for object_name, count in counts.items():
                record[object_name] = count

            results_list.append(record)

        except Exception as error:

            results_list.append(
                {
                    "Image": image_path.name,
                    "Total Objects": 0,
                    "Inference Time (sec)": 0,
                    "Average Confidence": 0,
                    "Error": str(error)
                }
            )

        progress.progress(
            (index + 1) / total
        )

    status.success(
        "Dataset analysis completed successfully."
    )

    return results_list


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class LiveCameraDetector(VideoProcessorBase):

    def __init__(self):

        self.model = model

        self.confidence = 0.25

        self.lock = threading.Lock()

        self.object_counts = Counter()

        self.total_objects = 0

        self.last_inference_time = 0

        self.average_confidence = 0

    def recv(self, frame):

        start = time.perf_counter()

        image = frame.to_ndarray(
            format="bgr24"
        )

        try:

            results = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False
            )

            result = results[0]

            counts = Counter()

            confidence_values = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])

                    confidence = float(
                        box.conf[0]
                    )

                    class_name = self.model.names[
                        class_id
                    ]

                    counts[class_name] += 1

                    confidence_values.append(
                        confidence
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    # Bounding box
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label
                    label = (
                        f"{class_name} "
                        f"{confidence:.0%}"
                    )

                    cv2.rectangle(
                        image,
                        (
                            x1,
                            max(0, y1 - 28)
                        ),
                        (
                            x1 + 190,
                            y1
                        ),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        image,
                        label,
                        (
                            x1 + 4,
                            max(18, y1 - 8)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2
                    )

            elapsed = (
                time.perf_counter()
                - start
            )

            avg_conf = (
                sum(confidence_values)
                / len(confidence_values)
                if confidence_values
                else 0
            )

            with self.lock:

                self.object_counts = counts

                self.total_objects = sum(
                    counts.values()
                )

                self.last_inference_time = elapsed

                self.average_confidence = avg_conf

        except Exception as error:

            cv2.putText(
                image,
                "Camera detection error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">

        <div style="font-size:52px;">
        🎯
        </div>

        <h2>AI VISION</h2>

        <p style="color:#9ca3af;">
        PYNQ FPGA Capstone
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🔍 Image Detection",
            "📹 Live Camera",
            "🗂️ Dataset Analysis",
            "📊 Analytics",
            "⚡ Performance",
            "🧠 AI Model",
            "⚙️ PYNQ / FPGA",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.caption(
        f"Current threshold: {confidence:.2f}"
    )

    st.divider()

    st.markdown(
        """
        <span class="status-online">
        ● YOLO MODEL READY
        </span>
        """,
        unsafe_allow_html=True
    )

    st.caption("YOLO11n Object Detection")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🎯 AI Vision Object Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Intelligent computer vision platform for object detection,
        counting, analytics and PYNQ FPGA acceleration.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dashboard metrics

    total_objects = 0

    total_images = len(
        st.session_state.history
    )

    total_people = 0

    total_vehicles = 0

    for record in st.session_state.history:

        total_objects += record["Objects"]

    for record in st.session_state.dataset_results:

        total_objects += record.get(
            "Total Objects",
            0
        )

        total_people += record.get(
            "person",
            0
        )

        total_vehicles += (
            record.get("car", 0)
            + record.get("bus", 0)
            + record.get("truck", 0)
            + record.get("motorcycle", 0)
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Images Processed",
            total_images,
            "🖼️"
        )

    with c2:
        metric_card(
            "Objects Detected",
            total_objects,
            "🎯"
        )

    with c3:
        metric_card(
            "People Detected",
            total_people,
            "👤"
        )

    with c4:
        metric_card(
            "Vehicles Detected",
            total_vehicles,
            "🚗"
        )

    st.write("")

    # Quick actions

    st.markdown("## 🚀 Quick Actions")

    q1, q2, q3 = st.columns(3)

    with q1:

        st.markdown(
            """
            <div class="info-card">
            <h3>🔍 Image Detection</h3>
            <p>
            Upload an image and detect objects
            using YOLO11n.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q2:

        st.markdown(
            """
            <div class="info-card">
            <h3>📹 Live Camera</h3>
            <p>
            Detect objects directly from
            your device camera.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q3:

        st.markdown(
            """
            <div class="info-card">
            <h3>🗂️ Dataset Analysis</h3>
            <p>
            Analyze all images inside
            dataset/test automatically.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🔄 AI Pipeline")

    st.markdown(
        """
        <div class="info-card">

        <h3>Camera / Image → YOLO11n → Detection →
        Counting → Analytics → Hardware Acceleration</h3>

        <p style="color:#9ca3af;">
        The system receives visual input, performs object
        detection using YOLO11n, draws bounding boxes,
        counts detected classes and presents the results
        through an interactive dashboard.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

elif page == "🔍 Image Detection":

    st.markdown(
        '<div class="section-title">🔍 Image Object Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Upload an image and let YOLO11n identify and count
        objects automatically.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        with st.spinner(
            "Running YOLO object detection..."
        ):

            (
                result,
                counts,
                detections,
                annotated,
                elapsed
            ) = detect_image(
                image,
                confidence
            )

        # History
        record = create_history_record(
            uploaded_file.name,
            counts,
            detections,
            elapsed
        )

        # Avoid duplicate records
        existing_names = [
            x["Image"]
            for x in st.session_state.history
        ]

        if uploaded_file.name not in existing_names:

            st.session_state.history.append(
                record
            )

        st.success(
            "Object detection completed."
        )

        left, right = st.columns(2)

        with left:

            st.markdown("### Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with right:

            st.markdown(
                "### Detection Result"
            )

            st.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        st.markdown("## 📊 Detection Results")

        m1, m2, m3 = st.columns(3)

        with m1:
            metric_card(
                "Total Objects",
                sum(counts.values()),
                "🎯"
            )

        with m2:
            metric_card(
                "Object Types",
                len(counts),
                "📦"
            )

        with m3:

            avg = (
                sum(
                    d["confidence"]
                    for d in detections
                )
                / len(detections)
                if detections
                else 0
            )

            metric_card(
                "Average Confidence",
                f"{avg:.1%}",
                "🎯"
            )

        st.markdown("## 🔢 Object Counts")

        if counts:

            cols = st.columns(
                min(4, len(counts))
            )

            for index, (
                object_name,
                count
            ) in enumerate(
                counts.items()
            ):

                with cols[
                    index % len(cols)
                ]:

                    metric_card(
                        object_name.title(),
                        count,
                        "🔹"
                    )

        else:

            st.warning(
                "No objects detected. "
                "Try lowering the confidence threshold."
            )

        st.markdown("## ⚡ Inference Performance")

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "Inference Time",
                f"{elapsed * 1000:.2f} ms"
            )

        with p2:

            fps = (
                1 / elapsed
                if elapsed > 0
                else 0
            )

            st.metric(
                "Approx. FPS",
                f"{fps:.2f}"
            )

        # Download annotated image

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        annotated_pil = Image.fromarray(
            annotated_rgb
        )

        buffer = io.BytesIO()

        annotated_pil.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download Detection Result",
            data=buffer.getvalue(),
            file_name="detected_result.png",
            mime="image/png"
        )

    else:

        st.info(
            "Upload an image above to start object detection."
        )


# ============================================================
# LIVE CAMERA
# ============================================================

elif page == "📹 Live Camera":

    st.markdown(
        '<div class="section-title">📹 Live Camera Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Real-time object detection using your device camera.
        YOLO11n analyzes each camera frame and displays
        bounding boxes and object counts.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "Press START below. Your browser will ask for "
        "camera permission. Select Allow."
    )

    ctx = webrtc_streamer(
        key="ai-vision-live-camera",

        video_processor_factory=LiveCameraDetector,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )

    if ctx.video_processor:

        st.markdown("## 📊 Live Camera Statistics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            metric_card(
                "Objects",
                ctx.video_processor.total_objects,
                "🎯"
            )

        with c2:

            metric_card(
                "Object Types",
                len(
                    ctx.video_processor.object_counts
                ),
                "📦"
            )

        with c3:

            metric_card(
                "Confidence",
                f"{ctx.video_processor.average_confidence:.1%}",
                "🎯"
            )

        with c4:

            fps = (
                1
                / ctx.video_processor.last_inference_time
                if ctx.video_processor.last_inference_time > 0
                else 0
            )

            metric_card(
                "Inference FPS",
                f"{fps:.1f}",
                "⚡"
            )

        st.markdown(
            "## 🔎 Objects Currently Detected"
        )

        counts = (
            ctx.video_processor.object_counts
        )

        if counts:

            cols = st.columns(
                min(4, len(counts))
            )

            for index, (
                name,
                count
            ) in enumerate(
                counts.items()
            ):

                with cols[
                    index % len(cols)
                ]:

                    metric_card(
                        name.title(),
                        count,
                        "🔹"
                    )

        else:

            st.info(
                "No objects detected in the current frame."
            )

    else:

        st.info(
            "Camera is not running. Press START to activate it."
        )

    st.markdown(
        """
        ### 📱 Camera Access

        **Laptop/Desktop:**  
        Open the website in Chrome or Edge and click
        **Allow** when the camera permission appears.

        **Phone:**  
        Camera access normally requires a secure HTTPS
        connection when accessing the application from
        another device.
        """
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "🗂️ Dataset Analysis":

    st.markdown(
        '<div class="section-title">🗂️ Dataset Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-subtitle">
        Automatically analyze all images inside:
        <b>{DATASET_PATH}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    if DATASET_PATH.exists():

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        }

        image_files = [
            p
            for p in DATASET_PATH.iterdir()
            if p.suffix.lower()
            in image_extensions
        ]

        st.info(
            f"Found {len(image_files)} image(s) in the dataset."
        )

        if image_files:

            with st.expander(
                "📁 Dataset Images",
                expanded=False
            ):

                for path in image_files:
                    st.write(
                        f"• {path.name}"
                    )

            if st.button(
                "🚀 Analyze Complete Dataset",
                use_container_width=True
            ):

                results = analyze_dataset(
                    confidence
                )

                st.session_state.dataset_results = results

                st.rerun()

        else:

            st.warning(
                "No images found in dataset/test."
            )

    else:

        st.error(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    if st.session_state.dataset_results:

        st.markdown(
            "## 📊 Dataset Results"
        )

        df = pd.DataFrame(
            st.session_state.dataset_results
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # Overall object counts

        object_totals = Counter()

        for record in (
            st.session_state.dataset_results
        ):

            for key, value in record.items():

                if key not in [
                    "Image",
                    "Total Objects",
                    "Inference Time (sec)",
                    "Average Confidence",
                    "Error"
                ]:

                    if isinstance(
                        value,
                        (int, float)
                    ):

                        object_totals[key] += value

        st.markdown(
            "## 🔢 Total Objects Across Dataset"
        )

        if object_totals:

            cols = st.columns(
                min(5, len(object_totals))
            )

            for index, (
                name,
                count
            ) in enumerate(
                object_totals.most_common()
            ):

                with cols[
                    index % len(cols)
                ]:

                    metric_card(
                        name.title(),
                        count,
                        "🔹"
                    )

        # Download CSV

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Dataset Report",
            csv_data,
            "dataset_detection_report.csv",
            "text/csv"
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">📊 Detection Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Visual analysis of detected objects across your
        dataset and uploaded images.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.dataset_results:

        st.info(
            "Run Dataset Analysis first to generate analytics."
        )

    else:

        df = pd.DataFrame(
            st.session_state.dataset_results
        )

        ignored_columns = [
            "Image",
            "Total Objects",
            "Inference Time (sec)",
            "Average Confidence",
            "Error"
        ]

        object_columns = [
            column
            for column in df.columns
            if column not in ignored_columns
        ]

        totals = {}

        for column in object_columns:

            totals[column] = int(
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0).sum()
            )

        analytics_df = pd.DataFrame(
            {
                "Object": list(
                    totals.keys()
                ),
                "Count": list(
                    totals.values()
                )
            }
        )

        analytics_df = analytics_df[
            analytics_df["Count"] > 0
        ].sort_values(
            "Count",
            ascending=False
        )

        if not analytics_df.empty:

            st.markdown(
                "### 📈 Object Distribution"
            )

            st.bar_chart(
                analytics_df.set_index(
                    "Object"
                )
            )

            st.markdown(
                "### 📋 Object Summary"
            )

            st.dataframe(
                analytics_df,
                use_container_width=True,
                hide_index=True
            )

            top_object = analytics_df.iloc[0]

            metric_card(
                "Most Frequently Detected",
                f"{top_object['Object'].title()} ({int(top_object['Count'])})",
                "🏆"
            )

        else:

            st.warning(
                "No object statistics available."
            )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "⚡ Performance":

    st.markdown(
        '<div class="section-title">⚡ Performance Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Inference and processing performance of the
        computer vision pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.dataset_results:

        df = pd.DataFrame(
            st.session_state.dataset_results
        )

        avg_time = df[
            "Inference Time (sec)"
        ].mean()

        max_time = df[
            "Inference Time (sec)"
        ].max()

        avg_conf = df[
            "Average Confidence"
        ].mean()

        fps = (
            1 / avg_time
            if avg_time > 0
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Average Inference",
                f"{avg_time * 1000:.2f} ms",
                "⚡"
            )

        with c2:
            metric_card(
                "Maximum Inference",
                f"{max_time * 1000:.2f} ms",
                "⏱️"
            )

        with c3:
            metric_card(
                "Approx. FPS",
                f"{fps:.2f}",
                "🚀"
            )

        with c4:
            metric_card(
                "Average Confidence",
                f"{avg_conf:.1%}",
                "🎯"
            )

        st.markdown(
            "## ⏱️ Inference Time Per Image"
        )

        performance_df = df[
            [
                "Image",
                "Inference Time (sec)"
            ]
        ].set_index("Image")

        st.bar_chart(
            performance_df
        )

    else:

        st.info(
            "Analyze your dataset to see performance statistics."
        )

    st.markdown("## 💻 System Information")

    system_df = pd.DataFrame(
        [
            [
                "AI Model",
                "YOLO11n"
            ],
            [
                "Framework",
                "Ultralytics"
            ],
            [
                "Interface",
                "Streamlit"
            ],
            [
                "Computer Vision",
                "OpenCV"
            ],
            [
                "Image Processing",
                "Pillow"
            ],
            [
                "Data Analysis",
                "Pandas"
            ],
            [
                "Hardware Target",
                "PYNQ FPGA"
            ],
            [
                "Inference Format",
                "PyTorch / ONNX compatible"
            ]
        ],
        columns=[
            "Component",
            "Technology"
        ]
    )

    st.dataframe(
        system_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI MODEL
# ============================================================

elif page == "🧠 AI Model":

    st.markdown(
        '<div class="section-title">🧠 AI Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        YOLO11n based real-time object detection system.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Model",
            "YOLO11n",
            "🧠"
        )

    with c2:

        metric_card(
            "Task",
            "Object Detection",
            "🎯"
        )

    with c3:

        metric_card(
            "Framework",
            "Ultralytics",
            "⚙️"
        )

    st.markdown("## 🔬 How YOLO Works")

    st.markdown(
        """
        <div class="info-card">

        <h3>1️⃣ Input Image</h3>

        The system receives an image or camera frame.

        <br><br>

        <h3>2️⃣ Neural Network</h3>

        YOLO11 processes the image using a deep learning
        neural network.

        <br><br>

        <h3>3️⃣ Object Detection</h3>

        The model predicts object classes, bounding boxes
        and confidence scores.

        <br><br>

        <h3>4️⃣ Filtering</h3>

        Predictions below the selected confidence threshold
        are removed.

        <br><br>

        <h3>5️⃣ Counting</h3>

        Objects are grouped by class and counted.

        <br><br>

        <h3>6️⃣ Visualization</h3>

        Bounding boxes and labels are displayed to the user.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 📦 Supported Object Classes")

    names = model.names

    names_df = pd.DataFrame(
        [
            {
                "Class ID": index,
                "Object": name
            }
            for index, name in names.items()
        ]
    )

    st.dataframe(
        names_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PYNQ / FPGA
# ============================================================

elif page == "⚙️ PYNQ / FPGA":

    st.markdown(
        '<div class="section-title">⚙️ PYNQ / FPGA Acceleration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        Hardware acceleration layer for the AI vision capstone.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Hardware",
            "PYNQ FPGA",
            "⚙️"
        )

    with c2:

        metric_card(
            "AI Model",
            "YOLO11n",
            "🧠"
        )

    with c3:

        metric_card(
            "Pipeline",
            "AI Vision",
            "🚀"
        )

    st.markdown("## 🔄 Proposed Hardware Pipeline")

    st.markdown(
        """
        <div class="info-card">

        <h3>📷 Image / Camera</h3>

        ↓

        <h3>🧠 YOLO Object Detection</h3>

        ↓

        <h3>⚡ FPGA Hardware Acceleration</h3>

        ↓

        <h3>📊 Object Counting</h3>

        ↓

        <h3>💻 Streamlit Dashboard</h3>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🧩 Technology Stack")

    technologies = pd.DataFrame(
        [
            ["Programming", "Python"],
            ["Deep Learning", "YOLO11n"],
            ["Framework", "Ultralytics"],
            ["Computer Vision", "OpenCV"],
            ["Web Interface", "Streamlit"],
            ["Data Analysis", "Pandas"],
            ["Hardware Platform", "PYNQ FPGA"],
            ["Potential Acceleration", "ONNX / FPGA"],
        ],
        columns=[
            "Layer",
            "Technology"
        ]
    )

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The dashboard represents the software AI pipeline. "
        "Actual FPGA acceleration should only be marked as "
        "active when the hardware implementation is connected "
        "and verified."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
        AI-Based Object Detection and PYNQ FPGA Vision System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🎯 Project Objective")

    st.markdown(
        """
        The objective of this project is to develop an
        intelligent computer vision system capable of detecting,
        identifying and counting objects from images and
        real-time camera input.

        The system uses the YOLO11n deep learning model and
        provides an interactive dashboard for visualization,
        analytics and performance evaluation.
        """
    )

    st.markdown("## 🚨 Problem Statement")

    st.markdown(
        """
        Traditional image analysis requires significant manual
        effort for identifying and counting objects.

        This project provides an automated AI-based solution that
        can identify multiple objects in a scene and provide
        their counts and confidence scores.
        """
    )

    st.markdown("## 💡 Proposed Solution")

    st.markdown(
        """
        The proposed system combines:

        - YOLO11n object detection
        - Computer vision
        - Real-time camera processing
        - Dataset analysis
        - Object counting
        - Performance monitoring
        - Interactive web dashboard
        - PYNQ FPGA hardware acceleration concept
        """
    )

    st.markdown("## 🛠️ Technology Stack")

    stack = pd.DataFrame(
        [
            ["Language", "Python"],
            ["AI Model", "YOLO11n"],
            ["Deep Learning", "Ultralytics"],
            ["Computer Vision", "OpenCV"],
            ["Web Application", "Streamlit"],
            ["Data Processing", "Pandas"],
            ["Image Processing", "Pillow"],
            ["Live Video", "WebRTC"],
            ["Hardware", "PYNQ FPGA"],
        ],
        columns=[
            "Technology Area",
            "Technology"
        ]
    )

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## 🚀 Future Scope")

    st.markdown(
        """
        ### Possible future improvements

        1. FPGA-based YOLO acceleration
        2. ONNX hardware inference
        3. Object tracking
        4. Multi-camera support
        5. Vehicle counting
        6. Traffic monitoring
        7. Edge AI deployment
        8. Automatic report generation
        9. Cloud deployment
        10. Mobile-friendly monitoring
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <hr>

    🎯 <b>AI Vision Object Detection System</b>

    <br>

    YOLO11n • Python • Ultralytics • OpenCV • Streamlit •
    PYNQ FPGA

    <br><br>

    PYNQ AI Vision Capstone Project

    </div>
    """,
    unsafe_allow_html=True
)