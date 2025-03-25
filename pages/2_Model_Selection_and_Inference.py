import streamlit as st
import subprocess
import threading
import numpy as np
import imageio_ffmpeg
from queue import Queue, Empty
import time
import tempfile
import logging
import cv2
import klvdata
import psutil
import datetime
import base64
import os
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from concurrent import futures
from utils import footer

logger = logging.getLogger(__name__)

# Initialize session state
if 'video_settings' not in st.session_state:
    st.session_state.video_settings = {
        'use_stream': False,
        'enable_fps_sync': True,
        'threads': 8,
        'enable_draw_predictions': True,
        'frame_skip_interval': 3,
        'resize_factor': 0.25,
        'max_queue_size': 60,
        'prediction_timeout': 2.0,
        'buffer_size': 30,
        'target_fps': 30,
        'prediction_reuse_frames': 2,
        'play_video_only': False
    }



if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {'fps': 0, 'processing_ms': 0, 'queue_size': 0}

if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

if 'detection_display_enabled' not in st.session_state:
    st.session_state.detection_display_enabled = True

if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Check authentication
required_keys = ["clarifai_pat", "clarifai_user_id", "clarifai_app_id", "clarifai_base_url", "models", "selected_video", "selected_input_id"]
if not all(key in st.session_state for key in required_keys):
    st.error("Please start from the home page and select a video first.")
    st.stop()

# Apply global CSS
if "global_css" in st.session_state:
    st.markdown(st.session_state["global_css"], unsafe_allow_html=True)

# Constants updated from session state
MAX_QUEUE_SIZE = st.session_state.video_settings['max_queue_size']
PREDICTION_TIMEOUT = st.session_state.video_settings['prediction_timeout']
BUFFER_SIZE = st.session_state.video_settings['buffer_size']
DISPLAY_DELAY = 1.0 / st.session_state.video_settings['target_fps']
PREDICTION_REUSE_FRAMES = st.session_state.video_settings['prediction_reuse_frames']

# Helper function for logo
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# UI Layout


st.markdown(
    f"""
    <div style="margin-bottom: -8px;">
        <strong>Input ID:</strong> <code>{st.session_state.selected_input_id}</code>
    </div>
    <div>
        <strong>Stream URL:</strong> <code>{st.session_state.selected_video}</code>
    </div>
    """,
    unsafe_allow_html=True
)

# Model Selection
model_names = [model["Name"] + " (" + model["URL"] + ")" for model in st.session_state["models"]]
selected_model_name = st.selectbox("Select a Model", model_names, disabled=False)
selected_model_url = next(model for model in st.session_state["models"] if model["Name"] + " (" + model["URL"] + ")" == selected_model_name)

# Columns for video and log
video_col, log_col = st.columns([0.7, 0.3])

with video_col:
    # Use st.session_state.selected_thumbnail as the initial frame
    frame_placeholder = st.empty()
    frame_placeholder.image(st.session_state.selected_thumbnail, channels="RGB", use_container_width=True)
    

    # Custom CSS for compact buttons
    st.markdown("""
        <style>
        .compact-buttons .stButton > button {
            padding: 0.2rem 1rem;
            font-size: 0.8rem;
            margin-top: -1rem;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Buttons container
    with st.container():
        st.markdown('<div class="compact-buttons">', unsafe_allow_html=True)
        control_cols = st.columns(3)  # Three equal columns
        with control_cols[0]:
            start_button = st.button("Start", use_container_width=True, disabled=False)
        with control_cols[1]:
            stop_button = st.button("Stop", use_container_width=True, disabled=False)
        with control_cols[2]:
            exit_button = st.button("Exit", use_container_width=True, disabled=False)
        st.markdown('</div>', unsafe_allow_html=True)

with log_col:
    #st.markdown("### Detection Log")

    # Performance Metrics Section
    # st.markdown("**Performance Metrics**")
    # metrics_container = st.container()
    # with metrics_container:
    #     stats = st.session_state.processing_stats
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("FPS", f"{stats.get('fps', 0):.1f}")
    #     with col2:
    #         st.metric("Proc. Time", f"{stats.get('processing_ms', 0):.1f}ms")
    #     with col3:
    #         st.metric("Queue", stats.get('queue_size', 0))

    # st.markdown("---")

    # Detection Log Controls
    

    st.session_state.min_confidence = st.slider("Min Confidence", 0.0, 1.0, st.session_state.get('min_confidence', 0.8), 0.1, disabled=False)
    st.session_state.detection_display_enabled = st.toggle(
        "Enable Live Updates",
        value=st.session_state.detection_display_enabled,
        disabled=False,
        help="Toggle live updates of detections. Disable for better video performance."
    )
    

    col2, col3, col4 = st.columns(3)
    with col2:
        st.session_state.search_term = st.text_input("Search", st.session_state.get('search_term', ""), disabled=not st.session_state.detection_display_enabled)
    with col3:
        st.session_state.sort_by = st.selectbox("Sort By", ["Time", "Confidence", "Label"], index=["Time", "Confidence", "Label"].index(st.session_state.get('sort_by', "Time")), disabled=not st.session_state.detection_display_enabled)
    with col4:
        st.session_state.sort_order = st.selectbox("Order", ["Descending", "Ascending"], index=["Descending", "Ascending"].index(st.session_state.get('sort_order', "Descending")), disabled=not st.session_state.detection_display_enabled)
    col5, col6 = st.columns(2)
    with col5:
        st.session_state.max_detections = st.slider("Max Display", min_value=5, max_value=50, value=10, step=5, disabled=not st.session_state.detection_display_enabled)
    with col6:
        st.session_state.max_detections_log = st.slider("Max Log", min_value=50, max_value=500, value=100, step=50, disabled=not st.session_state.detection_display_enabled)

    if st.button("Clear Log", use_container_width=True, disabled=not st.session_state.detection_display_enabled):
        st.session_state.detection_log = []

    log_container = st.empty()

def update_detection_log():
    if not st.session_state.detection_display_enabled:
        log_container.write('<div class="detection-log"><small>Stop video before adjusting controls to ensure stable performance.</small></div>', unsafe_allow_html=True)
        return

    min_confidence = st.session_state.get('min_confidence', 0.5)
    search_term = st.session_state.get('search_term', "")
    sort_by = st.session_state.get('sort_by', "Time")
    sort_order = st.session_state.get('sort_order', "Descending")
    max_detections_to_display = st.session_state.get('max_detections', 10)  # Limit number displayed
    max_detections_to_keep = st.session_state.get('max_detections_log', 100)  # Limit log size

    # Trim log to maintain max storage limit
    if len(st.session_state.detection_log) > max_detections_to_keep:
        st.session_state.detection_log = st.session_state.detection_log[-max_detections_to_keep:]

    # Filter log based on confidence and search term
    filtered_log = [
        entry for entry in st.session_state.detection_log
        if entry["confidence"] >= min_confidence and
        (search_term.lower() in entry["label"].lower() or not search_term)
    ]

    # Sort log based on the selected criteria
    if sort_by == "Time":
        filtered_log.sort(key=lambda x: x["timestamp"], reverse=(sort_order == "Descending"))
    elif sort_by == "Confidence":
        filtered_log.sort(key=lambda x: x["confidence"], reverse=(sort_order == "Descending"))
    else:  # Sort by label
        filtered_log.sort(key=lambda x: x["label"], reverse=(sort_order == "Descending"))
    
    # Limit the number of detections displayed
    display_log = filtered_log[-max_detections_to_display:]
    total_entries = len(filtered_log)

    # Generate HTML for the log
    log_html = '<div class="detection-log">'
    log_html += f'<small>Showing {len(display_log)} of {total_entries} detections (Max Display: {max_detections_to_display}, Log Limit: {max_detections_to_keep})</small><hr>'
    
    for detection in display_log:
        confidence = detection["confidence"]
        confidence_color = "red" if confidence >= 0.9 else "orange" if confidence >= 0.5 else "green"
        log_html += (
            f'<small><b>{detection["label"]}</b> '
            f'(<span style="color: {confidence_color}">{confidence:.2f}</span>) '
            f'F: {detection["frame"]} | {detection["timestamp"].strftime("%H:%M:%S.%f")[:-4]}</small>'
            f'<br><hr style="margin: 2px 0;">'
        )

    log_html += '</div>'
    log_container.write(log_html, unsafe_allow_html=True)


# Sidebar Settings
with st.sidebar:
    logo_path = os.path.join("assets", "clarifai_logo.png")
    try:
        logo_base64 = get_base64_encoded_image(logo_path)
        st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <img src="data:image/png;base64,{logo_base64}" alt="Clarifai Logo" style="width: 100%">
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Clarifai logo not found in assets directory")

    st.subheader("Basic Settings")
    st.session_state.video_settings['use_stream'] = st.toggle("Use Streaming Mode", value=st.session_state.video_settings['use_stream'], disabled=True, help="Clarifai Compute Orchestration is required for model streaming.")
    st.session_state.video_settings['enable_draw_predictions'] = st.toggle("Draw Predictions", value=st.session_state.video_settings['enable_draw_predictions'], disabled=False)
    st.session_state.video_settings['play_video_only'] = st.toggle("Play Video Only", value=st.session_state.video_settings['play_video_only'], disabled=False, help="Play video without processing.")
    st.session_state.video_settings['target_fps'] = st.slider("Target FPS", 1, 120, st.session_state.video_settings['target_fps'], disabled=False)


    with st.expander("Advanced Settings"):
    # Ensure video_settings exists in session state
        if 'video_settings' not in st.session_state:
            st.session_state.video_settings = {}

        # Ensure numeric defaults, avoiding DeltaGenerator issues
        if 'resize_factor' not in st.session_state.video_settings:
            st.session_state.video_settings['resize_factor'] = 0.5  # Default numeric value

        if 'prediction_timeout' not in st.session_state.video_settings:
            st.session_state.video_settings['prediction_timeout'] = 1.0

        if 'frame_skip_interval' not in st.session_state.video_settings:
            st.session_state.video_settings['frame_skip_interval'] = 1

        if 'buffer_size' not in st.session_state.video_settings:
            st.session_state.video_settings['buffer_size'] = 50

        if 'threads' not in st.session_state.video_settings:
            st.session_state.video_settings['threads'] = 4

        if 'prediction_reuse_frames' not in st.session_state.video_settings:
            st.session_state.video_settings['prediction_reuse_frames'] = 3

        # Row for sliders
        slider_col1, slider_col2 = st.columns(2)

        with slider_col1:
            resize_factor = st.slider(
                "Resize Factor", 0.1, 1.0, float(st.session_state.video_settings['resize_factor']), 0.05,
                disabled=False, help="Image Scaling for Inference."
            )
            st.session_state.video_settings['resize_factor'] = resize_factor  # Store only the float value

        with slider_col2:
            prediction_timeout = st.slider(
                "Prediction Timeout", 0.1, 5.0, float(st.session_state.video_settings['prediction_timeout']), 0.1,
                disabled=False
            )
            st.session_state.video_settings['prediction_timeout'] = prediction_timeout

        # Columns for number input boxes
        col1, col2, col3 = st.columns(3)

        with col1:
            frame_skip_interval = st.number_input(
                "Skip", 1, 30, int(st.session_state.video_settings['frame_skip_interval']), disabled=False
            )
            st.session_state.video_settings['frame_skip_interval'] = frame_skip_interval

        with col2:
            buffer_size = st.number_input(
                "Buffer", 1, 200, int(st.session_state.video_settings['buffer_size']), disabled=False
            )
            st.session_state.video_settings['buffer_size'] = buffer_size

        with col3:
            threads = st.number_input(
                "Thread", 1, 32, int(st.session_state.video_settings['threads']), disabled=False
            )
            st.session_state.video_settings['threads'] = threads

        # Separate row for Prediction Reuse
        prediction_reuse = st.number_input(
            "Prediction Reuse", 1, 10, int(st.session_state.video_settings['prediction_reuse_frames']),
            disabled=False, help="How many frames to reuse predictions for."
        )
        st.session_state.video_settings['prediction_reuse_frames'] = prediction_reuse




    st.subheader("Performance Presets")
    preset = st.selectbox("Quick Settings", ["Custom", "Performance", "Balanced", "Quality"], index=0, disabled=False)
    if preset != "Custom" and st.button("Apply Preset", disabled=False):
        presets = {
            "Performance": {'frame_skip_interval': 4, 'resize_factor': 0.25, 'buffer_size': 15, 'prediction_reuse_frames': 3, 'target_fps': 30},
            "Balanced": {'frame_skip_interval': 3, 'resize_factor': 0.5, 'buffer_size': 30, 'prediction_reuse_frames': 2, 'target_fps': 45},
            "Quality": {'frame_skip_interval': 2, 'resize_factor': 0.75, 'buffer_size': 60, 'prediction_reuse_frames': 1, 'target_fps': 60}
        }
        st.session_state.video_settings.update(presets[preset])
        st.rerun()

# VideoProcessor Class
class VideoProcessor:
    def __init__(self, model_url, video_url, pat):
        self.threads = []
        self.active_threads = set()
        self.stop_processing = False
        self.lock = threading.Lock()
        self.video_settings = st.session_state.video_settings.copy()
        self.detector_model = Model(url=model_url, pat=pat, base_url=st.session_state.get("clarifai_base_url"), root_certificates_path=st.session_state.get("clarifai_root_certificates_path"))
        self.model_url = model_url
        self.video_url = video_url
        self.pat = pat
        self.frame_width = 1920
        self.frame_height = 1080
        self.input_fps = 30
        self.frame_interval = 1.0 / self.input_fps
        self.target_frame_time = 1.0 / self.video_settings['target_fps']
        self.resize_size = (self.video_settings['resize_factor'], self.video_settings['resize_factor'])
        self.frame_counter = 0
        self.processed_frame_queue = Queue()
        self.last_prediction = None
        self.queue = Queue()
        self.predict_queue = Queue()
        self.decoded_frames = {}
        self.executor = futures.ThreadPoolExecutor(max_workers=self.video_settings['threads'])
        self.klv_metadata = {}
        self.use_ffmpeg = self.video_url.startswith(("rtsp://", "udp://", "http://", "https://"))
        if self.use_ffmpeg:
            self._init_regular_stream()

    def _init_regular_stream(self):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        try:
            probe_cmd = [ffmpeg_path.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", self.video_url]
            probe_output = subprocess.check_output(probe_cmd).decode().strip()
            width, height, fps_str = probe_output.split(",")
            self.frame_width, self.frame_height = int(width), int(height)
            num, denom = map(int, fps_str.split("/"))
            self.input_fps = num / denom if denom != 0 else 30.0
        except Exception as e:
            logger.warning(f"[WARN] FFmpeg probe failed: {str(e)}. Using defaults.")
            self.frame_width, self.frame_height, self.input_fps = 1280, 720, 30.0

        ffmpeg_cmd = [ffmpeg_path, "-i", self.video_url, "-fflags", "nobuffer", "-flags", "low_delay", "-strict", "experimental", "-vf", f"scale={self.frame_width}:{self.frame_height},format=bgr24", "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-loglevel", "verbose", "-"]
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        time.sleep(0.5)
        if self.ffmpeg_process.poll() is not None:
            raise Exception("FFmpeg process terminated unexpectedly")
        self.frame_size = self.frame_width * self.frame_height * 3
        self.frame_interval = 1.0 / self.input_fps * 0.9

        def log_ffmpeg_stderr():
            while True:
                line = self.ffmpeg_process.stderr.readline()
                if not line:
                    break
                logger.debug(f"[FFmpeg STDERR] {line.decode().strip()}")

        stderr_thread = threading.Thread(target=log_ffmpeg_stderr, daemon=True)
        stderr_thread.start()
        self.threads.append(stderr_thread)

    def prepare_frame(self, frame, frame_num):
        if frame is None:
            return None, None
        try:
            height, width = frame.shape[:2]
            target_width = int(width * self.resize_size[0])
            target_height = int(height * self.resize_size[1])
            resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            _, img_encoded = cv2.imencode('.jpg', resized_frame)
            img_bytes = img_encoded.tobytes()
            input_proto = Inputs.get_input_from_bytes(input_id=f"frame_{frame_num}", image_bytes=img_bytes)
            return input_proto, frame
        except Exception as e:
            print(f"[ERROR] Frame preparation failed: {str(e)}")
            return None, None

    def draw_predictions(self, frame, predictions, frame_num, fps):
        try:
            DISPLAY_WIDTH = 1000
            height, width = frame.shape[:2]
            scale_factor = DISPLAY_WIDTH / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            thickness_factor = new_width / 640
            bbox_thickness = max(1, int(2 * thickness_factor))
            font_scale = max(0.4, min(1.2, 0.5 * thickness_factor))
            min_confidence = st.session_state.min_confidence
            if predictions and predictions.outputs:
                for region in predictions.outputs[0].data.regions:
                    bbox = region.region_info.bounding_box
                    x1, y1 = int(bbox.left_col * new_width), int(bbox.top_row * new_height)
                    x2, y2 = int(bbox.right_col * new_width), int(bbox.bottom_row * new_height)
                    
                    for concept in region.data.concepts:
                        if concept.value < min_confidence:
                            continue
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
                        label = f"{concept.name}: {concept.value:.2f}"
                        text_y = max(y1 - 10, 20)
                        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), bbox_thickness)
                        st.session_state.detection_log.append({
                            "label": concept.name,
                            "confidence": concept.value,
                            "frame": frame_num,
                            "timestamp": datetime.datetime.now()
                        })
                        if len(st.session_state.detection_log) > 1000:
                            st.session_state.detection_log.pop(0)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), bbox_thickness)
            return frame
        except Exception as e:
            print(f"[DEBUG] Drawing error: {str(e)}")
            return frame

    def collect_frames(self):
        self.frame_counter = 0
        while not self.stop_processing:
            try:
                raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)
                if not raw_frame:
                    time.sleep(0.01)
                    continue
                frame = np.frombuffer(raw_frame, np.uint8).reshape((self.frame_height, self.frame_width, 3))
                current_frame = self.frame_counter
                self._process_frame(frame.copy(), current_frame)
                self.frame_counter += 1
            except Exception as e:
                print(f"[ERROR] FFmpeg frame collection error: {str(e)}")
                break
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process is not None:
            self.ffmpeg_process.terminate()

    def _process_frame(self, frame, current_frame):
        try:
            if self.video_settings['play_video_only']:
                with self.lock:
                    self.decoded_frames[current_frame] = (None, frame)
            else:
                fps_setting = self.video_settings['target_fps']
                frame_skip = max(1, int(self.input_fps / fps_setting))
                if current_frame % frame_skip == 0:
                    input_proto, raw_frame = self.prepare_frame(frame, current_frame)
                    if input_proto is not None:
                        if self.video_settings['use_stream']:
                            self.queue.put(input_proto)
                            with self.lock:
                                self.decoded_frames[current_frame] = (None, frame)
                        else:
                            fut = None
                            if self.video_settings['enable_draw_predictions'] and not self.stop_processing:
                                fut = self.executor.submit(self.detector_model.predict, [input_proto])
                            with self.lock:
                                self.decoded_frames[current_frame] = (fut, frame)
                    else:
                        with self.lock:
                            self.decoded_frames[current_frame] = (None, frame)
                else:
                    with self.lock:
                        self.decoded_frames[current_frame] = (None, frame)
        except Exception as e:
            print(f"[ERROR] Frame processing error: {str(e)}")
            with self.lock:
                self.decoded_frames[current_frame] = (None, frame)

    def display_frames(self, frame_placeholder):
        initial_time = time.time()
        last_display_time = 0
        last_log_update = 0
        while not self.stop_processing:
            try:
                current_time = time.time()
                if current_time - last_display_time < self.target_frame_time:
                    time.sleep(0.001)
                    continue
                with self.lock:
                    if not self.decoded_frames:
                        time.sleep(0.001)
                        continue
                    current_frame = min(self.decoded_frames.keys())
                    fut, frame = self.decoded_frames.pop(current_frame)
                if frame is None:
                    continue
                if fut is not None:
                    try:
                        prediction = fut.result(timeout=PREDICTION_TIMEOUT)
                        if prediction:
                            self.last_prediction = prediction
                    except Exception as e:
                        print(f"[ERROR] Prediction failed: {str(e)}")
                fps = self.frame_counter / max(0.001, current_time - initial_time)
                display_frame = frame.copy()
                if not self.video_settings['play_video_only'] and self.video_settings['enable_draw_predictions'] and self.last_prediction:
                    display_frame = self.draw_predictions(display_frame, self.last_prediction, current_frame, fps)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.session_state.selected_thumbnail = display_frame
                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                last_display_time = current_time
                st.session_state.processing_stats = {
                    'fps': fps,
                    'processing_ms': (time.time() - current_time) * 1000,
                    'queue_size': len(self.decoded_frames)
                }
                if current_time - last_log_update >= 1.0:
                    update_detection_log()
                    last_log_update = current_time
            except Exception as e:
                print(f"[ERROR] Display error: {str(e)}")
                time.sleep(0.001)
        # Update log one last time after stopping
        update_detection_log()

    def start_processing(self):
        self.cleanup_old_threads()
        collection_thread = threading.Thread(target=self.collect_frames, name=f"collect_frames_{time.time()}", daemon=True)
        self.threads.append(collection_thread)
        self.active_threads.add(collection_thread.name)
        collection_thread.start()

    def cleanup_old_threads(self):
        self.threads = [t for t in self.threads if t.is_alive()]
        self.active_threads = {t.name for t in self.threads}

    def stop(self):
        with self.lock:
            self.stop_processing = True
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
                try:
                    parent = psutil.Process(self.ffmpeg_process.pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    if self.ffmpeg_process:
                        self.ffmpeg_process.kill()
                finally:
                    self.ffmpeg_process = None
            for thread in self.threads:
                thread.join(timeout=2)
            self.threads.clear()
            self.active_threads.clear()
            self.executor.shutdown(wait=False, cancel_futures=True)
            for q in [self.processed_frame_queue, self.queue, self.predict_queue]:
                while not q.empty():
                    q.get_nowait()

# Button Actions
if start_button:
    try:
        st.session_state.selected_model = selected_model_url["URL"]
        st.session_state.stop_processing = False
        st.session_state.is_processing = True
        status_placeholder = st.empty()
        status_placeholder.info("Initializing video processor...")
        processor = VideoProcessor(st.session_state.selected_model, st.session_state.selected_video, st.session_state["clarifai_pat"])
        st.session_state.processor = processor
        status_placeholder.info("Starting video processing...")
        processor.start_processing()
        time.sleep(0.5)
        status_placeholder.info("Processing video...")
        processor.display_frames(frame_placeholder)
    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
        if "processor" in st.session_state:
            st.session_state.processor.stop()
            del st.session_state.processor
        st.session_state.is_processing = False

if stop_button:
    if "processor" in st.session_state:
        try:
            st.session_state.processor.stop()
            st.session_state.processor.cleanup_old_threads()
            del st.session_state.processor
            st.session_state.stop_processing = True
            st.session_state.is_processing = False
            st.success("Processing stopped.")
            update_detection_log()  # Ensure log remains visible
            frame_placeholder.image(st.session_state.selected_thumbnail, channels="RGB", use_container_width=True)
        except Exception as e:
            st.warning(f"Error while stopping: {str(e)}")
            if "processor" in st.session_state:
                del st.session_state.processor
            st.session_state.is_processing = False

if exit_button:
    if "processor" in st.session_state:
        st.session_state.processor.stop()
    st.session_state.selected_video = None
    st.session_state.selected_input_id = None
    st.session_state.selected_thumbnail = None
    st.session_state.selected_model = None
    st.session_state.detection_log = []  # Clear log only on exit
    st.session_state.is_processing = False
    st.success("Exited to the Video Selection page.")
    st.switch_page("pages/1_Video_Selection.py")

footer(st)