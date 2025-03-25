import streamlit as st
import select
import traceback
import subprocess
import threading
import numpy as np
import imageio_ffmpeg
from queue import Queue, Empty
import time
import logging
import cv2

logger = logging.getLogger(__name__)

# Initialize video settings
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
        'prediction_reuse_frames': 2
    }

# Initialize other required state
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {
        'fps': 0,
        'processing_ms': 0,
        'queue_size': 0
    }

if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

# Check for required session state variables
required_keys = [
    "clarifai_pat",
    "clarifai_user_id",
    "clarifai_app_id",
    "clarifai_base_url",
    "models",
    "selected_video",  # Additional required keys for this page
    "selected_input_id"
]

# Check if all required keys exist in session state
if not all(key in st.session_state for key in required_keys):
    st.error("Please start from the home page and select a video first.")
    st.stop()
    st.switch_page("app.py")  # Redirect to main page

# from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.utils.logging import logger
from concurrent import futures
from queue import Queue, Empty  # Add Empty here
from utils import footer

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
import subprocess
import threading
import time
import datetime
import base64
import os

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def check_authentication():
    required_keys = [
        "clarifai_pat",
        "clarifai_user_id",
        "clarifai_app_id",
        "clarifai_base_url",
        "models"
    ]
    
    if not all(key in st.session_state for key in required_keys):
        st.error("Authentication required. Please return to the home page.")
        st.stop()

check_authentication()

# Apply global CSS (retrieved from session state)
if "global_css" in st.session_state:
  st.markdown(st.session_state["global_css"], unsafe_allow_html=True)

#st.title("🔍 Video Processing")


def modelhash(model):
  return model["Name"] + " (" + model["URL"] + ")"

# Configuration toggles
USE_STREAM = False  # if false then it does predict calls which can be in parallel
ENABLE_FPS_SYNC = True
THREADS = 8  # only used when USE_STREAM is False

ENABLE_DRAW_PREDICTIONS = True
FRAME_SKIP_INTERVAL = 3  # Reduced from 4 to 3 for smoother appearance
RESIZE_SIZE = (0.25, 0.25)  # Process at 25% of original size for better performance
MAX_QUEUE_SIZE = 120  # Increased buffer for smoother playback
PREDICTION_TIMEOUT = 1.0  # Reduced timeout to prevent stalling
BUFFER_SIZE = 30  # Increased buffer size
DISPLAY_DELAY = 0.033  # ~30 FPS (1/30 ≈ 0.033)
PREDICTION_REUSE_FRAMES = 3  # Reuse predictions for this many frames
# st.session_state.selected_input_id = "mXqSMCXTd8zFTBiL"
# st.session_state.selected_video = "https://videos.pexels.com/video-files/2796077/2796077-hd_1280_720_25fps.mp4"
# st.info("HARDCODED VIDEO: " + st.session_state.selected_video)

# Check if a video is selected
if "selected_video" not in st.session_state or not st.session_state.selected_video:
  st.warning("No video selected. Please select a video from the 'Video Selection' page.")
  st.stop()

# Initialize video settings if not already present
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
        'prediction_reuse_frames': 2
    }

# Initialize processing stats if not present
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {
        'fps': 0,
        'processing_ms': 0,
        'queue_size': 0
    }

# Initialize detection log if not present
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

# Then define your constants
MAX_QUEUE_SIZE = st.session_state.video_settings['max_queue_size']
PREDICTION_TIMEOUT = st.session_state.video_settings['prediction_timeout']
BUFFER_SIZE = st.session_state.video_settings['buffer_size']
DISPLAY_DELAY = 1.0 / st.session_state.video_settings['target_fps']
PREDICTION_REUSE_FRAMES = st.session_state.video_settings['prediction_reuse_frames']

# Add to the session state initialization section at the top
if 'detection_display_enabled' not in st.session_state:
    st.session_state.detection_display_enabled = False

# Retrieve Clarifai Authentication from session state
# if 'auth' not in st.session_state:
#   st.session_state.auth = ClarifaiAuthHelper.from_streamlit(st)
  # st.error("Authentication not initialized. Please go back to the Video Selection page.")
  # st.stop()

# auth = st.session_state.auth
pat = st.session_state["clarifai_pat"]
base_url = st.session_state["clarifai_base_url"]
root_certificates_path = st.session_state["clarifai_root_certificates_path"]
models = st.session_state["models"]

if 'stop_processing' not in st.session_state:
  st.session_state.stop_processing = False

if 'processing_stats' not in st.session_state:
  st.session_state.processing_stats = {}

# Display Selected Input ID and Thumbnail
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

# Model Selection Dropdown
model_names = [modelhash(model) for model in models]
selected_model_name = st.selectbox("Select a Model", model_names)
selected_model_url = next(
    model for model in models if modelhash(model) == selected_model_name
)

# Create two columns - one for video, one for detection log
video_col, log_col = st.columns([0.7, 0.3])

with video_col:
    # Create two columns for the video and metrics
    frame_placeholder = video_container = st.empty()  # For the video frame
    metrics_cols = st.columns(3)  # For the performance metrics
    
    # Display initial thumbnail if available
    if "selected_thumbnail" in st.session_state:
        frame_placeholder.image(st.session_state.selected_thumbnail, channels="RGB")
    
    # Display performance metrics below the video
    if st.session_state.get('processing_stats'):
        with metrics_cols[0]:
            st.metric("FPS", f"{st.session_state.processing_stats.get('fps', 0):.1f}")
        with metrics_cols[1]:
            st.metric("Processing", f"{st.session_state.processing_stats.get('processing_ms', 0):.1f}ms")
        with metrics_cols[2]:
            st.metric("Queue", st.session_state.processing_stats.get('queue_size', 0))
    
    # Add custom CSS for compact buttons
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
    
    # Main control buttons in a more compact layout
    with st.container():
        st.markdown('<div class="compact-buttons">', unsafe_allow_html=True)
        control_cols = st.columns([1, 1, 1, 3])  # The last column is empty for spacing
        
        with control_cols[0]:
            start_button = st.button("🚀 Start", use_container_width=True)
        with control_cols[1]:
            stop_button = st.button("⏹ Stop", use_container_width=True)
        with control_cols[2]:
            exit_button = st.button("❌ Exit", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with log_col:
    # Custom CSS for the detection log
    st.markdown("""
        <style>
        .detection-log {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            max-height: 600px;
            overflow-y: auto;
            background-color: #f5f5f5;
        }
        .stMarkdown {
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### Detection Log")
    
    # Add enable/disable toggle at the top
    detection_display_enabled = st.toggle(
        "Enable Live Updates",
        value=st.session_state.detection_display_enabled,
        help="Toggle live updates of detections. Disable for better video performance."
    )
    st.session_state.detection_display_enabled = detection_display_enabled
    
    # Controls in columns
    col1, col2 = st.columns(2)
    with col1:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.1)
    with col2:
        search_term = st.text_input("Search Labels", "")
    
    col3, col4 = st.columns(2)
    with col3:
        sort_by = st.selectbox("Sort By", ["Time", "Confidence", "Label"])
    with col4:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])
    
    if st.button("Clear Log", use_container_width=True):
        st.session_state.detection_log = []

    # Create a container for the log with custom styling
    log_container = st.container()

    def update_detection_log():
        if not st.session_state.detection_display_enabled:
            return
        
        # Filter log entries based on confidence and search term
        filtered_log = [
            entry for entry in st.session_state.detection_log
            if entry["confidence"] >= min_confidence and
            (search_term.lower() in entry["label"].lower() or not search_term)
        ]

        # Sort based on selected criteria
        if sort_by == "Time":
            filtered_log.sort(key=lambda x: x["timestamp"], reverse=(sort_order == "Descending"))
        elif sort_by == "Confidence":
            filtered_log.sort(key=lambda x: x["confidence"], reverse=(sort_order == "Descending"))
        else:  # Label
            filtered_log.sort(key=lambda x: x["label"], reverse=(sort_order == "Descending"))

        # Take only the last 10 entries
        display_log = filtered_log[:10]
        total_entries = len(filtered_log)

        with log_container:
            st.markdown('<div class="detection-log">', unsafe_allow_html=True)
            if not st.session_state.detection_display_enabled:
                st.markdown("<small>Live updates disabled</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"<small>Showing {len(display_log)} of {total_entries} detections</small>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
                
                for detection in display_log:
                    confidence = detection["confidence"]
                    confidence_color = (
                        "green" if confidence >= 0.7
                        else "orange" if confidence >= 0.4
                        else "red"
                    )
                    
                    st.markdown(f"""
                        <small>
                        **{detection["label"]}** (:{confidence_color}[{confidence:.2f}])  
                        Frame: {detection["frame"]} | {detection["timestamp"].strftime("%H:%M:%S.%f")[:-4]}
                        </small>
                        ---
                    """)
            st.markdown('</div>', unsafe_allow_html=True)

# Add settings sidebar
with st.sidebar:
    # Add Clarifai logo at the top using local PNG image
    logo_path = os.path.join("assets", "clarifai_logo.png")
    try:
        logo_base64 = get_base64_encoded_image(logo_path)
        st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <img src="data:image/png;base64,{logo_base64}" 
                     alt="Clarifai Logo" 
                     style="width: 100%">
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Clarifai logo not found in assets directory")
    
    st.header("Video Processing Settings")
    
    # Basic Settings
    st.subheader("Basic Settings")
    st.session_state.video_settings['use_stream'] = st.toggle(
        "Use Streaming Mode",
        value=st.session_state.video_settings['use_stream'],
        help="Enable for real-time streaming, disable for parallel processing."
    )
    
    st.session_state.video_settings['enable_draw_predictions'] = st.toggle(
        "Draw Predictions",
        value=st.session_state.video_settings['enable_draw_predictions'],
        help="Show bounding boxes and labels"
    )
    
    st.session_state.video_settings['target_fps'] = st.slider(
        "Target FPS",
        min_value=1,
        max_value=120,
        value=st.session_state.video_settings['target_fps'],
        help="Target frames per second for display"
    )
    
    # Advanced Settings Expander
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.video_settings['frame_skip_interval'] = st.number_input(
                "Frame Skip Interval",
                min_value=1,
                max_value=10,
                value=st.session_state.video_settings['frame_skip_interval'],
                help="Process every Nth frame"
            )
            
            st.session_state.video_settings['buffer_size'] = st.number_input(
                "Buffer Size",
                min_value=1,
                max_value=120,
                value=st.session_state.video_settings['buffer_size'],
                help="Number of frames to buffer"
            )
            
            st.session_state.video_settings['threads'] = st.number_input(
                "Thread Count",
                min_value=1,
                max_value=32,
                value=st.session_state.video_settings['threads'],
                help="Number of processing threads"
            )
            
        with col2:
            st.session_state.video_settings['resize_factor'] = st.slider(
                "Resize Factor",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.video_settings['resize_factor'],
                step=0.05,
                help="Scale factor for processing (smaller = faster)"
            )
            
            st.session_state.video_settings['prediction_timeout'] = st.slider(
                "Prediction Timeout",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.video_settings['prediction_timeout'],
                step=0.1,
                help="Maximum time to wait for predictions"
            )
            
            st.session_state.video_settings['prediction_reuse_frames'] = st.number_input(
                "Prediction Reuse",
                min_value=1,
                max_value=10,
                value=st.session_state.video_settings['prediction_reuse_frames'],
                help="Reuse predictions for N frames"
            )
    
    # Performance Presets
    st.subheader("Performance Presets")
    preset = st.selectbox(
        "Quick Settings",
        ["Custom", "Performance", "Balanced", "Quality"],
        index=0
    )
    
    if preset != "Custom":
        if st.button("Apply Preset"):
            if preset == "Performance":
                st.session_state.video_settings.update({
                    'frame_skip_interval': 4,
                    'resize_factor': 0.25,
                    'buffer_size': 15,
                    'prediction_reuse_frames': 3,
                    'target_fps': 30
                })
            elif preset == "Balanced":
                st.session_state.video_settings.update({
                    'frame_skip_interval': 3,
                    'resize_factor': 0.5,
                    'buffer_size': 30,
                    'prediction_reuse_frames': 2,
                    'target_fps': 45
                })
            elif preset == "Quality":
                st.session_state.video_settings.update({
                    'frame_skip_interval': 2,
                    'resize_factor': 0.75,
                    'buffer_size': 60,
                    'prediction_reuse_frames': 1,
                    'target_fps': 60
                })
            st.rerun()

# Update the constants based on session state
USE_STREAM = st.session_state.video_settings['use_stream']
ENABLE_FPS_SYNC = st.session_state.video_settings['enable_fps_sync']
THREADS = st.session_state.video_settings['threads']
ENABLE_DRAW_PREDICTIONS = st.session_state.video_settings['enable_draw_predictions']
FRAME_SKIP_INTERVAL = st.session_state.video_settings['frame_skip_interval']
RESIZE_SIZE = (st.session_state.video_settings['resize_factor'], st.session_state.video_settings['resize_factor'])
MAX_QUEUE_SIZE = st.session_state.video_settings['max_queue_size']
PREDICTION_TIMEOUT = st.session_state.video_settings['prediction_timeout']
BUFFER_SIZE = st.session_state.video_settings['buffer_size']
DISPLAY_DELAY = 1.0 / st.session_state.video_settings['target_fps']
PREDICTION_REUSE_FRAMES = st.session_state.video_settings['prediction_reuse_frames']

# Add performance metrics display
if st.session_state.get('processing_stats'):
    st.sidebar.subheader("Performance Metrics")
    stats = st.session_state.processing_stats
    st.sidebar.metric("Actual FPS", f"{stats.get('fps', 0):.1f}")
    st.sidebar.metric("Processing Time", f"{stats.get('processing_ms', 0):.1f}ms")
    st.sidebar.metric("Queue Size", stats.get('queue_size', 0))


class VideoProcessor:

  def __init__(self, model_url, video_url, pat):
    # Store video settings locally instead of accessing session state in threads
    self.video_settings = {
        'use_stream': st.session_state.video_settings.get('use_stream', False),
        'enable_fps_sync': st.session_state.video_settings.get('enable_fps_sync', True),
        'threads': st.session_state.video_settings.get('threads', 8),
        'enable_draw_predictions': st.session_state.video_settings.get('enable_draw_predictions', True),
        'frame_skip_interval': st.session_state.video_settings.get('frame_skip_interval', 3),
        'resize_factor': st.session_state.video_settings.get('resize_factor', 0.25),
        'max_queue_size': st.session_state.video_settings.get('max_queue_size', 60),
        'prediction_timeout': st.session_state.video_settings.get('prediction_timeout', 2.0),
        'buffer_size': st.session_state.video_settings.get('buffer_size', 30),
        'target_fps': st.session_state.video_settings.get('target_fps', 30),
        'prediction_reuse_frames': st.session_state.video_settings.get('prediction_reuse_frames', 2)
    }

    global RESIZE_SIZE  # Ensure global access

    # Initialize model first
    model_kwargs = {'url': model_url, 'pat': pat}
    
    if st.session_state.get("clarifai_base_url"):
        model_kwargs['base_url'] = st.session_state["clarifai_base_url"]
    
    if st.session_state.get("clarifai_root_certificates_path"):
        model_kwargs['root_certificates_path'] = st.session_state["clarifai_root_certificates_path"]

    self.detector_model = Model(**model_kwargs)

    # Then initialize video parameters
    self.model_url = model_url
    self.video_url = video_url
    self.pat = pat
    self.frame_width = 1920  # Default HD resolution
    self.frame_height = 1080
    self.input_fps = 30  # Default FPS
    
    # Initialize timing variables
    self.frame_interval = 1.0 / self.input_fps
    self.frame_interval *= 0.9  # Adjust for overhead
    self.last_frame_time = time.time()
    self.frame_times = []
    self.target_frame_time = 1.0 / st.session_state.video_settings.get('target_fps', 30)
    
    if isinstance(RESIZE_SIZE, (tuple, list)) and len(RESIZE_SIZE) == 2:
        self.resize_size = (float(RESIZE_SIZE[0]), float(RESIZE_SIZE[1]))
    else:
        logger.error(f"[ERROR] Invalid RESIZE_SIZE format: {RESIZE_SIZE}. Using default (1.0, 1.0)")
        self.resize_size = (1.0, 1.0)

    # Initialize other attributes
    self.frame_counter = 0
    self.stop_processing = False
    self.lock = threading.Lock()
    self.processed_frame_queue = Queue()
    self.last_prediction = None
    self.queue = Queue()
    self.predict_queue = Queue()
    self.decoded_frames = {}
    self.executor = futures.ThreadPoolExecutor(max_workers=THREADS)
    
    # Determine if using FFmpeg (RTSP/UDP) or OpenCV (MP4)
    self.use_ffmpeg = self.video_url.startswith(("rtsp://", "udp://"))
    
    # Initialize video capture
    if self.video_url.startswith("udp://"):
        self._init_udp_stream()
    else:
        self._init_regular_stream()

  def _init_udp_stream(self):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info(f"[INFO] Using FFmpeg path: {ffmpeg_path}")
    
    ffmpeg_cmd = [
        ffmpeg_path,
        "-fflags", "nobuffer",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", self.video_url,
        "-flags", "low_delay",
        "-strict", "experimental",
        "-vf", f"scale={self.frame_width}:{self.frame_height},format=bgr24",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-reorder_queue_size", "0",
        "-tune", "zerolatency",
        "-"
    ]
    
    logger.info(f"[INFO] UDP FFmpeg command: {' '.join(ffmpeg_cmd)}")
    
    try:
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        self.frame_size = self.frame_width * self.frame_height * 3
        logger.info(f"[INFO] Frame size calculated: {self.frame_size} bytes")
        self.last_frame_time = time.time()
        self.frame_interval = 1.0 / 30.0  # Force 30fps for UDP
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize UDP stream: {str(e)}")
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.kill()
        raise

  def _init_regular_stream(self):
    self.cap = cv2.VideoCapture(self.video_url)
    if not self.cap.isOpened():
        logger.info("[INFO] OpenCV capture failed, falling back to FFmpeg")
        # RTSP/HTTP handling with FFmpeg
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            probe_cmd = [
                ffmpeg_path.replace("ffmpeg", "ffprobe"), "-v", "error",
                "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0", self.video_url
            ]
            probe_output = subprocess.check_output(probe_cmd).decode("utf-8").strip()
            self.frame_width, self.frame_height = map(int, probe_output.split(","))
            logger.info(f"[INFO] FFmpeg detected resolution: {self.frame_width}x{self.frame_height}")
        except Exception as e:
            logger.info(f"[INFO] Using default resolution: {self.frame_width}x{self.frame_height}")
        
        ffmpeg_cmd = [
            ffmpeg_path,
            "-i", self.video_url,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-vf", f"scale={self.frame_width}:{self.frame_height},format=bgr24",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]

        # Start FFmpeg process
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

        # Calculate frame size
        self.frame_size = self.frame_width * self.frame_height * 3  # 3 bytes per pixel (BGR)
        self.use_ffmpeg = True  # Switch to FFmpeg mode
    else:
        # Get video properties from OpenCV
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[INFO] OpenCV capture initialized: {self.frame_width}x{self.frame_height} @ {self.input_fps}fps")
    
    # Set frame interval for playback synchronization
    self.frame_interval = 1.0 / self.input_fps if self.input_fps > 0 else DISPLAY_DELAY
    self.frame_interval *= 0.9  # Adjust to maintain FPS with Streamlit overhead

    # Add frame timing variables
    self.last_frame_time = time.time()
    self.frame_times = []
    self.target_frame_time = 1.0 / 30.0  # Target 30 FPS
    self.stats_update_interval = 0.5  # Update stats every 0.5 seconds
    self.last_stats_update = time.time()


  def prepare_frame(self, frame, frame_num):
    if frame is None:
      print(f"[WARNING] Received None frame at {frame_num}")
      return None, None
    
    try:
      # Basic resize without color conversion
      height, width = frame.shape[:2]
      target_width = int(width * RESIZE_SIZE[0])
      target_height = int(height * RESIZE_SIZE[1])
      
      resized_frame = cv2.resize(frame, (target_width, target_height), 
                                 interpolation=cv2.INTER_LINEAR)

      # Convert frame to bytes
      _, img_encoded = cv2.imencode('.jpg', resized_frame)
      img_bytes = img_encoded.tobytes()

      # Create input using Clarifai's Input class with input_id
      input_proto = Inputs.get_input_from_bytes(
        input_id=f"frame_{frame_num}",  # Create unique ID for each frame
        image_bytes=img_bytes
      )
      
      return input_proto, frame  # Return original frame for display
      
    except Exception as e:
      print(f"[ERROR] Frame preparation failed: {str(e)}")
      return None, None


  def draw_predictions(self, frame, predictions, frame_num, fps):
    try:
        print(f"[DEBUG] Drawing predictions for frame {frame_num}")
        
        DISPLAY_WIDTH = 1000
        height, width = frame.shape[:2]
        scale_factor = DISPLAY_WIDTH / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        thickness_factor = new_width / 640
        bbox_thickness = max(1, int(2 * thickness_factor))
        font_scale = max(0.4, min(1.2, 0.5 * thickness_factor))

        # Access first output's regions
        if predictions and predictions.outputs:
            regions = predictions.outputs[0].data.regions
            for region in regions:
                try:
                    bbox = region.region_info.bounding_box
                    
                    x1 = int(bbox.left_col * new_width)
                    y1 = int(bbox.top_row * new_height)
                    x2 = int(bbox.right_col * new_width)
                    y2 = int(bbox.bottom_row * new_height)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
                    
                    for concept in region.data.concepts:
                        label = f"{concept.name}: {concept.value:.2f}"
                        text_y = max(y1 - 10, 20)
                        cv2.putText(frame, label, (x1, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                  (0, 255, 0), bbox_thickness)
                        
                        detection = {
                            "label": concept.name,
                            "confidence": concept.value,
                            "frame": frame_num,
                            "timestamp": datetime.datetime.now()
                        }
                        st.session_state.detection_log.append(detection)
                        
                        if len(st.session_state.detection_log) > 1000:
                            st.session_state.detection_log.pop(0)
                except Exception as e:
                    print(f"[DEBUG] Error drawing region: {str(e)}")
                    continue

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), bbox_thickness)

        if not self.use_ffmpeg:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    except Exception as e:
        print(f"[DEBUG] Drawing error: {str(e)}")
        return frame


  def collect_frames(self):
    self.frame_counter = 0
    
    if self.use_ffmpeg:
        try:
            while not self.stop_processing:
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
        finally:
            if hasattr(self, 'ffmpeg_process'):
                self.ffmpeg_process.terminate()
    else:
        try:
            while not self.stop_processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("[DEBUG] No more frames from OpenCV")
                    break
                
                current_frame = self.frame_counter
                self._process_frame(frame.copy(), current_frame)
                self.frame_counter += 1
                
                time_elapsed = time.time() - self.last_frame_time
                if time_elapsed < self.frame_interval:
                    time.sleep(max(0, self.frame_interval - time_elapsed))
                self.last_frame_time = time.time()
                
        except Exception as e:
            print(f"[ERROR] OpenCV frame collection error: {str(e)}")
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()

  def _process_frame(self, frame, current_frame):
    try:
        fps_setting = self.video_settings['target_fps']
        frame_skip = max(1, int(30 / fps_setting))
        
        if current_frame % frame_skip == 0:
            input_proto, raw_frame = self.prepare_frame(frame, current_frame)
            if input_proto is not None:
                if self.video_settings['use_stream']:
                    self.queue.put(input_proto)
                    with self.lock:
                        self.decoded_frames[current_frame] = (None, frame)
                else:
                    try:
                        enable_predictions = self.video_settings['enable_draw_predictions']
                        if enable_predictions and not self.stop_processing:
                            fut = self.executor.submit(self.detector_model.predict, [input_proto])
                            with self.lock:
                                self.decoded_frames[current_frame] = (fut, frame)
                        else:
                            with self.lock:
                                self.decoded_frames[current_frame] = (None, frame)
                    except Exception as e:
                        print(f"[ERROR] Failed to submit prediction: {str(e)}")
                        with self.lock:
                            self.decoded_frames[current_frame] = (None, frame)
        else:
            with self.lock:
                self.decoded_frames[current_frame] = (None, frame)
    except Exception as e:
        print(f"[ERROR] Frame processing error: {str(e)}")
        with self.lock:
            self.decoded_frames[current_frame] = (None, frame)

  def frame_iterator(self):
    while not self.stop_processing:
        try:
            input_proto = self.queue.get(timeout=PREDICTION_TIMEOUT)
            if input_proto is not None:
                yield [input_proto]
            else:
                print("[DEBUG] Received None input_proto")
                time.sleep(0.1)
        except Empty:
            print("[DEBUG] Queue empty, waiting...")
            time.sleep(0.1)
        except Exception as e:
            print(f"[WARNING] Iterator error: {e}")
            time.sleep(0.1)

  def display_frames(self, frame_placeholder):
    initial_time = time.time()
    last_processed_frame = -1
    last_display_time = 0
    
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
            if ENABLE_DRAW_PREDICTIONS and self.last_prediction:
                display_frame = self.draw_predictions(display_frame, self.last_prediction, current_frame, fps)
            
            if not self.use_ffmpeg:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
            last_display_time = current_time
            
            # Update stats
            st.session_state.processing_stats = {
                'fps': fps,
                'processing_ms': (time.time() - current_time) * 1000,
                'queue_size': len(self.decoded_frames)
            }

        except Exception as e:
            print(f"[ERROR] Display error: {str(e)}")
            time.sleep(0.001)

  def stop(self):
    with self.lock:
      self.stop_processing = True
      self.executor.shutdown(wait=False)


# Handle button actions
if start_button:
    try:
        st.session_state.selected_model = selected_model_url["URL"]
        st.session_state.stop_processing = False
        
        status_placeholder = st.empty()
        status_placeholder.info("Initializing video processor...")
        
        # Create processor instance with proper initialization
        processor = VideoProcessor(
            st.session_state.selected_model,
            st.session_state.selected_video,
            pat
        )
        
        # Explicitly set required attributes
        processor.stop_processing = False
        processor.lock = threading.Lock()
        processor.frame_counter = 0
        processor.last_frame_time = time.time()
        processor.processed_frame_queue = Queue()
        processor.last_prediction = None
        processor.queue = Queue()
        processor.predict_queue = Queue()
        processor.decoded_frames = {}
        processor.executor = futures.ThreadPoolExecutor(max_workers=THREADS)
        
        # Add target frame time based on FPS settings
        target_fps = st.session_state.video_settings.get('target_fps', 30)  # Default to 30 if not set
        processor.target_frame_time = 1.0 / target_fps
        
        st.session_state.processor = processor
        
        status_placeholder.info("Starting video processing...")
        
        collection_thread = threading.Thread(
            target=processor.collect_frames,
            daemon=True
        )
        collection_thread.start()
        
        
        time.sleep(0.5)
        
        status_placeholder.info("Processing video...")
        processor.display_frames(frame_placeholder)
        
    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
        if "processor" in st.session_state:
            try:
                st.session_state.processor.stop()
            except:
                pass

if stop_button:
    if "processor" in st.session_state:
        try:
            st.session_state.processor.stop_processing = True
            st.session_state.processor.stop()
            st.session_state.stop_processing = True
            st.success("Processing stopped.")
        except Exception as e:
            st.warning(f"Error while stopping: {str(e)}")
    else:
        st.warning("No active process to stop.")

if exit_button:
    if "processor" in st.session_state:
        st.session_state.processor.stop()
        st.session_state.stop_processing = True
        del st.session_state.processor

    st.session_state.selected_video = None
    st.session_state.selected_input_id = None
    st.session_state.selected_thumbnail = None
    st.session_state.selected_model = None
    st.success("Exited to the Video Selection page.")
    st.switch_page("pages/1_Video_Selection.py")

# Footer
footer(st)
