import streamlit as st
import backoff
from clarifai.client.input import Inputs

st.markdown("""
    <style>
    /* Hide source map warnings */
    .st-emotion-cache-16txtl3 { display: none; }
    
    /* Improve clickable images styling */
    .clickable-images-container img {
        margin: 5px;
        border-radius: 5px;
        transition: transform 0.2s;
    }
    .clickable-images-container img:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

def manual_input_form():
    st.warning("Unable to fetch inputs from Clarifai. You can manually enter video URLs below.")
    with st.form("manual_video_urls"):
        st.markdown("""
        Enter video URLs, one per line. Supported formats:
        - RTSP streams (e.g., `rtsp://camera.example.com:554/stream`)
        - HTTP/HTTPS streams (e.g., `http://camera.example.com/video.mp4`)
        - UDP streams (e.g., `udp://239.0.0.1:1234`)
        - Local network cameras (e.g., `http://192.168.1.100:8080/video`)
        """)
        
        urls_text = st.text_area(
            "Video URLs (one per line)",
            height=150,
            help="Enter each video URL on a new line"
        )
        
        submitted = st.form_submit_button("Use These Videos")
        
        if submitted and urls_text.strip():

            st.session_state.manual_urls = urls_text.strip()
            return True
    return False

def create_mock_inputs(urls_text):
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    mock_inputs = []
    
    for i, url in enumerate(urls):
        mock_input = {
            "id": f"manual_input_{i}",
            "data": {
                "metadata": {
                    "stream_url": url
                }
            }
        }
        mock_inputs.append(mock_input)
    
    return mock_inputs

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def get_inputs_with_retry(user_id, pat, app_id, base_url=None, root_certificates_path=None):
    try:
        if base_url and root_certificates_path:
            try:
                inputs = Inputs(
                    user_id=user_id,
                    pat=pat,
                    app_id=app_id,
                    base_url=base_url,
                    root_certificates_path=root_certificates_path
                )
                input_list = list(inputs.list_inputs(
                    page_no=st.session_state.page_no,
                    per_page=per_page
                ))
                if input_list:
                    return input_list
            except Exception as e:
                #st.warning(f"Failed to connect using custom base URL and certificate: {str(e)}")
                #st.info("Falling back to manual video URL input.")
                return manual_input_form()

        # If no custom base_url/cert or if they failed, try default Clarifai endpoint
        try:
            inputs = Inputs(user_id=user_id, pat=pat, app_id=app_id)
            input_list = list(inputs.list_inputs(
                page_no=st.session_state.page_no,
                per_page=per_page
            ))
            if input_list:
                return input_list
            else:
                st.info("No inputs found in Clarifai. You can manually enter video URLs.")
                return manual_input_form()
        except Exception as e:
            #st.warning(f"Failed to connect to Clarifai: {str(e)}")
            #st.info("Falling back to manual video URL input.")
            return manual_input_form()
            
    except Exception as e:
        #st.error(f"Error retrieving inputs: {str(e)}")
        return manual_input_form()

required_keys = [
    "clarifai_pat",
    "clarifai_user_id",
    "clarifai_app_id",
    "clarifai_base_url",
    "models"
]


if not all(key in st.session_state for key in required_keys):
    st.error("Please enter your Clarifai credentials on the home page first.")
    st.stop()
    st.switch_page("app.py")  

from clarifai.client.user import User
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from st_clickable_images import clickable_images
from utils import footer
import grpc
import time
import base64
import cv2
import imageio_ffmpeg
import numpy as np
import re
import streamlit.components.v1 as components
import subprocess

if "global_css" in st.session_state:
  st.markdown(st.session_state["global_css"], unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Stream Selection</h1>", unsafe_allow_html=True)

pat = st.session_state["clarifai_pat"]
user_id = st.session_state["clarifai_user_id"]
app_id = st.session_state["clarifai_app_id"]
base_url = st.session_state["clarifai_base_url"]
root_certificates_path = st.session_state["clarifai_root_certificates_path"]
models = st.session_state["models"]

if 'page_no' not in st.session_state:
    st.session_state.page_no = 1

per_page = 25


def detect_video_type(video_url):
  if video_url.startswith("rtsp://"):
    print("RTSP stream detected.")
    return "rtsp"  # Use FFmpeg
  elif video_url.endswith(".mp4"):
    return "mp4"  # Use OpenCV first, FFmpeg only if necessary
  elif re.match(r'http[s]?://', video_url):
    print("HTTP/HTTPS stream detected.")
    return "opencv"  # Use OpenCV
  elif video_url.startswith("udp://"):
    print("UDP stream detected.")
    return "udp"  # Use FFmpeg
  return "unsupported"

@st.cache_data
def get_bright_frame(video_url, brightness_threshold=50, max_attempts=25):
  video_type = detect_video_type(video_url)
  if video_type == "udp":
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_cmd = [
      ffmpeg_path,
      "-fflags", "nobuffer",
      "-probesize", "32",
      "-analyzeduration", "0",
      "-i", video_url,
      "-map", "0:0",
      "-flags", "low_delay",
      "-strict", "experimental",
      "-frames:v", "1",
      "-f", "image2pipe",
      "-vcodec", "png",
      "-reorder_queue_size", "0",  # Disable reordering
      "-tune", "zerolatency",      # Optimize for low latency
      "-"
    ]
    for attempt in range(5):
      try:
        process = subprocess.Popen(
          ffmpeg_cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          bufsize=10**8
        )
        try:
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                return cv2.imdecode(np.frombuffer(stdout, np.uint8), cv2.IMREAD_COLOR)
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"[WARNING] FFmpeg timeout for UDP stream: {video_url} (Attempt {attempt+1})")
            continue
            
      except Exception as e:
        print(f"[WARNING] FFmpeg error for UDP stream: {str(e)}")
        if process:
            process.kill()
        continue
        
    return None

  if video_type in ["opencv", "mp4"]:
    cap = cv2.VideoCapture(video_url)
    frame = None
    for _ in range(max_attempts):
      ret, frame = cap.read()
      if not ret:
        break
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      brightness = np.mean(gray_frame)
      if brightness > brightness_threshold:
        break
    cap.release()

    if frame is not None:
      return frame

  if video_type == "rtsp" or frame is None:
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_cmd = [
      ffmpeg_path, "-rtsp_transport", "tcp", "-i", video_url,
      "-fflags", "nobuffer", "-flags", "low_delay", "-strict", "experimental",
      "-vf", "select=gte(scene\\,0.2)", "-frames:v", "1",
      "-f", "image2pipe", "-vcodec", "png", "-"
    ]

    for attempt in range(3):
      try:
        process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if process.stdout:
          return cv2.imdecode(np.frombuffer(process.stdout, np.uint8), cv2.IMREAD_COLOR)
      except subprocess.TimeoutExpired:
        print(f"[ERROR] FFmpeg timeout expired for: {video_url} (Attempt {attempt+1})")

  return None


def center_crop_image(image_bytesio, crop_size=128):
  img_array = np.frombuffer(image_bytesio.getvalue(), dtype=np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  h, w, _ = img.shape
  min_dim = min(h, w)
  start_x = (w - min_dim) // 2
  start_y = (h - min_dim) // 2
  cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
  cropped_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
  _, buffer = cv2.imencode(".png", cropped_img)
  return buffer.tobytes()


def to_base64_data_url(image_bytesio):
  cropped_bytes = center_crop_image(image_bytesio)
  base64_str = base64.b64encode(cropped_bytes).decode("utf-8")
  return f"data:image/png;base64,{base64_str}"

def process_input(inp):
    try:
        if isinstance(inp, dict):
            meta_stream_url = inp["data"]["metadata"]["stream_url"]
            input_id = inp["id"]
        else:
            meta_stream_url = inp.data.metadata['stream_url']
            input_id = inp.id

        bright_frame = get_bright_frame(meta_stream_url) 

        if bright_frame is not None:
            h, w, _ = bright_frame.shape
            _, buffer = cv2.imencode(".jpg", bright_frame)
            return BytesIO(buffer.tobytes()), (w, h), meta_stream_url, input_id
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return None, None, None, None


try:
    if 'manual_urls' in st.session_state:
        inputs = create_mock_inputs(st.session_state.manual_urls)
    else:
        inputs = get_inputs_with_retry(
            user_id=user_id,
            pat=pat,
            app_id=app_id,
            base_url=base_url,
            root_certificates_path=root_certificates_path
        )
        if isinstance(inputs, bool) and inputs:  # Form was just submitted
            st.rerun()

    if not inputs:
        st.error("No inputs available. Please either configure your Clarifai credentials correctly or enter video URLs manually.")
        st.stop()

    thumbnails, thumbnail_sizes, video_urls, input_ids = [], [], [], []

    with st.spinner("Loading inputs..."):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_input, inputs))
        for result in results:
            if result is not None and len(result) == 4:
                thumbnail, size, video_url, input_id = result
                if all([thumbnail, size, video_url, input_id]):
                    thumbnails.append(thumbnail)
                    thumbnail_sizes.append(size)
                    video_urls.append(video_url)
                    input_ids.append(input_id)

    if not video_urls:
        st.error("No valid video streams found. Please check your URLs and try again.")
        if 'manual_urls' in st.session_state:
            del st.session_state.manual_urls
        st.stop()

except Exception as e:
    st.error(f"An error occurred while processing inputs: {str(e)}")
    st.error("Please try refreshing the page or check your connection.")
    if 'manual_urls' in st.session_state:
        del st.session_state.manual_urls
    st.stop()

thumbnails, thumbnail_sizes, video_urls, input_ids = [], [], [], []

with st.spinner("Loading inputs..."):
  with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_input, inputs))

  for thumbnail, size, video_url, input_id in results:
    thumbnails.append(thumbnail)
    thumbnail_sizes.append(size)  # Store sizes in a list
    video_urls.append(video_url)
    input_ids.append(input_id)


empty_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgAB/SmMIOoAAAAASUVORK5CYII="

image_urls = [to_base64_data_url(img) if img else empty_image for img in thumbnails]

cols_per_row = 5
if len(image_urls) % cols_per_row != 0:
  padding_needed = cols_per_row - (len(image_urls) % cols_per_row)
  image_urls.extend([empty_image] * padding_needed)

clicked_index = clickable_images(
  image_urls,
  div_style={
    "display": "flex", 
    "justify-content": "center", 
    "flex-wrap": "wrap"
  },
  img_style={"margin": "5px", "height": "125px", "border-radius": "10px"},
  key="clickable_images_main"
)

if clicked_index > -1 and clicked_index < len(video_urls) and image_urls[clicked_index] != empty_image:
    st.session_state.selected_video = video_urls[clicked_index]
    st.session_state.selected_input_id = input_ids[clicked_index]
    st.session_state.selected_thumbnail = thumbnails[clicked_index]
    st.session_state.selected_thumbnail_size = thumbnail_sizes[clicked_index]
    st.session_state.video_loaded = True
    st.switch_page("pages/2_Model_Selection_and_Inference.py")

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
  if st.button("⬅ Previous Page", disabled=(st.session_state.page_no == 1)):
    if st.session_state.page_no > 1:
      st.session_state.page_no -= 1
      st.rerun()

with col2:
  st.markdown(
    f"""
    <div style="text-align: center; font-size: 18px; font-weight: bold;">
      Page {st.session_state.page_no}
    </div>
    """,
    unsafe_allow_html=True
  )

with col3:
  if st.button("Next Page ➡", disabled=(len(inputs) < per_page)):
    st.session_state.page_no += 1
    st.rerun()


footer(st)
