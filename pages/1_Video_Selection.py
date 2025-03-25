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
    """Displays a form for manual video URL input when API connection fails."""
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
            # Store the URLs in session state
            st.session_state.manual_urls = urls_text.strip()
            return True
    return False

def create_mock_inputs(urls_text):
    """Create mock inputs from URLs text."""
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
    """
    Retrieves inputs from Clarifai with retry logic and fallback to manual input.
    """
    try:
        # First try with custom base_url and certificates if provided
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
                st.warning(f"Failed to connect using custom base URL and certificate: {str(e)}")
                st.info("Falling back to manual video URL input.")
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
            st.warning(f"Failed to connect to Clarifai: {str(e)}")
            st.info("Falling back to manual video URL input.")
            return manual_input_form()
            
    except Exception as e:
        st.error(f"Error retrieving inputs: {str(e)}")
        return manual_input_form()

# Check for required session state variables
required_keys = [
    "clarifai_pat",
    "clarifai_user_id",
    "clarifai_app_id",
    "clarifai_base_url",
    "models"
]

# Check if all required keys exist in session state
if not all(key in st.session_state for key in required_keys):
    st.error("Please enter your Clarifai credentials on the home page first.")
    st.stop()
    st.switch_page("app.py")  # Redirect to main page

# Rest of your imports
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

# Apply global CSS (retrieved from session state)
if "global_css" in st.session_state:
  st.markdown(st.session_state["global_css"], unsafe_allow_html=True)

# Streamlit Config
st.markdown("<h1 style='text-align: center;'>Stream Selection</h1>", unsafe_allow_html=True)


# Initialize Clarifai Authentication and store in session state
# if 'auth' not in st.session_state:
#   st.session_state.auth = ClarifaiAuthHelper.from_streamlit(st)

# auth = st.session_state.auth
# userDataObject = auth.get_user_app_id_proto()

pat = st.session_state["clarifai_pat"]
user_id = st.session_state["clarifai_user_id"]
app_id = st.session_state["clarifai_app_id"]
base_url = st.session_state["clarifai_base_url"]
root_certificates_path = st.session_state["clarifai_root_certificates_path"]
models = st.session_state["models"]

# Initialize page number in session state if not exists
if 'page_no' not in st.session_state:
    st.session_state.page_no = 1

# Pagination Config
per_page = 25  # Now handling up to 5x5 grid per page


def detect_video_type(video_url):
  """Determines if the video source should be handled by OpenCV or FFmpeg."""
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

# Function to get a bright frame for thumbnails
@st.cache_data
def get_bright_frame(video_url, brightness_threshold=50, max_attempts=25):
  """Extracts a bright frame using OpenCV for MP4/HTTP streams and FFmpeg only for RTSP or fallback."""
  video_type = detect_video_type(video_url)

  # For UDP streams, use FFmpeg directly
  if video_type == "udp":
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_cmd = [
      ffmpeg_path,
      "-fflags", "nobuffer",
      "-probesize", "32",
      "-analyzeduration", "0",
      "-i", video_url,
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
        # Increase buffer size and timeout
        process = subprocess.Popen(
          ffmpeg_cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          bufsize=10**8
        )
        
        # Read with timeout
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

  # Try OpenCV first for MP4 and HTTP sources
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

  # Use FFmpeg ONLY for RTSP streams or if OpenCV fails
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
  """Center-crop an image to a square of `crop_size` without distorting aspect ratio."""
  img_array = np.frombuffer(image_bytesio.getvalue(), dtype=np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

  # Get image dimensions
  h, w, _ = img.shape
  min_dim = min(h, w)
  start_x = (w - min_dim) // 2
  start_y = (h - min_dim) // 2

  # Crop to square
  cropped_img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

  # Resize to final crop size
  cropped_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

  # Convert back to bytes
  _, buffer = cv2.imencode(".png", cropped_img)
  return buffer.tobytes()


def to_base64_data_url(image_bytesio):
  """Convert center-cropped image to base64 data URL."""
  cropped_bytes = center_crop_image(image_bytesio)
  base64_str = base64.b64encode(cropped_bytes).decode("utf-8")
  return f"data:image/png;base64,{base64_str}"


# Function to process a single input (wrapper for multithreading)
def process_input(inp):
    """Processes a single input by extracting a bright frame and returning relevant metadata."""
    try:
        # Handle both Clarifai Input objects and our mock dictionary inputs
        if isinstance(inp, dict):
            # For manual/mock inputs
            meta_stream_url = inp["data"]["metadata"]["stream_url"]
            input_id = inp["id"]
        else:
            # For Clarifai inputs
            meta_stream_url = inp.data.metadata['stream_url']
            input_id = inp.id

        bright_frame = get_bright_frame(meta_stream_url)  # Extract bright frame

        if bright_frame is not None:
            h, w, _ = bright_frame.shape  # Extract the original size
            _, buffer = cv2.imencode(".jpg", bright_frame)
            return BytesIO(buffer.tobytes()), (w, h), meta_stream_url, input_id
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return None, None, None, None

# # for debugging. skips the whole "grab inputs from clarifai" part
# def process_input(inp):
#   """Processes a single input by extracting a bright frame and returning relevant metadata."""
#   meta_stream_url = inp["stream_url"]  # Directly access the stream URL from the manual list
#   bright_frame = get_bright_frame(meta_stream_url)

#   if bright_frame is not None:
#     _, buffer = cv2.imencode(".jpg", bright_frame)
#     return BytesIO(buffer.tobytes()), meta_stream_url, "manual_test_id"  # Use placeholder ID
#   else:
#     return None, None, None

# Retrieve inputs based on current page
try:
    # Check if we're in manual input mode
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

    # Process the inputs
    thumbnails, thumbnail_sizes, video_urls, input_ids = [], [], [], []

    with st.spinner("Loading inputs..."):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_input, inputs))

        # Unpack results into separate lists
        for result in results:
            if result is not None and len(result) == 4:
                thumbnail, size, video_url, input_id = result
                if all([thumbnail, size, video_url, input_id]):
                    thumbnails.append(thumbnail)
                    thumbnail_sizes.append(size)
                    video_urls.append(video_url)
                    input_ids.append(input_id)

    # Only proceed if we have valid results
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

# page_of_inputs = [
#   {"stream_url": "rtsp://localhost:8554/mystream"},  # RTSP stream
#   {"stream_url": "http://80.61.63.103:81/cam_1.cgi"},  # HTTP/MJPEG stream
#   {"stream_url": "http://124.143.25.100/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000"},  # Another HTTP stream
#   {"stream_url": "https://videos.pexels.com/video-files/30514807/13074060_1920_1080_30fps.mp4"},  # MP4 video file
# ]

# Display thumbnails
thumbnails, thumbnail_sizes, video_urls, input_ids = [], [], [], []

with st.spinner("Loading inputs..."):
  with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_input, inputs))

  # Unpack results into separate lists
  for thumbnail, size, video_url, input_id in results:
    thumbnails.append(thumbnail)
    thumbnail_sizes.append(size)  # Store sizes in a list
    video_urls.append(video_url)
    input_ids.append(input_id)


# Placeholder transparent image (1x1 pixel)
empty_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgAB/SmMIOoAAAAASUVORK5CYII="

# Convert images to base64 URLs
image_urls = [to_base64_data_url(img) if img else empty_image for img in thumbnails]

# Ensure grid format: 5 images per row, pad last row if needed
cols_per_row = 5
if len(image_urls) % cols_per_row != 0:
  padding_needed = cols_per_row - (len(image_urls) % cols_per_row)
  image_urls.extend([empty_image] * padding_needed)

# Display all images in a **single** instance of `clickable_images()`
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

# Handle click event (ignore clicks on `empty_image` placeholders)
if clicked_index > -1 and clicked_index < len(video_urls) and image_urls[clicked_index] != empty_image:
    st.session_state.selected_video = video_urls[clicked_index]
    st.session_state.selected_input_id = input_ids[clicked_index]
    st.session_state.selected_thumbnail = thumbnails[clicked_index]
    st.session_state.selected_thumbnail_size = thumbnail_sizes[clicked_index]
    st.session_state.video_loaded = True
    st.switch_page("pages/2_Model_Selection_and_Inference.py")

# Pagination Controls
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

# Footer
footer(st)
