import os
import tempfile
import datetime
import base64
import json
from datetime import timezone
import streamlit as st
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
import ssl
import socket
import traceback
from urllib.parse import urlparse
from get_cert import get_server_certificate

def form_has_values():
    return bool(st.session_state.get("pat_input") and 
                st.session_state.get("user_id_input") and 
                st.session_state.get("app_id_input"))

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def export_form_credentials():
    export_data = {
        "pat": st.session_state.get("pat_input", ""),
        "user_id": st.session_state.get("user_id_input", ""),
        "app_id": st.session_state.get("app_id_input", ""),
        "base_url": st.session_state.get("base_url_input", ""),
        "cert_server_url": st.session_state.get("cert_server_url", ""),
        "models": st.session_state.get("models_input", "")
    }
    cert_path = st.session_state.get("clarifai_root_certificates_path")
    if cert_path and os.path.exists(cert_path):
        try:
            with open(cert_path, 'r') as cert_file:
                export_data["certificate"] = cert_file.read()
        except Exception as e:
            st.error(f"Failed to read certificate: {str(e)}")
    
    return json.dumps(export_data, indent=2)


def import_credentials(json_file):
    try:
        data = json.loads(json_file.read())
        st.session_state.form_pat = data.get("pat", "")
        st.session_state.form_user_id = data.get("user_id", "")
        st.session_state.form_app_id = data.get("app_id", "")
        st.session_state.form_base_url = data.get("base_url", "")
        st.session_state.form_cert_server_url = data.get("cert_server_url", "")
        st.session_state.form_models = data.get("models", "")
        return True
    except Exception as e:
        st.error(f"Error importing credentials: {str(e)}")
        return False

def validate_certificate(cert_file):
    if cert_file is None:
        return False   
    try:
        cert_content = cert_file.read()
        cert_file.seek(0)
        try:
            cert = x509.load_pem_x509_certificate(cert_content, default_backend())
            now = datetime.datetime.now(timezone.utc)
            if cert.not_valid_after_utc < now:
                st.error("Certificate has expired")
                return False
            if cert.not_valid_before_utc > now:
                st.error("Certificate is not yet valid")
                return False
            st.info(f"""Certificate details:
            - Subject: {cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value}
            - Issuer: {cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value}
            - Valid until: {cert.not_valid_after_utc}
            """)
            return True
        except Exception as e:
            st.error(f"Invalid certificate format: {str(e)}")
            return False
    except Exception as e:
        st.error(f"Error reading certificate: {str(e)}")
        return False

def get_cert_from_base_url(base_url):
    try:
        parsed_url = urlparse(base_url)
        hostname = parsed_url.hostname or base_url
        port = parsed_url.port or 443
        cert_filename = os.path.join(st.session_state.temp_cert_dir, "root_cert.pem")
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Get the certificate in DER format
                cert_der = ssock.getpeercert(binary_form=True)
                # Convert DER to PEM format
                cert_pem = ssl.DER_cert_to_PEM_cert(cert_der)
        with open(cert_filename, 'w') as f:
            f.write(cert_pem)
        st.session_state["clarifai_root_certificates_path"] = cert_filename
        return True
    except Exception as e:
        st.error(f"Failed to get certificate: {str(e)}")
        return False

global_css = """
  <style>
    div.stButton > button {
      background-color: rgb(0, 109, 255) !important;
      color: white !important;
      border-radius: 8px !important;
      border: none !important;
      padding: 8px 16px !important;
      font-size: 16px !important;
      cursor: pointer !important;
      transition: background-color 0.3s ease-in-out;
    }
    div.stButton > button:hover {
      background-color: rgb(0, 89, 209) !important;
    }
    div.stButton > button:disabled,
    div.stButton > button[disabled] {
      background-color: rgba(0, 109, 255, 0.3) !important;
      color: rgba(255, 255, 255, 0.6) !important;
      cursor: not-allowed !important;
      opacity: 0.7 !important;
    }
  </style>
"""
st.session_state["global_css"] = global_css
st.markdown(global_css, unsafe_allow_html=True)
if "form_pat" not in st.session_state:
    st.session_state.form_pat = ""
if "form_user_id" not in st.session_state:
    st.session_state.form_user_id = ""
if "form_app_id" not in st.session_state:
    st.session_state.form_app_id = ""
if "form_base_url" not in st.session_state:
    st.session_state.form_base_url = ""
if "form_models" not in st.session_state:
    st.session_state.form_models = ""
if "models_input" not in st.session_state:
    st.session_state.models_input = ""


def credentials_exist():
    return all(key in st.session_state for key in ["clarifai_pat", "clarifai_user_id", "clarifai_app_id"])

if 'temp_cert_dir' not in st.session_state:
    st.session_state.temp_cert_dir = tempfile.mkdtemp()

if credentials_exist():
    st.switch_page("pages/1_Video_Selection.py")
else:
    logo_path = os.path.join("assets", "clarifai_logo.png")
    try:
        logo_base64 = get_base64_encoded_image(logo_path)
        st.markdown(f"""
            <div style="text-align: center; padding: 0px; margin-top: -60px; margin-bottom: -30px;">
                <img src="data:image/png;base64,{logo_base64}" 
                     alt="Clarifai Logo" 
                     style="width: 25%; max-width: 25%;">
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Clarifai logo not found in assets directory")
    st.markdown("<h1 style='text-align: center;'>JSOC Streaming Module</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Enter your credentials to proceed.</h4>", unsafe_allow_html=True)

    st.markdown("""
        <style>
        .uploadedFile {
            width: 100% !important;
        }
        .stFileUploader > div {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    config_file = st.file_uploader("Import Configuration", type=['json'])
    if config_file is not None:
        if import_credentials(config_file):
            st.success("Configuration imported successfully")

    with st.form("clarifai_credentials_form"):
        clarifai_pat = st.text_input(
            "Personal Access Token (PAT)", 
            type="password",
            key="pat_input",
            value=st.session_state.get("form_pat", "")
        )
        
        clarifai_user_id = st.text_input(
            "User ID",
            key="user_id_input",
            value=st.session_state.get("form_user_id", "")
        )
        
        clarifai_app_id = st.text_input(
            "App ID",
            key="app_id_input",
            value=st.session_state.get("form_app_id", "")
        )
        
        st.text_input(
            "Base URL (Optional)",
            key="base_url_input",
            value=st.session_state.get("form_base_url", ""),
            help="Custom API endpoint URL (e.g., https://api-dev.clarifai.com)"
        )

        cert_server_url = st.text_input(
            "Certificate Server URL (Required if using custom Base URL)",
            key="cert_server_url",
            value=st.session_state.get("form_cert_server_url", ""),
            help="Server URL for certificate (e.g., web-staging.clarifai.com)"
        )

        models_input = st.text_area(
            "Models (one per line, format: Name|URL)",
            key="models_input",
            value=st.session_state.get("form_models", ""),
            help="Enter each model on a new line in the format: ModelName|ModelURL"
        )

        download_config = st.checkbox("Download configuration after submission", value=False)
        submitted = st.form_submit_button("Validate")

    def update_pat():
        st.session_state.form_pat = st.session_state.pat_input

    def update_user_id():
        st.session_state.form_user_id = st.session_state.user_id_input

    def update_app_id():
        st.session_state.form_app_id = st.session_state.app_id_input

    def update_base_url():
        st.session_state.form_base_url = st.session_state.base_url_input

    def update_models():
        st.session_state.form_models = st.session_state.models_input

    def trigger_download():
        json_data = export_form_credentials()
        st.download_button(
            label="Download Configuration",
            data=json_data,
            file_name="clarifai_config.json",
            mime="application/json",
            key="auto_download"
        )
        st.markdown(
            """
            <script>
                document.querySelector('button[data-testid="stDownloadButton"]').click();
            </script>
            """,
            unsafe_allow_html=True
        )

    if submitted:
        if not all([st.session_state.pat_input, st.session_state.user_id_input, st.session_state.app_id_input]):
            st.error("Please fill in all required fields.")
            st.stop()
        models = []
        for line in st.session_state.models_input.strip().split('\n'):
            if line.strip():
                try:
                    name, url = line.split('|')
                    models.append({
                        "Name": name.strip(),
                        "URL": url.strip()
                    })
                except ValueError:
                    st.error(f"Invalid model format: {line}. Please use Name|URL format.")
                    st.stop()

        if not models:
            st.error("Please add at least one model.")
            st.stop()
        if st.session_state.base_url_input and not st.session_state.base_url_input.startswith("https://api.clarifai.com"):
            if not cert_server_url:
                st.error("Certificate Server URL is required when using a custom base URL")
                st.stop()
            try:
                cert_filename = os.path.join(st.session_state.temp_cert_dir, "root_cert.pem")
                get_server_certificate(cert_server_url, 443, cert_filename)
                st.session_state["clarifai_root_certificates_path"] = cert_filename
                st.success("Certificate retrieved successfully")
            except Exception as e:
                st.error(f"Failed to get certificate: {str(e)}")
                st.stop()
        else:
            st.session_state["clarifai_root_certificates_path"] = None
        st.session_state["clarifai_pat"] = st.session_state.pat_input
        st.session_state["clarifai_user_id"] = st.session_state.user_id_input
        st.session_state["clarifai_app_id"] = st.session_state.app_id_input
        st.session_state["clarifai_base_url"] = st.session_state.base_url_input
        st.session_state["models"] = models
        st.session_state["validation_complete"] = True
        st.success("Configuration validated successfully!")

    if st.session_state.get("validation_complete"):
        json_data = export_form_credentials()
        if download_config:
            if st.download_button(
                label="📥 Download and Continue",
                data=json_data,
                file_name="clarifai_config.json",
                mime="application/json",
                use_container_width=True
            ):
                st.switch_page("pages/1_Video_Selection.py")
        else:
            if st.button("Continue", use_container_width=True):
                st.switch_page("pages/1_Video_Selection.py")
    else:
        st.error("Please fill in all fields including at least one model.")
