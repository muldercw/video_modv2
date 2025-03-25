import ssl
import socket
from urllib.parse import urlparse

# Configuration variables (edit these as needed)
URL = "web-staging.clarifai.com"  # The server URL
PORT = 443                    # The port to connect to
OUTPUT_FILE = "staging_server_cert.pem"  # The file to save the certificate to

def get_server_certificate(url, port, output_file):
    """
    Retrieve the SSL/TLS certificate from a server and save it as a PEM file.
    
    Args:
        url (str): The URL of the server (e.g., 'web-dev.clarifai.com').
        port (int): The port to connect to.
        output_file (str): The file to save the certificate to.
    """
    # Remove any protocol prefix if present
    url = url.replace("https://", "").replace("http://", "")
    
    # Remove any path or query parameters
    url = url.split('/')[0]
    
    # Use the cleaned URL as hostname
    hostname = url

    if not hostname:
        raise ValueError("Invalid URL or hostname not provided.")

    print(f"Connecting to {hostname}:{port} to retrieve certificate...")

    # Create a socket and wrap it with SSL context
    context = ssl._create_unverified_context()#    create_default_context()
    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            # Get the certificate in DER format
            cert_der = ssock.getpeercert(binary_form=True)
            # Convert DER to PEM format
            cert_pem = ssl.DER_cert_to_PEM_cert(cert_der)

    # Save the certificate to a file
    with open(output_file, 'w') as f:
        f.write(cert_pem)
    print("Certificate retrieved successfully.")
    print(cert_pem)
    print(f"Certificate saved to {output_file}")

def main():
    try:
        get_server_certificate(URL, PORT, OUTPUT_FILE)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
