import streamlit as st

def footer(st):
    """Display the footer with Clarifai branding."""
    footer_html = """
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: transparent; color: #808080; text-align: center; padding: 10px;">
        <p>
            Powered by <a style='display: inline; text-align: left;' href="https://www.clarifai.com" target="_blank">Clarifai</a>
            <img src="https://www.clarifai.com/hubfs/Clarifais%20Stand%20alone%20Icon-1.svg" alt="Clarifai Logo" height="25" style="vertical-align: middle; margin-left: 5px;">
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)  # Changed from st.write(footer, ...) to st.markdown(footer_html, ...)
