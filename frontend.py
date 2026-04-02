import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(layout="wide")  # make full page width

st.title("Masked Autoencoder Demo")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Send image to FastAPI server
    files = {"file": ("image.png", img_bytes, "image/png")}

    if st.button("Send to Model"):

        with st.spinner("Processing..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

        if response.status_code == 200:

            data = response.json()
            psnr_value = data.get("psnr")
            ssim_value = data.get("ssim")

            # Decode images from base64
            masked_img = Image.open(io.BytesIO(base64.b64decode(data["masked"])))
            generated_img = Image.open(io.BytesIO(base64.b64decode(data["generated"])))

            # Display side by side with bigger images
            col1, col2, col3 = st.columns(3, gap="large")

            col_width = 400  # pixels

            with col1:
                st.image(image, caption="Original", width=col_width)

            with col2:
                st.image(masked_img, caption="Masked", width=col_width)

            with col3:
                st.image(generated_img, caption="Generated", width=col_width)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if psnr_value is not None:
                    st.metric("PSNR (dB)", f"{psnr_value:.2f}")
                else:
                    st.info("PSNR unavailable")

            with metric_col2:
                if ssim_value is not None:
                    st.metric("SSIM", f"{ssim_value:.2f}")
                else:
                    st.info("SSIM unavailable")
        else:
            st.error("Server error")