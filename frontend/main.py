import base64
import io

import httpx
import streamlit as st
from PIL import Image

from api_client import predict

st.set_page_config(page_title="Disaster Risk Monitoring", layout="wide")
st.title("Disaster Risk Monitoring")
st.caption("Upload a satellite image to estimate flooded areas.")

uploaded = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    data = uploaded.getvalue()
    original = Image.open(io.BytesIO(data))

    with st.spinner("Analyzing..."):
        try:
            result = predict(uploaded.name, data, uploaded.type)
        except httpx.ConnectError:
            st.error("Cannot reach the backend. Is it running?")
            st.stop()
        except httpx.HTTPStatusError as exc:
            st.error(exc.response.json().get("detail", "Request failed"))
            st.stop()

    mask = Image.open(io.BytesIO(base64.b64decode(result["mask_png_base64"])))

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(original)
    with right:
        st.subheader("Flood mask")
        st.image(mask)

    st.metric("Flooded area", f"{result['flooded_fraction']:.1%}")