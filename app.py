# ============================================================
#  JABU FER System — Facial Expression Recognition
#  Student  : Charles Dan | 2203030073
# ============================================================

import streamlit as st
import numpy as np
import cv2
import os
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Page Config
st.set_page_config(page_title="JABU FER System", layout="wide")

# 2. Memory-Efficient Model Loader
MODEL_PATH = "model/jabu_model_float16.tflite"

@st.cache_resource
def load_interpreter():
    if not os.path.exists(MODEL_PATH):
        return None
    # 1. Initialize
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    
    # 2. Force single-threaded mode to prevent RAM spikes
    interpreter.set_num_threads(1) 
    
    # 3. Allocate
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Main Logic
def main():
    st.title("JABU FER Mobile Monitor")
    
    interp = load_interpreter()
    cascade = load_cascade()

    if interp is None:
        st.error(f"🚨 Model file not found at `{MODEL_PATH}`.")
        st.stop()

    tab1, tab2 = st.tabs(["📸 Snapshot", "🎥 Live Camera"])
    
    with tab1:
        st.write("Upload an image for emotion detection.")
        # ... (Keep your existing tab1/snapshot logic here)

    with tab2:
        st.write("Live detection mode.")
        # ... (Keep your existing tab2/live camera logic here)

if __name__ == "__main__":
    main()
