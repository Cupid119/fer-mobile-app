# JABU FER System — Mobile Web App

**Student:** Charles Dan | 2203030073  
**Department:** Computer Science, Joseph Ayo Babalola University (JABU)  
**Project:** Facial Expression Recognition using MobileNetV3Large + Point Attention

---

## 🌐 Live Demo
> *(paste your Streamlit Cloud URL here after deployment)*

---

## 📁 Repository Structure

```
fer-mobile-app/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (apt)
├── model/
│   └── jabu_model_dynamic.tflite  # TFLite model (4 MB, ~14 ms/frame)
└── .streamlit/
    └── config.toml                 # UI theme
```

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set `app.py` as the main file
4. Click **Deploy**

---

## 🏗️ Architecture

- **Backbone:** MobileNetV3Large (ImageNet pretrained)
- **Attention:** Custom Point Attention (Channel + Spatial)
- **Preprocessing:** Global Contrast Normalisation (GCN) + Histogram Equalization
- **Input:** 224 × 224 × 3 (grayscale stacked)
- **Emotions:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Dataset:** FER2013 + RAF-DB

---

## 📊 Performance

| Metric            | Value         |
|-------------------|---------------|
| Test Accuracy     | ~73.7%        |
| TFLite Latency    | 14.1 ms/frame |
| TFLite FPS        | ~71 FPS       |
| Model Size        | 4.0 MB        |
