import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# Page config with dark theme
st.set_page_config(
    page_title="Leukemia Detection AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0f 50%, #0a0a1a 100%);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(255, 107, 107, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(255, 107, 107, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(255, 107, 107, 0.6)); }
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.2rem;
        color: #b8b8d1;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Card containers */
    .custom-card {
        background: rgba(20, 20, 30, 0.8);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 107, 107, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Upload section */
    .uploadedFile {
        border: 2px dashed rgba(255, 107, 107, 0.5) !important;
        background: rgba(20, 20, 30, 0.6) !important;
        border-radius: 15px !important;
    }
    
    /* Text colors */
    .stMarkdown, p, label {
        color: #e0e0e8 !important;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(196, 69, 105, 0.1) 100%);
        border: 2px solid rgba(255, 107, 107, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Prediction class */
    .pred-class {
        font-size: 2rem;
        font-weight: 700;
        color: #ff6b6b;
        text-align: center;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    /* Confidence */
    .confidence {
        font-size: 1.5rem;
        color: #4ecdc4;
        text-align: center;
        font-weight: 600;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b6b 0%, #c44569 100%);
    }
    
    /* Probability bars */
    .prob-bar {
        background: rgba(255, 107, 107, 0.2);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    
    .prob-label {
        color: #e0e0e8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prob-value {
        color: #4ecdc4;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
        padding: 1rem;
        border-radius: 8px;
        color: #e0e0e8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🩸 LEUKEMIA DETECTION AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ADVANCED BLOOD CELL CLASSIFICATION SYSTEM</p>', unsafe_allow_html=True)

# Info section
st.markdown("""
<div class="info-box">
    <strong>ℹ️ About This System</strong><br>
    This AI-powered system analyzes microscopic blood cell images to detect potential leukemia markers. 
    Upload a high-quality microscopic image for accurate classification.
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("leukemia_cnn_model.keras")

model = load_trained_model()

# Class names
class_names = ["Basophil", "Erythroblast", "Monocyte", "Myeloblast", "Segmented Neutrophil"]

# File uploader
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    st.markdown("<p style='text-align: center; color: #b8b8d1; font-size: 0.9rem;'>📤 Drop your blood cell image here</p>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Create columns for layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🔬 Uploaded Image")
        # Display uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_column_width=True)
    
    with col_right:
        st.markdown("### 🧬 Analysis in Progress...")
        
        # Simulate analysis with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("Preprocessing image...")
            elif i < 60:
                status_text.text("Running neural network...")
            else:
                status_text.text("Analyzing results...")
            time.sleep(0.01)
        
        status_text.text("✓ Analysis Complete!")
        
        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        pred_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        # Display results
        st.markdown("### 🎯 Diagnosis Results")
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="pred-class">{pred_class}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="confidence">Confidence: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full width probability section
    st.markdown("---")
    st.markdown("### 📊 Detailed Probability Analysis")
    
    # Create probability bars
    for cls, prob in zip(class_names, predictions[0]):
        col_name, col_bar, col_value = st.columns([2, 5, 1])
        
        with col_name:
            st.markdown(f"**{cls}**")
        
        with col_bar:
            st.progress(float(prob))
        
        with col_value:
            st.markdown(f"**{prob*100:.1f}%**")

else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; color: #7c7c94;'>
        <h2 style='color: #7c7c94;'>👆 Upload an image to begin analysis</h2>
        <p>Supported formats: JPG, PNG, JPEG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7c7c94; padding: 2rem;'>
    <p>⚕️ For research and educational purposes only. Not a substitute for professional medical diagnosis.</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>Powered by Deep Learning & TensorFlow</p>
</div>
""", unsafe_allow_html=True)