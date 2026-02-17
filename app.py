import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time
from datetime import datetime

import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Join it with the filename
MODEL_PATH = os.path.join(BASE_DIR, "Newly_trained.keras")

# Load it
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 🚩 SYSTEM CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Instrunet AI v2",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPDATED: Pointing to your new .keras model file
MODEL_PATH = os.path.join(BASE_DIR, "Newly_trained.keras")

# UPDATED: Full 11 instrument list from IRMAS
INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'tra', 'voi']

FULL_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'tra': 'Trombone', 'voi': 'Human Voice'
}

# ==========================================
# 🤖 CHATBOT LOGIC ENGINE
# ==========================================
def get_bot_response(user_input, last_prediction=None):
    user_input = user_input.lower()
    
    if "backend" in user_input or "pipeline" in user_input:
        return "Our v2 pipeline: 1. Audio Upload -> 2. Mono Normalization -> 3. Mel Spectrogram (130, 40) -> 4. CNN Inference -> 5. Multi-label Prediction."
    
    elif "accuracy" in user_input:
        return "The current InstruNet v2 model achieved ~76-80% accuracy during training with the improved CNN architecture."
    
    elif "prediction" in user_input:
        if last_prediction:
            return f"The last detected instrument was: {last_prediction}."
        return "Please analyze an audio file in the Studio first."
    
    else:
        return "I am the v2 guide. I can explain the new CNN layers, the 80% accuracy milestone, or the Mel Spectrogram shapes."

# ==========================================
# 🎨 ANIMATED CSS UI ENGINE
# ==========================================
def apply_ultra_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        .stApp { background: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stMarkdown, .stButton, .stPlotlyChart { animation: fadeInUp 0.6s ease-out; }

        [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
        .nav-header { color: #38bdf8; font-size: 28px; font-weight: 900; padding: 30px 0; text-align: center; border-bottom: 2px solid #1e293b; }

        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 28px; padding: 60px; text-align: center; margin: 40px 0;
            backdrop-filter: blur(25px); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }
        .hero-section h1 { font-size: 64px !important; font-weight: 900 !important; color: #ffffff !important; }

        .metric-card {
            background: rgba(30, 41, 59, 0.6); border-radius: 20px; padding: 30px;
            border: 1px solid #334155; text-align: center; margin-bottom: 20px;
        }

        .stButton>button {
            background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
            border: none; border-radius: 16px; color: white; height: 4.5em; font-weight: 800;
        }
        .ai-msg { background: #1e293b; border-radius: 18px; padding: 15px; margin: 10px 0; border-left: 5px solid #38bdf8; font-size: 0.9em; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 AI ANALYTICS ENGINE
# ==========================================
class InstrunetCoreV3:
    def __init__(self, path):
        self.model = self._load_model(path)

    @st.cache_resource
    def _load_model(_self, path):
        try:
            if os.path.exists(path):
                # .keras models are loaded identically to .h5 via this API
                return tf.keras.models.load_model(path, compile=False)
            else:
                st.error(f"Model file not found at {path}. Please upload 'Newly_trained.keras' to your project folder.")
                return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def process_signal(self, path):
        # Match the 3-second training duration
        y, sr = librosa.load(path, sr=22050, duration=3.0)
        
        # Match training feature extraction
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resizing to (130, 40) as per our successful training run
        features = log_spec[:40, :130].T 
        
        # Add padding if the clip is shorter than 3s
        if features.shape[0] < 130:
            pad_width = 130 - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')

        # Reshape for CNN input: (batch, time, freq, channel)
        input_data = features[np.newaxis, ..., np.newaxis]
        
        preds = self.model.predict(input_data, verbose=0)[0]
        top_idx = np.argmax(preds)
        
        return {
            "meta": {"id": datetime.now().strftime("%H:%M:%S")},
            "result": {"label": FULL_NAMES[INSTRUMENTS[top_idx]], "conf": preds[top_idx]},
            "data": {"dist": {FULL_NAMES[INSTRUMENTS[i]]: float(preds[i]) for i in range(len(INSTRUMENTS))}},
            "signal": {"y": y, "sr": sr, "spec": log_spec}
        }

# ==========================================
# 🖥️ ROUTING FUNCTIONS
# ==========================================
def render_home():
    st.markdown("<div class='hero-section'><h1>INSTRUNET v2</h1><p>Enhanced CNN Instrument Classifier</p></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='metric-card' style='max-width: 1200px; margin: 0 auto;'>
            <h3>New v2 Architecture</h3>
            <p style='font-size:1.1em; color:#cbd5e1; padding: 10px 40px;'>
                Featuring <b>Spatial Dropout</b> and <b>Batch Normalization</b>. 
                This version is trained on the full IRMAS dataset with improved generalization against overfitting.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("OPEN ANALYSIS STUDIO 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

def render_studio(engine):
    st.title("🎙️ Analysis Studio v2")
    if engine.model is None:
        st.warning("⚠️ Model not loaded. Ensure 'Newly_trained.keras' is in the project directory.")
        return

    file = st.file_uploader("Select audio source", type=["wav", "mp3"])
    if file:
        st.audio(file)
        if st.button("EXECUTE NEURAL SCAN"):
            with st.status("Analyzing Audio Timbre..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.getvalue()); p = tmp.name
                res = engine.process_signal(p)
                st.session_state.current = res
                st.session_state.history.append(res)
                st.session_state.page = "Instrument Distribution"
                st.rerun()

def render_distribution():
    if not st.session_state.current:
        st.info("No data found. Please run a scan first.")
        return
        
    res = st.session_state.current
    st.title("📊 Analysis Results")
    st.markdown(f"<div class='hero-section' style='padding:30px;'><h2>{res['result']['label'].upper()}</h2><h4>Confidence: {res['result']['conf']*100:.2f}%</h4></div>", unsafe_allow_html=True)
    
    df = pd.DataFrame(res['data']['dist'].items(), columns=['Inst', 'Val'])
    st.plotly_chart(px.bar(df, x='Inst', y='Val', color='Val', template="plotly_dark", title="Instrument Probability Map"), use_container_width=True)

# ==========================================
# 🚀 MAIN APPLICATION LOOP
# ==========================================
def main():
    apply_ultra_styles()
    engine = InstrunetCoreV3(MODEL_PATH)
    
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat" not in st.session_state: st.session_state.chat = []

    with st.sidebar:
        st.markdown("<div class='nav-header'>🎼 INSTRUNET v2</div>", unsafe_allow_html=True)
        nav = st.radio("NAVIGATE", ["Home", "Upload & Analyze", "Instrument Distribution"])
        if nav != st.session_state.page: st.session_state.page = nav; st.rerun()
        
        st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
        st.subheader("🤖 v2 Technical Guide")
        
        for c in st.session_state.chat[-4:]:
            role_label = "👤 You" if c["role"] == "user" else "🤖 Bot"
            st.markdown(f"<div class='ai-msg'><b>{role_label}:</b><br>{c['content']}</div>", unsafe_allow_html=True)
        
        if q := st.chat_input("Ask about the 80% accuracy..."):
            last_label = st.session_state.current['result']['label'] if st.session_state.current else None
            response = get_bot_response(q, last_label)
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    if st.session_state.page == "Home": render_home()
    elif st.session_state.page == "Upload & Analyze": render_studio(engine)
    elif st.session_state.page == "Instrument Distribution": render_distribution()

if __name__ == "__main__":
    main()
