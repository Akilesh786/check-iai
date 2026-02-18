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

# ==========================================
# 🚩 SYSTEM CORE & ROBUST DISCOVERY
# ==========================================
st.set_page_config(
    page_title="Instrunet AI",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def find_model(name):
    """Recursively searches the project for the model file."""
    for root, dirs, files in os.walk(os.getcwd()):
        if name in files:
            return os.path.join(root, name)
    return None

# Looking for YOUR specific model name
MODEL_FILENAME = "newly_trained.keras"
MODEL_PATH = find_model(MODEL_FILENAME)

INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
FULL_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'vio': 'Violin', 'voi': 'Human Voice'
}

# ==========================================
# 🤖 ENHANCED AI CHATBOT LOGIC
# ==========================================
def get_bot_response(user_input, last_result=None):
    user_input = user_input.lower()
    if any(word in user_input for word in ["backend", "pipeline", "process"]):
        return "<b>Backend Pipeline:</b><br>1. <b>Upload:</b> Audio is normalized.<br>2. <b>Landmarks:</b> We find 'onset peaks'.<br>3. <b>Extraction:</b> MFCCs generated.<br>4. <b>CNN:</b> Model predicts based on spectral textures."
    elif any(word in user_input for word in ["waveform", "peaks", "landmark"]):
        if last_result:
            count = len(last_result['signal']['landmarks'])
            return f"The red dashed lines are the <b>{count} temporal landmarks</b> I detected. These are 'attack' points where timbre is clearest."
        return "The waveform shows signal amplitude. Upload a file to see detected peaks!"
    elif "accuracy" in user_input:
        return "The model achieves <b>85–92% validation accuracy</b> depending on instrument class."
    return "I am the Instrunet Technical Guide. Ask about Waveforms, CNN Model, or Spectrograms!"

# ==========================================
# 🎨 ANIMATED CSS UI ENGINE
# ==========================================
def apply_ultra_styles():
    st.markdown("""
        <style>
        .stApp { background: #0b0f19; color: #e2e8f0; }
        [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
        .hero-section {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px; padding: 50px; text-align: center; margin-bottom: 40px;
        }
        .ai-msg { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 20px; border-left: 4px solid #38bdf8; }
        .stButton>button { background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%); color: white; border-radius: 12px; font-weight: 700; }
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
        if not path: return None
        try:
            return tf.keras.models.load_model(path, compile=False)
        except: return None

    def process_signal(self, path):
        y, sr = librosa.load(path, sr=22050, duration=15)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=30)
        times = librosa.frames_to_time(peaks, sr=sr)
        if len(times) == 0: times = [0.0]
        
        features = []
        for t in times[:10]:
            start = int(max(0, (t - 0.5) * sr))
            chunk = y[start : start + int(3*sr)]
            if len(chunk) < 3*sr: chunk = np.pad(chunk, (0, int(3*sr)-len(chunk)))
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40).T
            mfcc = mfcc[:130] if mfcc.shape[0] >= 130 else np.pad(mfcc, ((0, 130-mfcc.shape[0]), (0, 0)))
            features.append(self.model.predict(mfcc.reshape(1, 130, 40, 1), verbose=0)[0])

        avg_preds = np.mean(features, axis=0)
        top_idx = np.argmax(avg_preds)
        return {
            "meta": {"id": datetime.now().strftime("%H:%M:%S")},
            "result": {"label": FULL_NAMES[INSTRUMENTS[top_idx]], "conf": avg_preds[top_idx]},
            "data": {"dist": {FULL_NAMES[INSTRUMENTS[i]]: float(avg_preds[i]) for i in range(len(INSTRUMENTS))}},
            "signal": {"y": y, "sr": sr, "landmarks": times, "spec": librosa.feature.melspectrogram(y=y, sr=sr)}
        }

# ==========================================
# 🖥️ PAGE RENDERING FUNCTIONS
# ==========================================
def render_home():
    st.markdown("<div class='hero-section'><h1>INSTRUNET AI</h1><p>Neural Network Instrumentation Classifier</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='ai-msg'>Utilizing Deep CNNs for High-Resolution Spectral Mapping.</div>", unsafe_allow_html=True)
    if st.button("OPEN ANALYSIS STUDIO 🚀", use_container_width=True):
        st.session_state.page = "Upload & Analyze"
        st.rerun()

def render_studio(engine):
    st.title("🎙️ Analysis Studio")
    file = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])
    if file:
        st.audio(file)
        if st.button("EXECUTE NEURAL SCAN"):
            if engine.model is None:
                st.error("Model file missing from repo!")
            else:
                with st.spinner("Analyzing spectral patterns..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(file.getvalue()); p = tmp.name
                    res = engine.process_signal(p)
                    st.session_state.current = res
                    st.session_state.history.append(res)
                    st.session_state.page = "Instrument Distribution"
                    st.rerun()

def render_distribution():
    res = st.session_state.current
    st.title("📊 Analysis Results")
    st.markdown(f"<div class='hero-section'><h2>{res['result']['label'].upper()}</h2><h4>Confidence: {res['result']['conf']*100:.1f}%</h4></div>", unsafe_allow_html=True)
    fig = px.bar(x=list(res['data']['dist'].keys()), y=list(res['data']['dist'].values()), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def render_technical():
    res = st.session_state.current
    st.title("🔬 Deep Technical Analysis")
    st.subheader("1. Pulse Landmark & Temporal Peaks")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=res['signal']['y'][::100], line=dict(color='#38bdf8')))
    for l in res['signal']['landmarks']:
        fig.add_vline(x=l*res['signal']['sr']/100, line_dash="dash", line_color="#ef4444")
    st.plotly_chart(fig, use_container_width=True)

def render_history():
    st.title("📜 Neural Audit Logs")
    for item in reversed(st.session_state.history):
        st.markdown(f"<div class='ai-msg'><b>[{item['meta']['id']}]</b> {item['result']['label']} ({item['result']['conf']*100:.1f}%)</div>", unsafe_allow_html=True)

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    apply_ultra_styles()
    engine = InstrunetCoreV3(MODEL_PATH)
    
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "current" not in st.session_state: st.session_state.current = None
    if "history" not in st.session_state: st.session_state.history = []
    if "chat" not in st.session_state: st.session_state.chat = []

    with st.sidebar:
        st.title("🎼 INSTRUNET AI")
        nav = st.radio("SYSTEM", ["Home", "Upload & Analyze", "Instrument Distribution", "Deep Technical Analysis", "Audit Logs"])
        if nav != st.session_state.page: 
            st.session_state.page = nav
            st.rerun()
        
        st.markdown("---")
        st.subheader("🤖 AI Guide")
        for c in st.session_state.chat[-2:]:
            st.markdown(f"<div class='ai-msg'><b>{c['role'].upper()}:</b> {c['content']}</div>", unsafe_allow_html=True)
        if q := st.chat_input("Ask about CNN..."):
            resp = get_bot_response(q, st.session_state.current)
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": resp})
            st.rerun()

    if st.session_state.page == "Home": render_home()
    elif st.session_state.page == "Upload & Analyze": render_studio(engine)
    elif st.session_state.page == "Instrument Distribution": render_distribution()
    elif st.session_state.page == "Deep Technical Analysis": render_technical()
    elif st.session_state.page == "Audit Logs": render_history()

if __name__ == "__main__":
    main()
