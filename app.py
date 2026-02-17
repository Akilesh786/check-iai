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
from datetime import datetime

# ==========================================
# 🚩 SYSTEM CORE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Instrunet AI V2",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust Pathing: Check current directory and mount directory
MODEL_FILENAME = "Newly_trained.keras"
possible_paths = [
    os.path.join(os.getcwd(), MODEL_FILENAME),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME),
    f"/mount/src/check-iai/{MODEL_FILENAME}"
]

INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
FULL_NAMES = {
    'cel': 'Cello', 'cla': 'Clarinet', 'flu': 'Flute', 'gac': 'Acoustic Guitar',
    'gel': 'Electric Guitar', 'org': 'Organ', 'pia': 'Piano', 'sax': 'Saxophone',
    'tru': 'Trumpet', 'vio': 'Violin', 'voi': 'Human Voice'
}

# ==========================================
# 🧠 AI ANALYTICS ENGINE
# ==========================================
class InstrunetCore:
    def __init__(self, paths):
        self.model = self._load_model(paths)

    @st.cache_resource
    def _load_model(_self, paths):
        for path in paths:
            if os.path.exists(path):
                try:
                    # compile=False prevents version mismatch errors
                    return tf.keras.models.load_model(path, compile=False)
                except Exception as e:
                    st.error(f"Attempted {path} but failed: {e}")
        return None

    def process_signal(self, path):
        y, sr = librosa.load(path, sr=22050, duration=15)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # FIXED: Explicit keyword arguments for compatibility with new Librosa
        peaks = librosa.util.peak_pick(
            onset_env, 
            pre_max=7, 
            post_max=7, 
            pre_avg=7, 
            post_avg=7, 
            delta=0.5, 
            wait=30
        )
        
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
            "result": {"label": FULL_NAMES[INSTRUMENTS[top_idx]], "conf": avg_preds[top_idx]},
            "data": {"dist": {FULL_NAMES[INSTRUMENTS[i]]: float(avg_preds[i]) for i in range(len(INSTRUMENTS))}},
            "signal": {"y": y, "sr": sr, "landmarks": times, "spec": librosa.feature.melspectrogram(y=y, sr=sr)}
        }

# ==========================================
# 🎨 UI & STYLING
# ==========================================
def apply_custom_styles():
    st.markdown("""
        <style>
        .stApp { background: #0b0f19; color: #e2e8f0; }
        [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
        div[role="radiogroup"] > label { padding: 15px 0px !important; font-size: 1.1rem !important; }
        .hero-section { background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%); border-radius: 24px; padding: 40px; text-align: center; margin-bottom: 30px; }
        .metric-card { background: rgba(30, 41, 59, 0.4); border-radius: 16px; padding: 30px; border: 1px solid #334155; text-align: center; margin-bottom: 40px; }
        .ai-msg { background: #1e293b; border-radius: 12px; padding: 15px; margin-bottom: 20px; border-left: 4px solid #38bdf8; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🚀 MAIN APP
# ==========================================
def main():
    apply_custom_styles()
    engine = InstrunetCore(possible_paths)
    
    if "current" not in st.session_state: st.session_state.current = None

    with st.sidebar:
        st.title("🎼 INSTRUNET AI")
        nav = st.radio("NAVIGATE", ["Home", "Upload & Analyze", "Instrument Distribution", "Deep Analysis"])

    if nav == "Home":
        st.markdown("<div class='hero-section'><h1>INSTRUNET V2</h1><p>Enhanced CNN Instrument Classifier</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-card'><h3>System Architecture</h3><p>Using CNN for Spectral Mapping and Peak Pick Landmarks.</p></div>", unsafe_allow_html=True)

    elif nav == "Upload & Analyze":
        st.title("🎙️ Analysis Studio")
        tab1, tab2 = st.tabs(["📁 File Upload", "🎤 Live Record"])
        with tab1: u_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
        with tab2: r_file = st.audio_input("Record Instrument")
        
        source = u_file if u_file else r_file
        if source:
            st.audio(source)
            if st.button("RUN NEURAL SCAN"):
                if engine.model is None:
                    st.error(f"Critical Error: {MODEL_FILENAME} not found in project paths.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(source.getvalue()); p = tmp.name
                    res = engine.process_signal(p)
                    st.session_state.current = res
                    st.success("Analysis complete! Switch to Distribution tab.")

    elif nav == "Instrument Distribution":
        if st.session_state.current:
            res = st.session_state.current
            st.header(f"Result: {res['result']['label']}")
            df = pd.DataFrame(res['data']['dist'].items(), columns=['Inst', 'Val'])
            st.plotly_chart(px.bar(df, x='Inst', y='Val', template="plotly_dark"), use_container_width=True)
        else:
            st.warning("Analyze audio in the Studio first.")

    elif nav == "Deep Analysis":
        if st.session_state.current:
            res = st.session_state.current
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res['signal']['y'][::100], name="Waveform", line=dict(color="#38bdf8")))
            for l in res['signal']['landmarks']:
                fig.add_vline(x=l*220.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
