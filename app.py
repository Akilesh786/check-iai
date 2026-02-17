import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os

# ==========================================
# 🚩 ROBUST FILE DISCOVERY
# ==========================================
def find_model(name):
    """Searches the entire project directory for the model file."""
    for root, dirs, files in os.walk(os.getcwd()):
        if name in files:
            return os.path.join(root, name)
    return None

MODEL_FILENAME = "newly_trained.keras"
MODEL_PATH = find_model(MODEL_FILENAME)

# ==========================================
# 🧠 AI ENGINE
# ==========================================
class InstrunetCore:
    def __init__(self, path):
        self.model = self._load_model(path)

    @st.cache_resource
    def _load_model(_self, path):
        if not path:
            st.error(f"❌ '{MODEL_FILENAME}' not found in the repository. Please check your GitHub file list.")
            return None
        try:
            # compile=False is vital for loading across different Keras versions
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            st.error(f"❌ Model found at {path} but failed to load. Error: {e}")
            st.info("Tip: If the error mentions 'zip file', your upload might be corrupted. Try re-uploading to GitHub.")
            return None

    def process_signal(self, path):
        y, sr = librosa.load(path, sr=22050, duration=15)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=30
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
        return {"label": np.argmax(avg_preds), "dist": avg_preds, "y": y, "peaks": times}

# ==========================================
# 🎨 UI STYLING
# ==========================================
st.set_page_config(page_title="Instrunet AI", layout="wide")
st.markdown("""<style>
    .stApp { background: #0b0f19; color: white; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; }
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #38bdf8; background: rgba(56, 189, 248, 0.1); }
</style>""", unsafe_allow_html=True)

# ==========================================
# 🚀 MAIN APP
# ==========================================
def main():
    engine = InstrunetCore(MODEL_PATH)
    
    with st.sidebar:
        st.title("🎼 INSTRUNET AI")
        nav = st.radio("MENU", ["Studio", "Results"])
        st.markdown("---")
        if MODEL_PATH:
            st.success(f"✅ Model found: `{MODEL_PATH.split('/')[-1]}`")
        else:
            st.error("❌ Model Missing")

    if nav == "Studio":
        st.title("🎙️ Analysis Studio")
        tab1, tab2 = st.tabs(["📁 Upload", "🎤 Record"])
        with tab1: file = st.file_uploader("Audio", type=["wav", "mp3"])
        with tab2: rec = st.audio_input("Record")
        
        src = file if file else rec
        if src and st.button("RUN SCAN"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(src.getvalue()); p = tmp.name
            st.session_state.results = engine.process_signal(p)
            st.success("Analysis complete! Go to Results.")

    if nav == "Results" and 'results' in st.session_state:
        res = st.session_state.results
        st.header(f"Detected Instrument Index: {res['label']}")
        st.plotly_chart(px.bar(y=res['dist'], template="plotly_dark"))

if __name__ == "__main__":
    main()
