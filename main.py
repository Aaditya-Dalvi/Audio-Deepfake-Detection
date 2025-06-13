import streamlit as st
import torch
import torchaudio
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from torch import nn
import requests
import tempfile
from streamlit_lottie import st_lottie
from torch.serialization import StorageType, normalize_storage_type, _get_restore_location
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Audio Deepfake Detection",
    page_icon="🔊",
    layout="centered"
)



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Why are model classes defined again in the app if the model is already trained?

You're not retraining the models in this Streamlit app. You're:
    Re-defining the architecture of each model (just the class structure).
    Loading the pre-trained weights from the saved .pth file.
    Using the loaded models only for inference (prediction).

    
Why It’s Required?

When you save a model with: torch.save(model.state_dict(), "path.pth")
You are saving just the weights, not the actual architecture or class definition.
To use it again:
    You need to redefine the same architecture in code.
    Then use model.load_state_dict(...) to load the weights into that architecture.
"""



# Model definitions 
class CNN_BiLSTM(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        
        lstm_input_size = 64 * (n_mels // 8)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=128, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x).squeeze()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

class EnsembleModel:
    def __init__(self, cnn_model, mlp_model, wav2vec_model, alpha, device):
        self.cnn_model = cnn_model
        self.mlp_model = mlp_model
        self.wav2vec_model = wav2vec_model
        self.alpha = alpha
        self.device = device
        
        # Set models to eval mode
        self.cnn_model.eval()
        self.mlp_model.eval()
        self.wav2vec_model.eval()
    
    def predict_audio_file(self, audio_path):
        """Predict single audio file using ensemble"""
        # Audio processing parameters (must match training)
        SAMPLE_RATE = 16000
        DURATION = 3  # seconds
        SEGMENT_SAMPLES = SAMPLE_RATE * DURATION
        N_MELS = 128
        N_FFT = 1024
        HOP_LENGTH = 512
        
        # Generate spectrogram
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        y = librosa.util.fix_length(y, size=SEGMENT_SAMPLES)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        spec_tensor = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Generate wav2vec embedding
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            features, _ = self.wav2vec_model(waveform)
            embedding = features.mean(dim=1).squeeze().unsqueeze(0)
        
        # Get predictions from both models
        with torch.no_grad():
            spec_tensor = spec_tensor.to(self.device)
            cnn_pred = torch.sigmoid(self.cnn_model(spec_tensor)).cpu().item()
            mlp_pred = torch.sigmoid(self.mlp_model(embedding)).cpu().item()
        
        # Ensemble prediction
        ensemble_pred = self.alpha * cnn_pred + (1 - self.alpha) * mlp_pred
        
        return {
            'ensemble_score': ensemble_pred,
            'cnn_score': cnn_pred,
            'mlp_score': mlp_pred,
            'prediction': 'FAKE' if ensemble_pred > 0.5 else 'REAL'
        }


def safe_load_model(model_path):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def load_ensemble_model(model_path):
    """Load the saved ensemble model"""
    # Load wav2vec 2.0 model
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec_model = bundle.get_model().to(device).eval()
    
    # Load saved ensemble model with safe loading
    saved_data = safe_load_model(model_path)
    if saved_data is None:
        return None
    
    # Initialize models
    cnn_model = CNN_BiLSTM(n_mels=saved_data['model_config']['n_mels']).to(device)
    mlp_model = MLPClassifier(saved_data['embedding_dim']).to(device)
    
    # Load weights
    cnn_model.load_state_dict(saved_data['cnn_state_dict'])
    mlp_model.load_state_dict(saved_data['mlp_state_dict'])
    
    # Create ensemble model
    ensemble_model = EnsembleModel(
        cnn_model=cnn_model,
        mlp_model=mlp_model,
        wav2vec_model=wav2vec_model,
        alpha=saved_data['alpha'],
        device=device
    )
    
    return ensemble_model

def main():
    def load_css(file_path):
        with open(file_path, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)   
    load_css("main_app_styles.css")

    def load_lottieurl(url):
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except ValueError:
            return None
        
    lottie_url = "https://lottie.host/dfce807a-fad5-47e1-b29a-ba79f018692c/Fohdw1wMya.json"
    lottie_coding = load_lottieurl(lottie_url)

    if lottie_coding:
        st_lottie(lottie_coding, height=300, width=650, key="lottie")
    else:
        st.warning("⚠️ Could not load animation.")
    
    st.title("Audio Deepfake Detection")

    
    # Load model (cache this so it only loads once)
    model_path = "D:\\BE Project\\Self\\APPROACH3-CNN-BiLSTM\\models\\complete_ensemble_model.pth"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check the path.")
        return
    
    ensemble_model = load_ensemble_model(model_path)
    if ensemble_model is None:
        st.error("Failed to load the model. Please check the error messages above.")
        return
    
    

    option = st.radio("Select Input Mode:", ("Upload Audio File", "Record Audio in Real Time"))
    st.markdown("----------------------------------------------------------")
    temp_path = None

    if option == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV format recommended)",
            type=["wav", "mp3", "ogg", "flac"]
        )
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            st.audio(uploaded_file, format='audio/wav')

    elif option == "Record Audio in Real Time":
        st.markdown("🎤 Use the buttons below to start and stop recording.")
        # Show reading prompt text
        st.markdown("📄 Please read the following text aloud:")

        default_prompt = "Hello, I am testing this system to detect whether this audio is real or AI-generated. Please detect the voice and give me correct results for it. I will check the answer."
        user_prompt = st.text_area("🎙️ Text to Read (You can cutomize)", default_prompt, height=100)


        # Setup session state
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False
            st.session_state.audio_buffer = None

        fs = 16000  # sample rate

        def start_recording():
            st.session_state.is_recording = True
            st.session_state.fs = 16000
            st.session_state.duration = 30  # max length in seconds
            st.session_state.audio_buffer = sd.rec(int(st.session_state.duration * st.session_state.fs),
                                                samplerate=st.session_state.fs, channels=1)
            st.session_state.recording_started = True

        def stop_recording():
            sd.stop()
            st.session_state.is_recording = False

            audio = st.session_state.audio_buffer.squeeze()  # (N,) for librosa

            # Save raw audio to a temp WAV file and CLOSE IT immediately
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw_temp:
                write(raw_temp.name, st.session_state.fs, audio)
                raw_path = raw_temp.name  # store path before closing

            # Load using librosa
            y, sr = librosa.load(raw_path, sr=st.session_state.fs)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # Save trimmed audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as trimmed_file:
                write(trimmed_file.name, sr, (y_trimmed * 32767).astype(np.int16))
                st.session_state.temp_path = trimmed_file.name

            # Now safely delete the raw temp file
            os.unlink(raw_path)



        col1, col2 = st.columns(2)
        with col1:
            st.button("▶️ Start Recording", on_click=start_recording)
        with col2:
            st.button("⏹️ Stop Recording", on_click=stop_recording)

        if st.session_state.is_recording:
            st.info("🔴 Recording... Speak into your microphone.")

        temp_path = st.session_state.get("temp_path", None)
        if temp_path:
            st.audio(temp_path)



    # ========== Prediction ==========
    if temp_path:
        try:
            with st.spinner("Analyzing audio..."):
                result = ensemble_model.predict_audio_file(temp_path)

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", result['prediction'])
            with col2:
                st.metric("Fakeness Confidence (%)", f"{result['ensemble_score'] * 100:.2f}%")

            with st.expander("Detailed Scores"):
                st.write(f"**CNN Model Score:** {result['cnn_score'] * 100:.2f}%")
                st.write(f"**MLP Model Score:** {result['mlp_score'] * 100:.2f}%")
                st.write(f"**Ensemble Weight (α):** {ensemble_model.alpha:.3f}")


        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    main()