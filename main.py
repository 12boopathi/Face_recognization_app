import streamlit as st
import time
import uuid
import tempfile
import threading
import whisper
import pyttsx3
import subprocess
import os
import cv2
import numpy as np
from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# --- CONFIGURATIONS ---
WHISPER_MODEL = "tiny"
NAME_PROMPT = "Adarsh, Aarav, Nayan, Riya, Lakshmi, Arjun, Priya, Deepa, Neha, Rohan, Anjali, Vijay, Vinay, Vikram, Sanjay, Suraj"
COMMAND_PROMPT = "new, add, remove, delete, quit"
DEFAULT_AUDIO_DEVICE = "default"
DEFAULT_TTS_DEVICE = "default"
COMMAND_DURATION = 10
NAME_DURATION = 4
CONFIRM_DURATION = 3
TTS_DEBOUNCE = 3.0
PADDING = 10
DEFAULT_VIDEO_DEVICE_INDEX = 0

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-bar {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .success-status {
        background: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .warning-status {
        background: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    .error-status {
        background: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .control-panel {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .camera-section {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .name-list {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .recording-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ff4757;
        border-radius: 50%;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- DEVICE ENUMERATION HELPERS ---
def get_audio_input_devices():
    # List audio input devices (capture)
    devices = []
    try:
        arecord_output = subprocess.check_output(['arecord', '-l'], stderr=subprocess.DEVNULL).decode()
        for line in arecord_output.splitlines():
            if 'card' in line and 'device' in line:
                import re
                m = re.search(r'card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+): ([^\[]+)\[([^\]]+)\]', line)
                if m:
                    card_num = m.group(1)
                    card_desc = m.group(3).strip()
                    device_num = m.group(4)
                    device_desc = m.group(6).strip()
                    hw_id = f"plughw:{card_num},{device_num}"
                    disp = f"{card_desc} - {device_desc} ({hw_id})"
                    devices.append((disp, hw_id))
    except Exception:
        pass
    if not devices:
        devices = [("Default", DEFAULT_AUDIO_DEVICE)]
    return devices

def get_audio_output_devices():
    # List audio output devices (playback)
    devices = []
    try:
        aplay_output = subprocess.check_output(['aplay', '-l'], stderr=subprocess.DEVNULL).decode()
        for line in aplay_output.splitlines():
            if 'card' in line and 'device' in line:
                import re
                m = re.search(r'card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+): ([^\[]+)\[([^\]]+)\]', line)
                if m:
                    card_num = m.group(1)
                    card_desc = m.group(3).strip()
                    device_num = m.group(4)
                    device_desc = m.group(6).strip()
                    hw_id = f"plughw:{card_num},{device_num}"
                    disp = f"{card_desc} - {device_desc} ({hw_id})"
                    devices.append((disp, hw_id))
    except Exception:
        pass
    if not devices:
        devices = [("Default", DEFAULT_TTS_DEVICE)]
    return devices

def get_video_devices(max_devices=10):
    # List connected video devices (webcams or capture cards)
    available = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            available.append((f"Camera {idx}", idx))
            cap.release()
    if not available:
        available = [("Default (0)", 0)]
    return available

# --- INIT SESSION STATE ---
def init_session_state():
    defaults = {
        'operation': None, 'add_state': None, 'add_name': None,
        'remove_state': None, 'remove_name': None,
        'is_recording': False, 'last_embedding': None,
        'running': False, 'status_message': 'System Ready',
        'status_type': 'info', 'camera_status': 'Disconnected',
        'total_faces': 0, 'faces_detected': 0,
        'audio_device': DEFAULT_AUDIO_DEVICE,
        'tts_device': DEFAULT_TTS_DEVICE,
        'video_device_index': DEFAULT_VIDEO_DEVICE_INDEX,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Initialize models
    if 'detector' not in st.session_state:
        st.session_state.detector = YoloFace("yoloface_int8.tflite", "")
        update_status("Face detector loaded", "success")
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = Facenet("facenet_512_int_quantized.tflite", "")
        update_status("Face recognizer loaded", "success")
    if 'face_db' not in st.session_state:
        st.session_state.face_db = FaceDatabase()
        update_status("Database initialized", "success")

# --- STATUS MANAGEMENT ---
def update_status(message, status_type="info"):
    st.session_state['status_message'] = message
    st.session_state['status_type'] = status_type
    st.session_state['status_time'] = time.time()

def render_status_bar():
    status_class = f"{st.session_state['status_type']}-status"
    icon_map = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
    icon = icon_map.get(st.session_state['status_type'], 'ℹ️')
    st.markdown(f"""
    <div class="status-bar {status_class}">
        <strong>{icon} Status:</strong> {st.session_state['status_message']}
    </div>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL)
whisper_model = load_whisper_model()

# --- TTS ENGINE ---
class TTSEngine:
    def __init__(self, tts_device=None):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
        except Exception:
            self.engine = None
        self.tts_thread = None
        self.last_tts_time = 0
        self.tts_device = tts_device or st.session_state.get("tts_device", DEFAULT_TTS_DEVICE)

    def say(self, text):
        if self.tts_thread and self.tts_thread.is_alive():
            return
        if time.time() - self.last_tts_time < TTS_DEBOUNCE:
            return
        def run_tts():
            if not self.engine:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_file = tmpfile.name
            try:
                self.engine.save_to_file(text, tmp_file)
                self.engine.runAndWait()
                tts_dev = st.session_state.get("tts_device", DEFAULT_TTS_DEVICE)
                subprocess.run(f"aplay -D {tts_dev} {tmp_file}", shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = time.time()

# --- AUDIO FUNCTIONS ---
def record_audio(filename, duration=4):
    try:
        st.session_state['is_recording'] = True
        update_status(f"🎤 Recording audio for {duration}s...", "warning")
        audio_dev = st.session_state.get("audio_device", DEFAULT_AUDIO_DEVICE)
        cmd = ['arecord', '-D', audio_dev, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        st.session_state['is_recording'] = False
        update_status("Recording completed", "success")
        return os.path.exists(filename)
    except Exception:
        st.session_state['is_recording'] = False
        update_status("Recording failed", "error")
        return False

def whisper_transcribe(audio_file, prompt_context=""):
    try:
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, prompt=prompt_context, temperature=0.5)
        result = whisper.decode(whisper_model, mel, options)
        return result.text.strip().lower()
    except Exception:
        return ""
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

def recognize_command():
    tmp_file = f"cmd_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, COMMAND_DURATION):
        return ""
    cmd_text = whisper_transcribe(tmp_file, f"Commands: {COMMAND_PROMPT}")
    if "new" in cmd_text or "add" in cmd_text:
        return "add"
    elif "remove" in cmd_text or "delete" in cmd_text:
        return "delete"
    elif "quit" in cmd_text or "exit" in cmd_text:
        return "quit"
    return ""

def recognize_name():
    tmp_file = f"name_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, NAME_DURATION):
        return None
    name_text = whisper_transcribe(tmp_file, f"Example names: {NAME_PROMPT}")
    if not name_text:
        return None
    import string
    name_text = name_text.translate(str.maketrans('', '', string.punctuation))
    words = [w.strip().capitalize() for w in name_text.split()
             if w.strip().isalpha() and w.lower() not in ['names', 'name', 'example']]
    if not words or len(words) > 2:
        return None
    return ' '.join(words)

# --- DATABASE HELPERS ---
def add_name_to_db(name, embedding):
    st.session_state.face_db.add_name(name, embedding)
    st.session_state.total_faces = len(get_all_names())

def delete_name_from_db(name):
    st.session_state.face_db.del_name(name)
    st.session_state.total_faces = len(get_all_names())

def get_all_names():
    return st.session_state.face_db.get_names()

def find_name_from_embedding(embedding):
    return st.session_state.face_db.find_name(embedding)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Face Recognition System", layout="wide", initial_sidebar_state="collapsed")
    load_css()
    init_session_state()

    # --- Device Selection Section ---
    with st.sidebar:
        st.header("🔧 Device Settings")
        # Video Device
        video_devices = get_video_devices(10)
        video_device_labels = [d[0] for d in video_devices]
        video_device_indices = [d[1] for d in video_devices]
        vdev_idx = 0
        # Try to keep previous selection if possible
        try:
            vdev_idx = video_device_indices.index(st.session_state.get("video_device_index", 0))
        except Exception:
            vdev_idx = 0
        video_sel = st.selectbox("Video Device", video_device_labels, index=vdev_idx)
        st.session_state["video_device_index"] = video_devices[video_device_labels.index(video_sel)][1]

        # Audio IN device
        audio_in_devices = get_audio_input_devices()
        audio_in_labels = [d[0] for d in audio_in_devices]
        audio_in_hwids = [d[1] for d in audio_in_devices]
        adev_idx = 0
        try:
            adev_idx = audio_in_hwids.index(st.session_state.get("audio_device", DEFAULT_AUDIO_DEVICE))
        except Exception:
            adev_idx = 0
        audio_in_sel = st.selectbox("Audio Input Device", audio_in_labels, index=adev_idx)
        st.session_state["audio_device"] = audio_in_devices[audio_in_labels.index(audio_in_sel)][1]

        # Audio OUT (TTS) device
        audio_out_devices = get_audio_output_devices()
        audio_out_labels = [d[0] for d in audio_out_devices]
        audio_out_hwids = [d[1] for d in audio_out_devices]
        tdev_idx = 0
        try:
            tdev_idx = audio_out_hwids.index(st.session_state.get("tts_device", DEFAULT_TTS_DEVICE))
        except Exception:
            tdev_idx = 0
        audio_out_sel = st.selectbox("TTS Output Device", audio_out_labels, index=tdev_idx)
        st.session_state["tts_device"] = audio_out_devices[audio_out_labels.index(audio_out_sel)][1]

    # Assign TTS engine with current device
    global tts
    tts = TTSEngine(tts_device=st.session_state["tts_device"])

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced Face Recognition System</h1>
        <p>AI-Powered Face Detection & Voice Control</p>
    </div>
    """, unsafe_allow_html=True)

    # Status Bar
    render_status_bar()

    # Main Layout
    col1, col2 = st.columns([2, 1])

    # Camera Section
    with col1:
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.subheader("📹 Live Camera Feed")

        # Camera controls
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🟢 Start Camera", use_container_width=True):
                st.session_state['running'] = True
                st.session_state['camera_status'] = 'Connected'
                update_status("Camera started", "success")
                st.rerun()
        with col1b:
            if st.button("🔴 Stop Camera", use_container_width=True):
                st.session_state['running'] = False
                st.session_state['camera_status'] = 'Disconnected'
                update_status("Camera stopped", "warning")
                st.rerun()
        with col1c:
            camera_status_color = "🟢" if st.session_state['camera_status'] == 'Connected' else "🔴"
            st.write(f"{camera_status_color} {st.session_state['camera_status']}")

        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # Control Panel
    with col2:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🎛️ Control Panel")

        # Recording indicator
        if st.session_state['is_recording']:
            st.markdown('<span class="recording-indicator"></span> **Recording in progress...**', unsafe_allow_html=True)

        # Voice Commands
        st.markdown("**🎙️ Voice Commands**")
        if st.button("🎤 Give Voice Command", use_container_width=True):
            command = recognize_command()
            if command == "add":
                st.session_state["operation"] = "add"
                st.session_state["add_state"] = "ask_name"
                update_status("Ready to add new face", "info")
                tts.say("Say the name to add")
            elif command == "delete":
                st.session_state["operation"] = "delete"
                st.session_state["remove_state"] = "ask_name"
                update_status("Ready to delete face", "info")
                tts.say("Say the name to delete")
            elif command == "quit":
                st.session_state["running"] = False
                st.session_state["operation"] = None
                update_status("Camera stopped by voice command", "warning")
                tts.say("Camera stopped")
            else:
                update_status("Unknown voice command", "error")
                tts.say("Unknown command")
            st.rerun()

        # Statistics
        st.markdown("**📊 Statistics**")
        names = get_all_names()
        col2a, col2b = st.columns(2)
        with col2a:
            st.markdown(f'<div class="metric-card"><h3>{len(names)}</h3><p>Total Faces</p></div>', unsafe_allow_html=True)
        with col2b:
            st.markdown(f'<div class="metric-card"><h3>{st.session_state.get("faces_detected", 0)}</h3><p>Detected</p></div>', unsafe_allow_html=True)

        # Database
        st.markdown("**👥 Face Database**")
        if names:
            st.markdown('<div class="name-list">', unsafe_allow_html=True)
            for i, name in enumerate(names, 1):
                st.markdown(f"**{i}.** {name}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No faces in database")

        # Add Operations
        if st.session_state.get('operation') == "add":
            st.markdown("---")
            st.markdown("**➕ Add New Face**")
            if st.session_state.get("add_state") == "ask_name":
                if st.button("🎤 Record Name", use_container_width=True):
                    name = recognize_name()
                    if name:
                        st.session_state["add_name"] = name
                        st.session_state["add_state"] = "confirm"
                        update_status(f"Ready to add: {name}", "info")
                        tts.say(f"Confirm to add {name}")
                        st.rerun()
            if st.session_state.get("add_state") == "confirm" and st.session_state.get("add_name"):
                st.success(f"Add: **{st.session_state['add_name']}**?")
                cola, colb = st.columns(2)
                with cola:
                    if st.button("✅ Yes", use_container_width=True):
                        if st.session_state.get('last_embedding') is not None:
                            add_name_to_db(st.session_state['add_name'], st.session_state['last_embedding'])
                            update_status(f"Added {st.session_state['add_name']}", "success")
                            tts.say(f"Added {st.session_state['add_name']}")
                        else:
                            update_status("No face detected", "error")
                            tts.say("No face detected")
                        st.session_state["operation"] = None
                        st.session_state["add_state"] = None
                        st.session_state["add_name"] = None
                        st.rerun()
                with colb:
                    if st.button("❌ No", use_container_width=True):
                        update_status("Add operation cancelled", "warning")
                        tts.say("Cancelled")
                        st.session_state["operation"] = None
                        st.session_state["add_state"] = None
                        st.session_state["add_name"] = None
                        st.rerun()

        # Delete Operations
        if st.session_state.get('operation') == "delete":
            st.markdown("---")
            st.markdown("**🗑️ Delete Face**")
            if st.session_state.get("remove_state") == "ask_name":
                if st.button("🎤 Record Name to Delete", use_container_width=True):
                    name = recognize_name()
                    if name:
                        st.session_state["remove_name"] = name
                        st.session_state["remove_state"] = "confirm"
                        update_status(f"Ready to delete: {name}", "warning")
                        tts.say(f"Confirm to delete {name}")
                        st.rerun()
            if st.session_state.get("remove_state") == "confirm" and st.session_state.get("remove_name"):
                st.warning(f"Delete: **{st.session_state['remove_name']}**?")
                cola, colb = st.columns(2)
                with cola:
                    if st.button("✅ Yes", key="del_yes", use_container_width=True):
                        if st.session_state['remove_name'] in names:
                            delete_name_from_db(st.session_state['remove_name'])
                            update_status(f"Deleted {st.session_state['remove_name']}", "success")
                            tts.say(f"Deleted {st.session_state['remove_name']}")
                        else:
                            update_status("Name not found", "error")
                            tts.say("Name not found")
                        st.session_state["operation"] = None
                        st.session_state["remove_state"] = None
                        st.session_state["remove_name"] = None
                        st.rerun()
                with colb:
                    if st.button("❌ No", key="del_no", use_container_width=True):
                        update_status("Delete operation cancelled", "info")
                        tts.say("Cancelled")
                        st.session_state["operation"] = None
                        st.session_state["remove_state"] = None
                        st.session_state["remove_name"] = None
                        st.rerun()

        # Quick Delete
        if names and not st.session_state.get('operation'):
            st.markdown("---")
            st.markdown("**🗑️ Quick Delete**")
            selected_name = st.selectbox("Select name:", names, key="quick_del")
            if st.button("🗑️ Delete Selected", use_container_width=True):
                delete_name_from_db(selected_name)
                update_status(f"Deleted {selected_name}", "success")
                tts.say(f"Deleted {selected_name}")
                st.rerun()

        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['detector', 'recognizer', 'face_db', 'audio_device', 'tts_device', 'video_device_index']:
                    del st.session_state[key]
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Camera Processing
    if st.session_state['running']:
        video_index = st.session_state.get("video_device_index", 0)
        cap = cv2.VideoCapture(video_index)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            st.session_state['camera_status'] = 'Error'
            update_status("Camera access failed", "error")
        else:
            st.session_state['camera_status'] = 'Connected'
            detected_count = 0
            while st.session_state['running']:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (640, 480))
                boxes = st.session_state.detector.detect(frame)
                current_embedding = None
                detected_count = len(boxes)
                for box in boxes:
                    box[[0, 2]] *= frame.shape[1]
                    box[[1, 3]] *= frame.shape[0]
                    x1, y1, x2, y2 = box.astype(np.int32)
                    x1, y1 = max(x1 - PADDING, 0), max(y1 - PADDING, 0)
                    x2, y2 = min(x2 + PADDING, frame.shape[1]), min(y2 + PADDING, frame.shape[0])
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    embeddings = st.session_state.recognizer.get_embeddings(face)
                    name = "Unknown"
                    if embeddings is not None:
                        current_embedding = embeddings
                        try:
                            name = find_name_from_embedding(embeddings)
                        except Exception:
                            name = "Error"
                    color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                st.session_state['last_embedding'] = current_embedding
                st.session_state['faces_detected'] = detected_count
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.03)
        cap.release()
    else:
        frame_placeholder.info("📷 Camera is stopped. Click 'Start Camera' to begin.")

if __name__ == "__main__":
    main()
