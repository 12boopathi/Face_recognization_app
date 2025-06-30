import streamlit as st
import time
import uuid
import tempfile
import threading
import whisper
import pyttsx3
import subprocess
import os
import queue
import cv2
import numpy as np
from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# --- CONFIGURATIONS ---
WHISPER_MODEL = "tiny"
NAME_PROMPT = "Adarsh, Aarav, Nayan, Riya, Lakshmi, Arjun, Priya, Deepa, Neha, Rohan, Anjali, Vijay, Vinay, Vikram, Sanjay, Suraj"
COMMAND_PROMPT = "new, add, remove, delete, quit"
CONFIRM_PROMPT = "yes, no"
AUDIO_DEVICE = "plughw:2,0"
TTS_DEVICE = "plughw:5,0"
COMMAND_DURATION = 10
NAME_DURATION = 4
CONFIRM_DURATION = 3
TTS_DEBOUNCE = 3.0
COMMAND_DEBOUNCE = 2.0
TEXT_TIMEOUT = 5.0
RECORDING_TIMEOUT = 6.0
PADDING = 10

# --- INIT SESSION STATE ---
def init_session_state():
    if 'operation' not in st.session_state:
        st.session_state['operation'] = None
    if 'add_state' not in st.session_state:
        st.session_state['add_state'] = None
    if 'add_name' not in st.session_state:
        st.session_state['add_name'] = None
    if 'remove_state' not in st.session_state:
        st.session_state['remove_state'] = None
    if 'remove_name' not in st.session_state:
        st.session_state['remove_name'] = None
    if 'recording_message' not in st.session_state:
        st.session_state['recording_message'] = None
    if 'recording_message_time' not in st.session_state:
        st.session_state['recording_message_time'] = 0
    if 'is_recording' not in st.session_state:
        st.session_state['is_recording'] = False
    if 'last_embedding' not in st.session_state:
        st.session_state['last_embedding'] = None
    if "running" not in st.session_state:
        st.session_state["running"] = True
    if 'detector' not in st.session_state:
        st.session_state.detector = YoloFace("yoloface_int8.tflite", "")
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = Facenet("facenet_512_int_quantized.tflite", "")
    if 'face_db' not in st.session_state:
        st.session_state.face_db = FaceDatabase()

# --- WHISPER MODEL ---
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model(WHISPER_MODEL)
    return model
whisper_model = load_whisper_model()

# --- TTS ---
tts_lock = threading.Lock()
class TTSEngine:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS init error: {e}")
            self.engine = None
        self.tts_thread = None
        self.last_tts_time = 0
        self.current_text = ""
        self.text_start_time = 0
    def say(self, text):
        current_time = time.time()
        if self.tts_thread and self.tts_thread.is_alive():
            return
        if current_time - self.last_tts_time < TTS_DEBOUNCE:
            return
        self.current_text = text
        self.text_start_time = current_time
        def run_tts():
            with tts_lock:
                if not self.engine:
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmp_file = tmpfile.name
                try:
                    self.engine.save_to_file(text, tmp_file)
                    self.engine.runAndWait()
                    subprocess.run(f"aplay -D {TTS_DEVICE} {tmp_file}", shell=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = current_time
tts = TTSEngine()

# --- AUDIO PROCESSING FUNCTIONS ---
def record_audio(filename, duration=4):
    try:
        st.session_state['recording_message'] = "Recording started..."
        st.session_state['recording_message_time'] = time.time()
        st.session_state['is_recording'] = True
        cmd = ['arecord', '-D', AUDIO_DEVICE, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-d', str(duration), filename]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(0.5)
        st.session_state['recording_message'] = "Recording stopped."
        st.session_state['recording_message_time'] = time.time()
        st.session_state['is_recording'] = False
        return os.path.exists(filename)
    except Exception as e:
        print(f"Audio error: {e}")
        st.session_state['recording_message'] = None
        st.session_state['is_recording'] = False
        return False

def whisper_transcribe(audio_file, prompt_context=""):
    try:
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, prompt=prompt_context, temperature=0.5)
        result = whisper.decode(whisper_model, mel, options)
        text = result.text.strip().lower()
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
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
    words = [w.strip().capitalize() for w in name_text.split() if w.strip().isalpha() and w.lower() not in ['names', 'name', 'example']]
    if not words or len(words) > 2:
        return None
    name = ' '.join(words)
    return name

def recognize_confirmation():
    tmp_file = f"confirm_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, CONFIRM_DURATION):
        return None
    confirm_text = whisper_transcribe(tmp_file, f"Responses: {CONFIRM_PROMPT}")
    if "yes" in confirm_text:
        return "yes"
    elif "no" in confirm_text:
        return "no"
    return None

# --- FACE DB ACCESS HELPERS ---
def add_name_to_db(name, embedding):
    st.session_state.face_db.add_name(name, embedding)

def delete_name_from_db(name):
    st.session_state.face_db.del_name(name)

def get_all_names():
    return st.session_state.face_db.get_names()

def find_name_from_embedding(embedding):
    return st.session_state.face_db.find_name(embedding)

# --- MAIN STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Real-time Face Recognition", layout="wide")
    st.title("😎 Real-time Face Recognition")
    init_session_state()

    # Status/info messages
    if st.session_state['recording_message']:
        rec_time_diff = time.time() - st.session_state['recording_message_time']
        if rec_time_diff < TEXT_TIMEOUT:
            st.info(st.session_state['recording_message'])
    if st.session_state['is_recording']:
        st.warning("Recording in progress...")

    col1, col2 = st.columns([2, 1])
    # --- CONTROLS ---
    with col2:
        st.subheader("🕹️ Controls")
        st.session_state['running'] = st.checkbox("Run Camera", value=st.session_state['running'])
        if st.button("🎙️ Voice Command"):
            command = recognize_command()
            if command == "add":
                st.session_state["operation"] = "add"
                st.session_state["add_state"] = "ask_name"
                tts.say("Say the name to add after clicking the button")
            elif command == "delete":
                st.session_state["operation"] = "delete"
                st.session_state["remove_state"] = "ask_name"
                tts.say("Say the name to delete after clicking the button")
            elif command == "quit":
                st.session_state["running"] = False
                st.session_state["operation"] = None
                tts.say("Quitting camera")
            else:
                st.session_state["operation"] = None
                tts.say("Unknown command")
        if st.button("Reset State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("### 📋 Database")
        names = get_all_names()
        st.markdown("\n".join(f"- {name}" for name in names) if names else "Database is empty!")

        # --- ADD LOGIC ---
        if st.session_state.get('operation') == "add":
            st.subheader("Add New Name")
            if st.session_state.get("add_state") == "ask_name":
                if st.button("🎤 Record Name"):
                    name = recognize_name()
                    if name:
                        st.session_state["add_name"] = name
                        st.session_state["add_state"] = "confirm"
                        tts.say(f"Do you want to add {name}? Click Yes or No.")
                        st.rerun()
            if st.session_state.get("add_state") == "confirm" and st.session_state.get("add_name"):
                st.write(f"Add name: {st.session_state['add_name']}?")
                coly1, coly2 = st.columns(2)
                with coly1:
                    if st.button("Yes"):
                        if st.session_state.get('last_embedding') is not None:
                            add_name_to_db(st.session_state['add_name'], st.session_state['last_embedding'])
                            tts.say(f"Added {st.session_state['add_name']}")
                        else:
                            tts.say("No face detected to add.")
                        st.session_state["operation"] = None
                        st.session_state["add_state"] = None
                        st.session_state["add_name"] = None
                        st.rerun()
                with coly2:
                    if st.button("No"):
                        tts.say("Name not added")
                        st.session_state["operation"] = None
                        st.session_state["add_state"] = None
                        st.session_state["add_name"] = None
                        st.rerun()

        # --- DELETE LOGIC (VOICE) ---
        if st.session_state.get('operation') == "delete":
            st.subheader("Delete Name")
            if st.session_state.get("remove_state") == "ask_name":
                if st.button("🎤 Record Name to Delete"):
                    name = recognize_name()
                    if name:
                        st.session_state["remove_name"] = name
                        st.session_state["remove_state"] = "confirm"
                        tts.say(f"Do you want to delete {name}? Click Yes or No.")
                        st.rerun()
            if st.session_state.get("remove_state") == "confirm" and st.session_state.get("remove_name"):
                st.write(f"Delete name: {st.session_state['remove_name']}?")
                coly1, coly2 = st.columns(2)
                with coly1:
                    if st.button("Yes", key="delete_yes"):
                        if st.session_state['remove_name'] in get_all_names():
                            delete_name_from_db(st.session_state['remove_name'])
                            tts.say(f"Deleted {st.session_state['remove_name']}")
                        else:
                            tts.say(f"{st.session_state['remove_name']} not found.")
                        st.session_state["operation"] = None
                        st.session_state["remove_state"] = None
                        st.session_state["remove_name"] = None
                        st.rerun()
                with coly2:
                    if st.button("No", key="delete_no"):
                        tts.say("Name not deleted")
                        st.session_state["operation"] = None
                        st.session_state["remove_state"] = None
                        st.session_state["remove_name"] = None
                        st.rerun()

        # --- DELETE OPTION (UI) ---
        st.markdown("---")
        st.subheader("Delete Name (UI)")
        if names:
            selected_name = st.selectbox("Select name to delete", names, key="ui_delete_select")
            if st.button("Confirm Removal (UI)"):
                delete_name_from_db(selected_name)
                tts.say(f"Removed {selected_name}")
                st.rerun()
        else:
            st.info("No names to delete.")

    # --- VIDEO FEED ---
    with col1:
        st.subheader("📷 Live Camera Feed")
        frame_placeholder = st.empty()
        if st.session_state['running']:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access the webcam.")
            else:
                while st.session_state['running']:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.resize(frame, (640, 480))
                    boxes = st.session_state.detector.detect(frame)
                    current_embedding = None
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    # Store last face embedding for Add
                    st.session_state['last_embedding'] = current_embedding
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    # Let Streamlit render other components
                    time.sleep(0.03)
            cap.release()
        else:
            frame_placeholder.warning("⏸️ Camera Paused")

if __name__ == "__main__":
    main()
