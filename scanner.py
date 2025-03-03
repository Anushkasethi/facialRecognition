import cv2
import face_recognition
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
from ultralytics import YOLO
import cv2
import os
import json

firebase_config = st.secrets["firebase"]
firebase_credentials = {key: firebase_config[key] for key in firebase_config.keys()}

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
db = firestore.client()

if "face_encoding" not in st.session_state:
    st.session_state["face_encoding"] = None
if "visitor_name" not in st.session_state:
    st.session_state["visitor_name"] = ""
if "scanned_frame" not in st.session_state:
    st.session_state["scanned_frame"] = None
if "identified_visitor" not in st.session_state:
    st.session_state["identified_visitor"] = None

def scan_face():
    """ Capture and encode a face """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    face_encodings = face_recognition.face_encodings(frame)
    if face_encodings:
        st.success("Face Detected!")
        st.session_state["face_encoding"] = face_encodings[0]
        st.session_state["scanned_frame"] = frame  # Store frame for persistent display
        return True
    return False

def check_database(encoding):
    """ Check if the face exists in Firebase """
    st.write("Checking database...")
    visitors = db.collection("Visitors").stream()
    for visitor in visitors:
        data = visitor.to_dict()
        stored_encoding = np.array(data["encoding"])

        match = face_recognition.compare_faces([stored_encoding], encoding, tolerance=0.5)
        if match[0]:
            return data["name"]  
    return None  

def register_new_visitor():
    """ Save new visitor to Firebase """
    st.write(st.session_state["visitor_name"])
    try:
        if not st.session_state["visitor_name"]:
            st.error("Please enter a name before registering!")
            return
        if st.session_state["face_encoding"] is None:
            st.error("No scanned face to register!")
            return
        st.write("Registering new visitor...")
        # Check if the visitor already exists in the database
        existing_visitor = check_database(st.session_state["face_encoding"])
        if existing_visitor:
            st.warning(f"{existing_visitor} is already registered.")
            return
        
        visitor_data = {
            "name": st.session_state["visitor_name"], 
            "encoding": st.session_state["face_encoding"].tolist()
        }
        db.collection("Visitors").add(visitor_data)
        st.success(f"{st.session_state['visitor_name']} registered successfully!")
        st.session_state["face_encoding"] = None
        st.session_state["visitor_name"] = ""
        st.session_state["scanned_frame"] = None
        st.session_state["identified_visitor"] = None
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

st.title("Visitor Registration System")

# Scan Face Button
if st.button("Scan Face"):
    if scan_face():
        st.session_state["identified_visitor"] = check_database(st.session_state["face_encoding"])

if st.session_state["scanned_frame"] is not None:
    st.image(st.session_state["scanned_frame"], caption="Scanned Face", channels="BGR")
    if st.session_state["identified_visitor"]:
        st.success(f"Visitor Identified: {st.session_state['identified_visitor']}")
    else:
        st.session_state["visitor_name"] = st.text_input("Enter Name for Registration:", st.session_state["visitor_name"])
        if st.button("Register"):
            register_new_visitor()
            st.success("Visitor Registered!")


import urllib.request

model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
model_path = "yolov8n-face.pt"

urllib.request.urlretrieve(model_url, model_path)
# Load the model
model = YOLO(model_path)
st.title("Real-Time Visitor Tracking System")

video_placeholder = st.empty()

if "tracking" not in st.session_state:
    st.session_state.tracking = False  # Default: Not tracking

def recognize_faces(frame, known_encodings, known_names):
    """Detect faces and compare with known encodings to identify people."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)
    labels = []

    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(distances)  
        if distances[best_match_index] < 0.55:  
            labels.append(known_names[best_match_index])  
        else:
            labels.append("Unidentified")  

    return labels

def track_visitors():
    """ Real-time visitor tracking """
    cap = cv2.VideoCapture(0)
    visitors = db.collection("Visitors").stream()
    known_encodings = []
    known_names = []
    
    for visitor in visitors:
        data = visitor.to_dict()
        known_encodings.append(np.array(data["encoding"]))
        known_names.append(data["name"])

    while cap.isOpened() and st.session_state.tracking:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        results = model(frame)  # YOLO detects faces
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        labels = recognize_faces(frame, known_encodings, known_names)
        for i, (label, result) in enumerate(zip(labels, results)):
            for box in result.boxes:
                x1, y1, _, y2 = map(int, box.xyxy[0])
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_column_width=True)

        time.sleep(0.03)  

    cap.release()

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Tracking", key="start_tracking"):
        st.session_state.tracking = True
        track_visitors()

with col2:
    if st.button("Stop Tracking", key="stop_tracking"):
        st.session_state.tracking = False