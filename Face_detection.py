import streamlit as st
from streamlit_extras.colored_header import colored_header
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np

## page setting
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
)

#### app functions

### insert external css
@st.cache_data
def insert_css(css_file:str):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


### jquarery animation
def particle(Js_file):
    with open(Js_file) as f:
        components.html(f"{f.read()}", height=400)
    
######################  face detection functions  ######################

### Load pre-trained models
AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"


### Load models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

#### Age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

### Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def detect_faces_and_display(web_cam):
    """
    Main function to perform face detection and age/gender prediction.
    """
    # Initialize webcam
    video_capture = cv2.VideoCapture(web_cam)

    if not video_capture.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    stframe = st.empty()  # Placeholder for the video feed

    while st.session_state["toggle"]:  # Run detection while toggle is ON
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video frame. Exiting...")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the face region
            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            ### Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            ### Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            # Display the label and box
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ####  Streamlit video window
        stframe.image(frame, channels="RGB", use_container_width=False,width=920)

    # Release the webcam
    video_capture.release()
    cv2.destroyAllWindows()


#########################  hand dection  ##########################



###  Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


### Function to detect hands and draw landmarks
def detect_hand_and_display(frame, show_landmarks):
    """
    this function is used to detect hand (Left, Right).
    or show hand land marks on video
    """
    # Flip the frame horizontally for a selfie view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=6, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Process the frame for hand detection
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = "Left" if hand_class.classification[0].label == "Left" else "Right"

                # Display hand type
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
                cv2.putText(frame, f"{hand_type} Hand", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                ### Draw landmarks if toggle is True
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    )
    return frame

def select_cam(webcam):
    if webcam == "Default Webcam":
        return 0
    elif webcam == "Webcam 1":
        return 1
    elif webcam == "Webcam 2":
        return 2

#### sidebar for face and hand detection
face_detection_sidebar = st.sidebar

with face_detection_sidebar:
    ### detection type select
    select_detection_type = st.selectbox(
        label="Select Type",
        options=["Gender Detection","Hand Detection"],
        key="select box for face and hand detection"
    )

    ### webcam select box
    webcam_selection = st.selectbox(
        label="Select Webcam",
        options=["Default Webcam","Webcam 1","Webcam 2"],
        index=0,
        key="web cam selection"
    )


    
### creating app columns
face_detection_col = st.columns([2,8,2],gap="small")

### blank columns
with face_detection_col[0]:
    pass
with face_detection_col[2]:
    pass

### main app column
with face_detection_col[1]:

    ### app heading
    colored_header(
        label="Face & Hand Detection",
        color_name="violet-70",
        description="Detect Gender, Age & Hands"
    )

    ### if face detection is selected
    if select_detection_type == "Gender Detection":
        ### session state for toggle button
        if "toggle" not in st.session_state:
            st.session_state["toggle"] = False

        ### toggle button for enable face detection
        toggle_detection_face = st.toggle("Face Detection", key="toggle",value=False)

        if toggle_detection_face:
            detect_faces_and_display(web_cam=select_cam(webcam=webcam_selection))
        else:
            particle("jsanimation/particles.html")

    elif select_detection_type == "Hand Detection":

        Hand_detection_col = st.columns([3,3,2,4],gap="small")

        with Hand_detection_col[0]:
            ### toggle button for hand detection
            toggle_detection_hand = st.toggle(
                label="Hand Detection",key="toggle",
                value=False
            )
        with Hand_detection_col[1]:
            ### toggle button for hand detection
            toggle_detection_landmark = st.checkbox(
                label="Landmarks",key="landmark enabling",
                value=False
            )
            
        with Hand_detection_col[1]:
            # st.write('')
            pass

        with Hand_detection_col[2]:
            pass
        
        ### condition for landmark detection
        if not toggle_detection_hand and toggle_detection_landmark == True:
            st.warning("Turn on hand detection",icon="⚠️")

        ### hand detection
        if toggle_detection_hand:
            ## Access the webcam
            cap = cv2.VideoCapture(select_cam(webcam=webcam_selection))
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to access webcam. Please ensure it is connected and available.")
                    break

                ### Process the frame
                processed_frame = detect_hand_and_display(frame, toggle_detection_landmark)

                ### Display the frame in Streamlit
                stframe.image(processed_frame, channels="BGR", use_container_width=False,width=920)

            cap.release()
            
            
        else:
            ### js animation
            particle("jsanimation/particles.html")

insert_css("cssfiles/settings.css")
