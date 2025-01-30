# Face Detection with Gender and Age Prediction

## Overview
This project leverages Deep Neural Networks (DNN) and OpenCV to achieve real-time face detection and classification. It accurately detects faces in live camera feeds or uploaded images and predicts the gender and age group of detected individuals. Additionally, it features a hand detection module using Mediapipe, which can track hand landmarks for gesture recognition and motion analysis.

## Features
### Face Detection with Gender and Age Prediction
- Real-time face detection using OpenCV's DNN module.
- Predicts gender (Male/Female) and age group of detected individuals.
- Supports live camera feeds and uploaded images.
- High accuracy and efficient performance.

### Hand Detection with Landmark Tracking
- Detects one or both hands with high accuracy.
- Tracks 21 key landmarks on each hand, including fingertips, joints, and the wrist.
- Supports gesture recognition and motion tracking.
- Option to enable or disable landmark tracking.
- Optimized for lightweight and efficient computation.

## Technologies Used
- **Python** - Core programming language for implementing detection algorithms.
- **OpenCV** - Used for face detection and gender & age prediction using deep neural networks.
- **Mediapipe** - Enables hand detection and landmark tracking.
- **Streamlit** - Provides an interactive and user-friendly web interface.
- **NumPy** - Performs efficient numerical computations in image processing.
- **Pillow** - Handles image uploads and draws bounding boxes around detected objects.
- **Requests** - Fetches image URLs for processing and analysis.
- **HTML, CSS, JavaScript** - Enhances the frontend with structured content, styles, animations, and interactivity.

## Installation
### Prerequisites
Ensure you have Python installed (>= 3.7). Install dependencies using the following command:

Clone the repository
```bash
git clone https://github.com/Nishant43S/computer-vision-app.git
```

install requirements
```bash
pip install -r requirements.txt
```

## Usage
### Running the Application
To start the Streamlit web application, execute:

```bash
python -m streamlit run app.py
```

### How It Works
1. Open the web application in your browser.
2. enable the live camera feed.
3. The model detects faces, predicts gender and age, and optionally tracks hand landmarks.
4. View real-time results with bounding boxes and annotations.

## Project Structure
```
📂 Face-Detection-Project
 ├── 📄 app.py                 # Main application script
 ├── 📄 .py                 # Face Detection script
 ├── 📂 models                 # Pre-trained models for face detection
 ├── 📂 static                 # CSS, JS, and images
 ├── 📂 templates              # HTML templates
 ├── 📄 requirements.txt        # Dependencies
 ├── 📄 README.md               # Project documentation
```

## Screenshots




## Contributors
- **Nishant Maity** - Developer

## License
This project is licensed under the MIT License. Feel free to modify and distribute it as per the license terms.

---
**Enjoy using the Face Detection and Hand Tracking System! 🚀**
