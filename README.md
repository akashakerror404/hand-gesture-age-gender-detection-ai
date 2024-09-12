

---



# Hand Gesture, Age, and Gender Detection

This project is a real-time computer vision application that detects hand gestures (counts fingers and identifies gestures like "Peace", "Thumbs Up", etc.), estimates age, and predicts gender using a webcam. It combines hand tracking with age and gender detection, providing an interactive experience with live feedback on the screen.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [How It Works](#how-it-works)
- [Demo](#demo)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Technologies Used
This project uses the following technologies:
- **Python 3.x**
- **OpenCV** - For real-time face detection, gesture recognition, and frame processing.
- **MediaPipe** - For hand tracking and finger counting.
- **Deep Learning (Caffe)** - Pre-trained models for age and gender prediction.
- **NumPy** - For numerical computations.
- **Haar Cascade Classifier** - For face detection.

## Features
- Detects hand gestures and counts the number of fingers raised.
- Recognizes gestures like "Thumbs Up", "Peace Sign", "OK Sign", and "High Five".
- Predicts the gender (Male/Female) of the detected face.
- Estimates the age group of the detected face.
- Real-time feedback with information displayed on the screen.
- Uses the webcam for live video capture and processing.

## Installation

### Prerequisites
- Python 3.x installed on your machine.
- A webcam (built-in or external) for real-time video processing.

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/akashakerror404/hand-gesture-age-gender-detection-ai.git
   cd hand-gesture-age-gender-detection
   ```

2. **Install required Python libraries**
   Create a virtual environment (optional but recommended) and install the dependencies using pip:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The `requirements.txt` should contain:
   ```txt
   opencv-python
   mediapipe
   numpy
   ```

3. **Download Pre-trained Models**
   - Download the pre-trained Caffe models for age and gender detection:
     - [Age Model](https://github.com/serengil/deepface_models/releases/download/v1.0/age_net.caffemodel)
     - [Gender Model](https://github.com/serengil/deepface_models/releases/download/v1.0/gender_net.caffemodel)
   
   Place the `.prototxt` and `.caffemodel` files in the same directory as the Python script or provide the path to these models in the script.

## Running the Project

Once the setup is complete, you can run the project with the following command:
```bash
python hand_gesture_age_gender.py
```

This will open up a window displaying the webcam feed. As soon as a face is detected, the age and gender will be predicted, and any detected hand gestures will be displayed on the screen.

## How It Works

1. **Face Detection**: Using OpenCV's Haar Cascade Classifier, the system detects faces in real-time from the webcam feed.
2. **Age and Gender Prediction**: Once a face is detected, the pre-trained Caffe models are used to predict the person's age and gender.
3. **Hand Tracking**: MediaPipe’s hand-tracking model is used to detect and track hands in the video frame. It identifies the positions of the hand landmarks.
4. **Finger Counting and Gesture Recognition**: By analyzing the relative positions of the detected hand landmarks, the system counts the number of fingers raised and identifies certain predefined gestures.
5. **Real-Time Feedback**: All detected information (age, gender, gesture, and finger count) is displayed on the screen for real-time interaction.

## Demo

Here’s a preview of how the application works:

- **Age and Gender Detection**: 
  ![Age and Gender Detection](docs/age-gender-demo.png)

- **Hand Gesture Detection**:
  ![Hand Gesture Detection](docs/hand-gesture-demo.png)

## Acknowledgments
- **MediaPipe**: For providing an excellent hand-tracking library.
- **OpenCV**: For real-time video and image processing.
- **Pre-trained Caffe Models**: For age and gender prediction.
  
