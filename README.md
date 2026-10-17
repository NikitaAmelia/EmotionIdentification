
# 🎭 Emotion Identification using Deep Learning

A **Computer Vision project** that performs facial emotion detection using a browser camera and Deep Learning.  
This application captures a facial image and predicts the **dominant human emotion** using DeepFace.

The project demonstrates the implementation of **Face Detection + Emotion Recognition** using Python, OpenCV, TensorFlow, and DeepFace, with a web-based interface built using Streamlit.

---

## 🔗 Live Demo (Streamlit App)

👉 [https://emotionidentification.streamlit.app/]

---

## 🚀 Features

- 📷 Capture facial images using a browser camera
- 😊 Automatic emotion classification
- 🧠 Deep Learning–based emotion analysis
- 🔍 Face detection and emotion prediction
- 📊 Display emotion analysis results through a web interface
- 🌐 Web-based application accessible through a browser

---

## 🧠 Technologies Used

- Python
- Streamlit
- OpenCV
- DeepFace
- TensorFlow / Keras
- Haar Cascade Classifier

---

## 📂 Project Structure

```text
EmotionIdentification/
│
├── app.py
├── requirements.txt
├── packages.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/nikitaamelia/emotionidentification.git
cd emotionidentification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the application locally:

```bash
streamlit run app.py
```

- Open the application in your browser.
- Allow camera access when prompted.
- Capture an image using the browser camera.
- Click **Analyze Emotions** to process the captured image.
- View the detected face and emotion analysis results.

**Note:** This application analyzes captured images rather than processing a continuous live video feed.

---

## 🎯 Supported Emotions

DeepFace analyzes facial expressions and identifies emotion categories such as:

- Happy
- Sad
- Angry
- Surprise
- Fear
- Neutral
- Disgust

---

## ☁️ Deployment

This application can be deployed using **Streamlit Community Cloud**.

1. Push the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new application and select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the application and wait for the installation process to finish.

---

## 💡 Project Purpose

This project was developed as part of a **Computer Vision & AI portfolio** to demonstrate skills in:

- Image processing
- Face detection
- Deep Learning integration
- Facial emotion recognition
- Web application development using Streamlit
- Deployment of AI applications

---

## 📌 Future Improvements

- Continuous real-time emotion detection
- Emotion recognition history
- Model inference performance optimization
- Emotion analytics dashboard
- Improved face detection and visualization

---

## 👩‍💻 Author

Nikita Amelia Valencia

Machine Learning & Data Science Enthusiast

---

## 📜 License

This project is intended for educational and portfolio purposes.
