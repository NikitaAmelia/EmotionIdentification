
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="centered",
)

st.title("😊 Facial Emotion Detection")
st.write(
    "Use your device's camera to capture an image. The application will "
    "detect faces and estimate the emotions expressed."
)
st.info(
    "Note: The browser camera captures a single image each time you take a photo. "
    "This application does not continuously analyze a real-time video stream."
)

@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load the Haar Cascade model.")
    return cascade

camera_image = st.camera_input("Take a photo using your camera")

if camera_image is not None:
    pil_image = Image.open(camera_image).convert("RGB")
    rgb_image = np.array(pil_image)

    st.image(
        rgb_image,
        caption="Captured Image",
        use_container_width=True,
    )

    if st.button("Analyze Emotions", type="primary", use_container_width=True):
        with st.spinner("Detecting faces and analyzing emotions..."):
            try:
                cascade = load_face_cascade()
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )

                if len(faces) == 0:
                    st.warning(
                        "No face was detected. Please try taking another photo "
                        "with your face clearly visible and sufficient lighting."
                    )
                else:
                    # OpenCV uses BGR, which is accepted by DeepFace.
                    annotated_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    results = []

                    for (x, y, w, h) in faces:
                        face_roi = annotated_bgr[y:y + h, x:x + w]

                        try:
                            analysis = DeepFace.analyze(
                                img_path=face_roi,
                                actions=["emotion"],
                                enforce_detection=False,
                                silent=True,
                            )

                            if isinstance(analysis, list):
                                analysis = analysis[0]

                            emotion = analysis.get(
                                "dominant_emotion",
                                "Unknown",
                            )
                            scores = analysis.get("emotion", {})
                            results.append((emotion, scores))

                            cv2.rectangle(
                                annotated_bgr,
                                (x, y),
                                (x + w, y + h),
                                (0, 0, 255),
                                2,
                            )

                            cv2.putText(
                                annotated_bgr,
                                emotion,
                                (x, max(y - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )

                        except Exception as exc:
                            st.warning(
                                f"Failed to analyze one of the detected faces: {exc}"
                            )

                    if results:
                        annotated_rgb = cv2.cvtColor(
                            annotated_bgr,
                            cv2.COLOR_BGR2RGB,
                        )

                        st.subheader("Detection Results")
                        st.image(
                            annotated_rgb,
                            caption="Detected Faces and Emotion Labels",
                            use_container_width=True,
                        )

                        for index, (emotion, scores) in enumerate(
                            results,
                            start=1,
                        ):
                            st.markdown(
                                f"**Face {index}: {emotion.capitalize()}**"
                            )

                            if scores:
                                sorted_scores = sorted(
                                    scores.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )

                                st.write(
                                    "Emotion scores are model estimates and "
                                    "should not be considered a diagnosis."
                                )

                                st.dataframe(
                                    {
                                        "Emotion": [
                                            name.capitalize()
                                            for name, _ in sorted_scores
                                        ],
                                        "Score (%)": [
                                            round(float(score), 2)
                                            for _, score in sorted_scores
                                        ],
                                    },
                                    hide_index=True,
                                    use_container_width=True,
                                )

            except Exception as exc:
                st.error(
                    "The application failed to process the image. "
                    "Please check the application logs for technical details."
                )
                st.exception(exc)

st.caption(
    "AI-based emotion analysis may be inaccurate and does not necessarily "
    "reflect a person's actual feelings."
)
