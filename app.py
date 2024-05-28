import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('./100-fix.pt')

# Streamlit UI
st.title("YOLO Model Prediction")
st.write("Upload an image and set the parameters for prediction")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Set prediction parameters
    conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1)
    line_width = st.slider("Line Width", min_value=1, max_value=10, value=1)
    
    # Fixed image size for prediction
    fixed_imgsz = (640, 640)
    
    # Perform prediction
    if st.button("Predict"):
        result = model.predict(image, save=False, imgsz=fixed_imgsz, conf=conf, line_width=line_width)

        # Draw the bounding boxes on the image
        for bbox in result[0].boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), line_width)
            label = f"{bbox.conf[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with bounding boxes
        st.image(image, caption='Predicted Image with Bounding Boxes', use_column_width=True)
