import cv2
import numpy as np
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized_frame = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors (MNIST is white digit on black background)
    inverted_frame = cv2.bitwise_not(resized_frame)
    # Normalize
    normalized_frame = inverted_frame.astype('float32') / 255.0
    # Reshape for model input (add batch and channel dimensions)
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=-1)
    return preprocessed_frame

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    preprocessed_input = preprocess_frame(frame)

    # Make prediction
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the prediction on the frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time MNIST Prediction', display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()