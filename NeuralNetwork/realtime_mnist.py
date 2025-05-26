import cv2
import numpy as np
import tensorflow as tf
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = 'mnist_model.h5'

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info('Loaded model successfully.')
except Exception as e:
    logger.error(f'Could not load model: {e}')
    sys.exit(1)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype('float32') / 255.0
    # Reshape for CNN input (add channel dimension)
    processed_image = norm.reshape(1, 28, 28, 1)
    # Save the preprocessed image for debugging
    cv2.imwrite('debug_preprocessed.png', (norm*255).astype(np.uint8))
    logger.info('Saved preprocessed 28x28 image as debug_preprocessed.png')
    return processed_image

def draw_prediction_box(img, pred):
    box_h = 60
    box_w = 180
    x, y = 10, 10
    cv2.rectangle(img, (x, y), (x+box_w, y+box_h), (0,0,0), thickness=-1)
    cv2.putText(img, f'Predicted: {pred}', (x+10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img

def main():
    if len(sys.argv) < 2:
        logger.error('Usage: python realtime_mnist.py <image_path>')
        sys.exit(1)
    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f'Could not read image: {image_path}')
        sys.exit(1)
    input_data = preprocess(frame)
    preds = model.predict(input_data, verbose=0)
    pred_digit = np.argmax(preds)
    display = draw_prediction_box(frame.copy(), pred_digit)
    # Show the preprocessed image as well
    pre_img = cv2.imread('debug_preprocessed.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Model Input (28x28)', cv2.resize(pre_img, (140,140), interpolation=cv2.INTER_NEAREST))
    cv2.imshow('MNIST Image Prediction', display)
    logger.info(f'Predicted digit: {pred_digit}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

