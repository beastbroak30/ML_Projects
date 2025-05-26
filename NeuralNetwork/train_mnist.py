import os
import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'train-images.idx3-ubyte')
TRAIN_LABELS = os.path.join(DATA_DIR, 'train-labels.idx1-ubyte')
MODEL_PATH = 'mnist_model.h5'
TFLITE_PATH = 'mnist_model.tflite'
HISTORY_PATH = 'training_history.pkl'

# Data loading function
def load_mnist(images_path, labels_path):
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing MNIST data files: {images_path} or {labels_path}")
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)
    return images, labels

# Preprocessing function
def preprocess_data(images, labels):
    # Reshape images to be 28x28x1 for CNN input
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = images.reshape((images.shape[0], 28, 28, 1))
    labels = to_categorical(labels, 10)
    return images, labels

# Model creation function
def create_model():
    model = keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training routine
def main():
    try:
        logger.info('Loading MNIST training data...')
        x_train, y_train = load_mnist(TRAIN_IMAGES, TRAIN_LABELS)
        x_train, y_train = preprocess_data(x_train, y_train)
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        return

    # Load or create model
    if os.path.exists(MODEL_PATH):
        logger.info('Loading existing model...')
        model = load_model(MODEL_PATH)
    else:
        logger.info('Creating new model...')
        model = create_model()

    # Train
    logger.info('Training model...')
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Save model
    logger.info('Saving model...')
    model.save(MODEL_PATH)

    # Convert to TFLite
    logger.info('Converting to TensorFlow Lite...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    # Save training history
    logger.info('Saving training history...')
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    logger.info('Training complete. Model and history saved.')

if __name__ == '__main__':
    main()

