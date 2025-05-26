import os
import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
TEST_IMAGES = os.path.join(DATA_DIR, 't10k-images.idx3-ubyte')
TEST_LABELS = os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte')
MODEL_PATH = 'mnist_model.h5'
HISTORY_PATH = 'training_history.pkl'
PLOT_PATH = 'training_history.png'

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

def plot_history(history, plot_path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        logger.error(f'Model file {MODEL_PATH} not found. Please train the model first.')
        return
    logger.info('Loading trained model...')
    model = load_model(MODEL_PATH)

    # Load test data
    try:
        logger.info('Loading MNIST test data...')
        x_test, y_test = load_mnist(TEST_IMAGES, TEST_LABELS)
        x_test, y_test = preprocess_data(x_test, y_test)
    except Exception as e:
        logger.error(f'Error loading test data: {e}')
        return

    # Evaluate
    logger.info('Evaluating model...')
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test accuracy: {acc*100:.2f}%')

    # Load and plot training history
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        plot_history(history, PLOT_PATH)
        logger.info(f'Training history plot saved as {PLOT_PATH}')
    else:
        logger.warning('Training history not found.')

if __name__ == '__main__':
    main()

