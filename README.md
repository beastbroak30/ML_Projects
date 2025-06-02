# Machine Learning Starter Projects: MNIST, Diabetes Prediction & Sentiment Analysis

Welcome! This repository is designed as a launchpad for aspiring Machine Learning engineers and data scientists. If you're looking to bridge the gap between theory and practice, you've come to the right place. Here, you'll find three distinct, hands-on projects covering fundamental ML concepts and applications:

1.  **Neural Network for MNIST Digit Recognition:** Dive into the world of computer vision and deep learning by building a model to recognize handwritten digits.
2.  **Random Forest for Diabetes Prediction:** Tackle a real-world classification problem using health data and a powerful ensemble method.
3.  **Support Vector Machine (SVM) for Sentiment Analysis:** Explore Natural Language Processing (NLP) by classifying text sentiment.

Each project is self-contained within its directory, complete with datasets (where applicable), Python scripts for training and evaluation, and saved models/results. This structure allows you to focus on one concept at a time or explore them in any order.

My goal is to provide clear, practical examples that not only demonstrate *how* to implement these algorithms but also *why* they work and *where* they are applied.

Let's start building!

---

## Projects Overview

Below is a summary of each project included in this repository. Navigate to the respective directories for detailed code and data.



### 1. Neural Network (MNIST Digit Recognition)

*   **Goal:** Classify handwritten digits (0-9) using a Neural Network (likely a CNN).
*   **Directory:** `NeuralNetwork/`
*   **Highlights:** Covers fundamental image loading (MNIST dataset included), model training (`train_mnist.py`), evaluation (`test_mnist.py`), and even real-time prediction examples (`realtime_mnist.py`).
*   **Concepts:** Computer Vision, Convolutional Neural Networks (CNNs), TensorFlow/Keras, Model Training & Evaluation.

### 2. Random Forest (Diabetes Prediction)

*   **Goal:** Predict diabetes risk based on health indicators using a Random Forest classifier.
*   **Directory:** `RandomClassifier/`
*   **Highlights:** Works with tabular health data (datasets included), trains a robust ensemble model (`train_model.py`), evaluates performance (`evaluate_model.py`), and visualizes results like feature importance (`results_png/`).
*   **Concepts:** Classification, Ensemble Methods, Random Forest, Feature Importance, Data Preprocessing, Scikit-learn.

### 3. Support Vector Machine (SVM) for Sentiment Analysis

*   **Goal:** Classify text sentiment (e.g., positive/negative) using SVM with different kernels.
*   **Directory:** `SVM/`
*   **Highlights:** Introduces basic Natural Language Processing (NLP). Uses TF-IDF for text vectorization (`tfidf_vectorizer.joblib`), trains Linear and RBF SVM models (`svm_sentiment.py`), evaluates them (`svm_evaluate_visualize.py`), and visualizes decision boundaries (`Results_png/`). PCA might be used for dimensionality reduction (`pca_transformer.joblib`).
*   **Concepts:** NLP, Sentiment Analysis, SVM (Linear & RBF Kernels), TF-IDF, Text Classification, Scikit-learn, Dimensionality Reduction (PCA).

---

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/beastbroak30/ML_Projects.git
    cd ML_Projects
    ```
2.  **Navigate to a project directory:**
    ```bash
    cd NeuralNetwork  # or RandomClassifier, or SVM
    ```
3.  **Install Dependencies:**
    *   For `NeuralNetwork` and `SVM`, install requirements:
        ```bash
        pip install -r requirements.txt
        ```
    *   For `RandomClassifier`, ensure you have libraries like `pandas`, `scikit-learn`, and `joblib` installed. You might need to create a `requirements.txt` or install them manually:
        ```bash
        pip install pandas scikit-learn joblib matplotlib seaborn
        ```
    *   *(Note: It's recommended to use a virtual environment for each project to manage dependencies.)*
4.  **Run the Scripts:**
    *   Explore the Python scripts (`.py` files) within each project directory.
    *   Typically, you'll run a training script first (e.g., `train_model.py`, `train_mnist.py`, `svm_sentiment.py`) followed by evaluation or prediction scripts.

---

## Learning Resources

This repository provides practical code, but understanding the underlying concepts is crucial! Here are some excellent external resources to deepen your knowledge:

**General Machine Learning:**

*   **Google AI Education:** [https://ai.google/education/](https://ai.google/education/)
*   **Kaggle Learn Courses:** [https://www.kaggle.com/learn](https://www.kaggle.com/learn)
*   **StatQuest with Josh Starmer (YouTube):** [https://www.youtube.com/c/StatQuestwithJoshStarmer](https://www.youtube.com/c/StatQuestwithJoshStarmer) (Highly recommended for intuitive explanations!)

**Project-Specific Topics:**

*   **Neural Networks & CNNs (Project 1):**
    *   *GeeksforGeeks:* [Introduction to Convolutional Neural Networks](https://www.geeksforgeeks.org/introduction-convolutional-neural-network/)
    *   *YouTube (3Blue1Brown):* [What is a Convolutional Neural Network?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
    *   *TensorFlow Tutorials:* [Basic image classification](https://www.tensorflow.org/tutorials/images/classification)
*   **Random Forests (Project 2):**
    *   *GeeksforGeeks:* [Random Forest Algorithm](https://www.geeksforgeeks.org/random-forest-algorithm/)
    *   *YouTube (StatQuest):* [Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) & [Part 2](https://www.youtube.com/watch?v=sQ870aTKqiM)
    *   *Scikit-learn Docs:* [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
*   **SVM & Sentiment Analysis (Project 3):**
    *   *GeeksforGeeks:* [Support Vector Machine (SVM) Algorithm](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)
    *   *GeeksforGeeks:* [Sentiment Analysis](https://www.geeksforgeeks.org/sentiment-analysis-in-python/)
    *   *YouTube (StatQuest):* [Support Vector Machines](https://www.youtube.com/watch?v=efR1C6CvhmE)
    *   *Scikit-learn Docs:* [SVM](https://scikit-learn.org/stable/modules/svm.html), [TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

---

Feel free to explore, experiment, and adapt the code. Happy learning!

