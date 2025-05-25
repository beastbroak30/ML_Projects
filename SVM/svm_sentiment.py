import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import nltk
import string
from svm_eval_viz import plot_decision_boundary, evaluate_model
import joblib
import logging

# Download NLTK punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info('Starting SVM sentiment training pipeline.')
logging.info('Loading dataset...')
df = pd.read_csv('datasets/Sentiment dataset.csv')
logging.info('Preprocessing text and filtering labels...')
df = df.dropna(subset=['Text', 'Sentiment'])
df = df[df['Sentiment'].str.strip().isin(['Positive', 'Negative'])]
df['Sentiment'] = df['Sentiment'].str.strip().map({'Negative': 0, 'Positive': 1})
df['Text'] = df['Text'].astype(str).apply(preprocess_text)
logging.info('Vectorizing text with TF-IDF...')
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Text']).toarray()
y = df['Sentiment'].values
logging.info('Splitting data into train and test sets...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
logging.info('Applying PCA for visualization...')
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
logging.info('Training SVM models...')
svm_linear = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
logging.info('Training SVM models for visualization...')
svm_linear_vis = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_rbf_vis = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_linear_vis.fit(X_train_pca, y_train)
svm_rbf_vis.fit(X_train_pca, y_train)
logging.info('Plotting decision boundaries...')
plot_decision_boundary(svm_linear_vis, X_train_pca, y_train, 'SVM Decision Boundary with Linear Kernel', 'sentiment_linear_boundary.png')
plot_decision_boundary(svm_rbf_vis, X_train_pca, y_train, 'SVM Decision Boundary with RBF Kernel', 'sentiment_rbf_boundary.png')
logging.info('Evaluating models...')
print("\nEvaluation Results:")
evaluate_model(svm_linear, X_test, y_test, "Linear")
evaluate_model(svm_rbf, X_test, y_test, "RBF")
print("\nDecision boundary plots saved as 'sentiment_linear_boundary.png' and 'sentiment_rbf_boundary.png'.")
logging.info('Saving models and vectorizer...')
joblib.dump(svm_linear, 'svm_linear_model.joblib')
joblib.dump(svm_rbf, 'svm_rbf_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(pca, 'pca_transformer.joblib')
logging.info('Pipeline completed successfully.')