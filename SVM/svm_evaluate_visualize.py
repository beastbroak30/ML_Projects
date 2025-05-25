import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from svm_eval_viz import plot_decision_boundary, evaluate_model
import nltk
import string
import joblib
import logging
import os

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
logging.info('Starting SVM sentiment evaluation and visualization pipeline.')

# Define results directory
results_dir = 'Results_png'

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    logging.info(f'Created results directory: {results_dir}')

logging.info('Loading dataset...')
df = pd.read_csv('datasets/Sentiment dataset.csv')
logging.info('Preprocessing text and filtering labels...')
df = df.dropna(subset=['Text', 'Sentiment'])
df = df[df['Sentiment'].str.strip().isin(['Positive', 'Negative'])]
df['Sentiment'] = df['Sentiment'].str.strip().map({'Negative': 0, 'Positive': 1})
df['Text'] = df['Text'].astype(str).apply(preprocess_text)

logging.info('Loading models and vectorizer...')
svm_linear = joblib.load('svm_linear_model.joblib')
svm_rbf = joblib.load('svm_rbf_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
pca = joblib.load('pca_transformer.joblib')

logging.info('Transforming data with loaded vectorizer and PCA...')
X = vectorizer.transform(df['Text']).toarray()
y = df['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

logging.info('Training SVM models for visualization...')
svm_linear_vis = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_rbf_vis = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_linear_vis.fit(X_train_pca, y_train)
svm_rbf_vis.fit(X_train_pca, y_train)

logging.info('Plotting decision boundaries...')
print("\n--- Decision Boundary Visualization (Train Set, PCA) ---")
plot_decision_boundary(svm_linear_vis, X_train_pca, y_train, 'SVM Decision Boundary with Linear Kernel', os.path.join(results_dir, 'sentiment_linear_boundary.png'))
plot_decision_boundary(svm_rbf_vis, X_train_pca, y_train, 'SVM Decision Boundary with RBF Kernel', os.path.join(results_dir, 'sentiment_rbf_boundary.png'))
print(f"Plots saved to '{results_dir}'.")

logging.info('Evaluating models...')
print("\n--- Evaluation on Test Set (Full TF-IDF Features) ---")
evaluate_model(svm_linear, X_test, y_test, "Linear")

# Capture RBF evaluation results
print("--- Evaluation on Test Set (Full TF-IDF Features) ---")
rbf_metrics = evaluate_model(svm_rbf, X_test, y_test, "RBF")

# Save RBF evaluation results to a text file
rbf_results_file = os.path.join(results_dir, 'rbf_evaluation_results.txt')
with open(rbf_results_file, 'w') as f:
    f.write("RBF Kernel Evaluation Results (Full TF-IDF Features):\n")
    f.write(f"  Accuracy:  {rbf_metrics[0]:.4f}\n")
    f.write(f"  Precision: {rbf_metrics[1]:.4f}\n")
    f.write(f"  Recall:    {rbf_metrics[2]:.4f}\n")
    f.write(f"  AUC:       {rbf_metrics[3]:.4f}\n")
logging.info(f'RBF evaluation results saved to {rbf_results_file}')

print("\n--- Evaluation on Test Set (PCA-Reduced Features, for visualization) ---")
evaluate_model(svm_linear_vis, X_test_pca, y_test, "Linear (PCA)")
evaluate_model(svm_rbf_vis, X_test_pca, y_test, "RBF (PCA)")

logging.info('Pipeline completed successfully.')