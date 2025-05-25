import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

results_dir = "results_png"
model_filename = 'random_forest_diabetes_model.joblib'
data_path = 'datasets\\merged_diabetes_dataset.csv'

try:
    model = joblib.load(model_filename)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error(f"Model file not found. Make sure you have run train_model.py first.")
    exit()

try:
    df = pd.read_csv(data_path)
    logger.info("Dataset loaded successfully.")
except FileNotFoundError:
    logger.error(f"Dataset not found at {data_path}")
    exit()

# Preprocessing 
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
X_test = df.drop('diabetes', axis=1)
y_test = df['diabetes']

y_pred = model.predict(X_test)

logger.info("Model Evaluation:")
cm = confusion_matrix(y_test, y_pred)
logger.info(f"Confusion Matrix:\n{cm}")

class_report = classification_report(y_test, y_pred)
logger.info(f"Classification Report:\n{class_report}")

accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Accuracy: {accuracy:.4f}")

f1 = f1_score(y_test, y_pred)
logger.info(f"F1 Score: {f1:.4f}")

logger.info("Feature Importance:")
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    logger.info(f"\n{feature_importances_sorted}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_sorted.index, y=feature_importances_sorted.values)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance1.png'))
    plt.close()
else:
    logger.warning("Model does not have feature_importances_ attribute.")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix1.png'))
plt.close()

logger.info(f"Visualizations saved to the '{results_dir}' directory.")