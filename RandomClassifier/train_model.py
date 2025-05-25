import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
# Load the dataset

data_path = 'datasets\\training_diabetes_dataset.csv' 
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}")
    exit()

# Preprocessing
# Ensure all categories and columns are handled consistently
categorical_columns = ['gender', 'smoking_history']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str)
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning using GridSearchCV
# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a Random Forest Classifier model
rf = RandomForestClassifier(random_state=42)

# Define filename for best hyperparameters
best_params_filename = 'best_params.joblib'

# Check if best hyperparameters file exists
if os.path.exists(best_params_filename):
    print(f"Loading best hyperparameters from {best_params_filename}...")
    best_params = joblib.load(best_params_filename)
    # Note: best_score is not loaded, as it's not strictly needed for training
    print("Best hyperparameters loaded successfully.")
else:
    print("Best hyperparameters file not found. Starting hyperparameter tuning...")
    # Create GridSearchCV object
    # Using a smaller cv value (e.g., 3) for faster training
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print("Hyperparameter tuning finished.")

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    # Save the best parameters
    joblib.dump(best_params, best_params_filename)
    print(f"Best hyperparameters saved to {best_params_filename}")

# Train the final model with the best parameters
print("Training final model with best parameters...")
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
print("Final model training finished.")

# Evaluate the final model on the test set (optional, but good practice)
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy with best model: {accuracy:.4f}")

# Save the trained model and test data
model_filename = 'random_forest_diabetes_model.joblib'
test_data_filename = 'diabetes_test_data.joblib'

joblib.dump(final_model, model_filename)
joblib.dump({'X_test': X_test, 'y_test': y_test}, test_data_filename)

print(f"Trained model saved to {model_filename}")
print(f"Test data saved to {test_data_filename}")