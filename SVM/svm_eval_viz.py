import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title, filename):
    logging.info(f'Plotting decision boundary: {title} -> {filename}')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Use the model to predict on the meshgrid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    plt.figure(figsize=(10, 8)) # Increased figure size for better visibility
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot also the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30) # Increased marker size
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title, fontsize=16) # Added title with increased font size
    plt.xlabel('Principal Component 1', fontsize=12) # Added x-label with increased font size
    plt.ylabel('Principal Component 2', fontsize=12) # Added y-label with increased font size
    plt.legend(*scatter.legend_elements(), title="Classes") # Added a legend
    plt.grid(True) # Added grid for better readability
    plt.savefig(filename)
    plt.close()
    logging.info(f'Decision boundary plot saved: {filename}')

def evaluate_model(model, X_test, y_test, kernel_name):
    logging.info(f'Evaluating model: {kernel_name}')
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For SVMs without probability, use decision_function
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{kernel_name} Kernel:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  AUC:       {auc:.4f}")
    logging.info(f"{kernel_name} Kernel - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")
    return acc, prec, rec, auc