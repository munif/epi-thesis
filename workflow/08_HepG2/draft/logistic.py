import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_array_features(df, array_columns):
    """
    Preprocess array-type columns by flattening and concatenating them.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    array_columns (list): List of column names containing arrays
    
    Returns:
    np.array: Processed feature matrix
    """
    processed_arrays = []
    
    for col in array_columns:
        # Convert string representations of arrays to numpy arrays if needed
        if df[col].dtype == 'object':
            arrays = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
        else:
            arrays = df[col]
        
        # Stack arrays into a 2D matrix
        stacked = np.vstack(arrays)
        processed_arrays.append(stacked)
    
    # Concatenate all processed arrays horizontally
    X = np.hstack(processed_arrays)
    
    return X

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
    """
    Plot ROC curve and calculate AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

def create_and_train_model(X, y, random_state=42):
    """
    Create, train and evaluate a logistic regression model with train/validation/test splits.
    
    Parameters:
    X (np.array): Feature matrix
    y (np.array): Target variable
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: Trained model and metrics for both validation and test sets
    """
    # First split: separate test set (1/3 of data)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=1/3, random_state=random_state
    )
    
    # Second split: divide remaining data into train and validation (50% each)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    # Create and train the model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    val_metrics = classification_report(y_val, val_pred)
    test_metrics = classification_report(y_test, test_pred)
    
    # Plot confusion matrices
    print("\nValidation Set Confusion Matrix:")
    plot_confusion_matrix(y_val, val_pred, title='Validation Set Confusion Matrix')
    
    print("\nTest Set Confusion Matrix:")
    plot_confusion_matrix(y_test, test_pred, title='Test Set Confusion Matrix')
    
    # Plot ROC curves
    print("\nValidation Set ROC Curve:")
    val_auc = plot_roc_curve(y_val, val_prob, title='Validation Set ROC Curve')
    
    print("\nTest Set ROC Curve:")
    test_auc = plot_roc_curve(y_test, test_prob, title='Test Set ROC Curve')
    
    return model, {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'validation_auc': val_auc,
        'test_auc': test_auc,
        'splits': {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    }

# Example usage
def main():
    # Sample data structure (replace with your actual data)
    data = {
        'array_col1': [np.random.rand(5) for _ in range(100)],
        'array_col2': [np.random.rand(3) for _ in range(100)],
        'array_col3': [np.random.rand(4) for _ in range(100)],
        'array_col4': [np.random.rand(6) for _ in range(100)],
        'array_col5': [np.random.rand(2) for _ in range(100)],
        'target': np.random.randint(0, 2, 100)  # Binary target variable
    }
    df = pd.DataFrame(data)
    
    # Specify array columns
    array_columns = ['array_col1', 'array_col2', 'array_col3', 'array_col4', 'array_col5']
    
    # Preprocess features
    X = preprocess_array_features(df, array_columns)
    y = df['target'].values
    
    # Train and evaluate model
    model, results = create_and_train_model(X, y)
    
    # Print metrics
    print("\nValidation Set Metrics:")
    print(results['validation_metrics'])
    print("\nTest Set Metrics:")
    print(results['test_metrics'])
    
    # Print AUC scores
    print(f"\nValidation Set AUC: {results['validation_auc']:.3f}")
    print(f"Test Set AUC: {results['test_auc']:.3f}")
    
    return model, results