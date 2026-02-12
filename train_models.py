"""
Heart Disease Classification - Model Training Script
BITS Pilani ML Assignment 2
Student: SAIRAM GAJABINKAR (2025AA05138)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='heart.csv'):
    """Load and preprocess the heart disease dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Instances: {df.shape[0]}")

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save test data for Streamlit app
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv('test_data.csv', index=False)

    print(f"Train set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print("-" * 80)

    return X_train_scaled, X_test_scaled, y_train, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    # AUC requires probability predictions
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['AUC'] = 0.0

    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and calculate metrics"""

    # Define all models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

    results = {}

    print("\nTraining and Evaluating Models...")
    print("=" * 80)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Get probability predictions for AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # Store results
        results[name] = metrics

        # Save the model
        model_filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        # Print metrics
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  AUC:       {metrics['AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1 Score:  {metrics['F1']:.4f}")
        print(f"  MCC:       {metrics['MCC']:.4f}")
        print(f"  Model saved: {model_filename}")

    return results

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)

    # Create DataFrame for better visualization
    df_results = pd.DataFrame(results).T
    df_results = df_results[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

    print(df_results.to_string())

    # Save results to CSV for README
    df_results.to_csv('model_results.csv')
    print("\nResults saved to 'model_results.csv'")
    print("=" * 80)

def main():
    """Main execution function"""
    print("=" * 80)
    print("HEART DISEASE CLASSIFICATION - MODEL TRAINING")
    print("BITS Pilani ML Assignment 2")
    print("Student: SAIRAM GAJABINKAR (2025AA05138)")
    print("=" * 80)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and evaluate all models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Print results table
    print_results_table(results)

    print("\n✓ All models trained and saved successfully!")
    print("✓ Test data saved as 'test_data.csv' for Streamlit app")
    print("✓ Scaler saved as 'model/scaler.pkl'")

if __name__ == "__main__":
    main()
