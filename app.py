"""
Heart Disease Classification - Streamlit Web Application
BITS Pilani ML Assignment 2
Student: SAIRAM GAJABINKAR (2025AA05138)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">❤️ Heart Disease Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Model ML Classifier with Interactive Evaluation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BITS Pilani M.Tech (AIML) - Machine Learning Assignment-2</p>', unsafe_allow_html=True)

st.markdown("""
**Name: SAIRAM GAJABINKAR**\n
**ID: 2025AA05138**
""")

st.divider()

# Model information
MODEL_INFO = {
    'Logistic Regression': {
        'file': 'model/logistic_regression.pkl',
        'description': 'Linear model for binary classification using logistic function'
    },
    'Decision Tree': {
        'file': 'model/decision_tree.pkl',
        'description': 'Tree-based model that splits data based on feature values'
    },
    'kNN': {
        'file': 'model/knn.pkl',
        'description': 'Instance-based learning using k-nearest neighbors'
    },
    'Naive Bayes': {
        'file': 'model/naive_bayes.pkl',
        'description': 'Probabilistic classifier based on Bayes theorem'
    },
    'Random Forest': {
        'file': 'model/random_forest.pkl',
        'description': 'Ensemble of decision trees for robust predictions'
    },
    'XGBoost': {
        'file': 'model/xgboost.pkl',
        'description': 'Gradient boosting ensemble for high performance'
    }
}

@st.cache_resource
def load_model(model_path):
    """Load a trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['AUC'] = 0.0

    return metrics

def plot_confusion_matrix(cm, title):
    """Plot confusion matrix using seaborn heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig

# Sidebar for model selection
st.sidebar.header("🔧 Configuration")
st.sidebar.markdown("### Select ML Model")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_INFO.keys()),
    help="Select one of the 6 trained models for prediction"
)

# Display model information
st.sidebar.info(f"**Model:** {selected_model}\n\n{MODEL_INFO[selected_model]['description']}")

# Main content area
st.header("📤 Upload Test Data")
st.markdown("Upload a CSV file containing heart disease features for prediction.")

# Feature 1: Dataset upload option
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload test data with the same features as training data"
)

if uploaded_file is not None:
    try:
        # Load the uploaded data
        df = pd.read_csv(uploaded_file)

        st.success(f"✓ File uploaded successfully! Shape: {df.shape}")

        # Display data preview
        with st.expander("📊 View Data Preview"):
            st.dataframe(df.head(10))
            st.write(f"Total samples: {len(df)}")

        # Check if target column exists
        if 'target' in df.columns:
            X_test = df.drop('target', axis=1)
            y_test = df['target']
            has_labels = True
        else:
            X_test = df
            y_test = None
            has_labels = False
            st.warning("No 'target' column found. Predictions will be made without evaluation metrics.")

        # Load model and scaler
        model = load_model(MODEL_INFO[selected_model]['file'])
        scaler = load_scaler()

        if model is not None:
            # Make predictions
            # If scaler exists, use it to transform data
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                # If no scaler, use raw data (fallback for testing)
                st.warning("⚠️ Scaler not found. Using raw data. Run train_models.py to generate scaler.")
                X_test_scaled = X_test.values

            y_pred = model.predict(X_test_scaled)

            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                y_pred_proba = None

            st.divider()

            # Feature 3: Display evaluation metrics
            if has_labels:
                st.header("📊 Model Performance Metrics")

                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("Precision", f"{metrics['Precision']:.4f}")

                with col2:
                    st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                    st.metric("Recall", f"{metrics['Recall']:.4f}")

                with col3:
                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    st.metric("MCC Score", f"{metrics['MCC']:.4f}")

                st.divider()

                # Feature 4: Display confusion matrix and classification report
                st.header("📈 Detailed Evaluation")

                tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])

                with tab1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model}")
                    st.pyplot(fig)

                    # Display confusion matrix values
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**True Negatives:**", cm[0, 0])
                        st.write("**False Positives:**", cm[0, 1])
                    with col2:
                        st.write("**False Negatives:**", cm[1, 0])
                        st.write("**True Positives:**", cm[1, 1])

                with tab2:
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred,
                                                   target_names=['No Disease', 'Disease'],
                                                   output_dict=True)

                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))

                st.divider()

                # Feature 5: Download Predictions Report
                st.header("📥 Download Predictions Report")

                # Create detailed predictions DataFrame
                predictions_report_df = X_test.copy()
                predictions_report_df['Actual'] = ['Disease' if val == 1 else 'No Disease' for val in y_test]
                predictions_report_df['Predicted'] = ['Disease' if val == 1 else 'No Disease' for val in y_pred]
                predictions_report_df['Correct'] = ['✓' if actual == pred else '✗'
                                                     for actual, pred in zip(y_test, y_pred)]

                # Add probability scores if available
                if y_pred_proba is not None:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                        predictions_report_df['Probability_No_Disease'] = y_pred_proba[:, 0]
                        predictions_report_df['Probability_Disease'] = y_pred_proba[:, 1]
                    else:
                        predictions_report_df['Probability_Disease'] = y_pred_proba

                # Show preview
                st.subheader("Preview of Predictions Report")
                st.dataframe(predictions_report_df.head(10))

                # Download button for predictions
                csv_predictions = predictions_report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Predictions Report (CSV)",
                    data=csv_predictions,
                    file_name=f"predictions_report_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    help="Download complete predictions with actual vs predicted values"
                )

                # Download button for classification report
                csv_classification = report_df.to_csv()
                st.download_button(
                    label="📥 Download Classification Report (CSV)",
                    data=csv_classification,
                    file_name=f"classification_report_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    help="Download classification metrics report"
                )

            else:
                st.header("🔮 Predictions")
                st.write(f"Predictions made using **{selected_model}**")

                # Create predictions DataFrame with features
                predictions_df = X_test.copy()
                predictions_df['Prediction'] = ['Disease' if p == 1 else 'No Disease' for p in y_pred]
                predictions_df['Prediction_Code'] = y_pred

                # Add probability scores if available
                if y_pred_proba is not None:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                        predictions_df['Probability_No_Disease'] = y_pred_proba[:, 0]
                        predictions_df['Probability_Disease'] = y_pred_proba[:, 1]
                    else:
                        predictions_df['Probability_Disease'] = y_pred_proba

                st.dataframe(predictions_df)

                # Summary statistics
                disease_count = sum(y_pred == 1)
                no_disease_count = sum(y_pred == 0)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Disease", disease_count)
                with col2:
                    st.metric("Predicted No Disease", no_disease_count)

                st.divider()

                # Download predictions
                st.subheader("📥 Download Predictions")
                csv_data = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions Report (CSV)",
                    data=csv_data,
                    file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    help="Download predictions with all features and probability scores"
                )

        else:
            st.error("Failed to load model. Please check if model files exist in the model/ directory.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please ensure your CSV has the correct format with the same features as training data.")

else:
    st.info("👆 Please upload a CSV file to begin prediction and evaluation.")

    # Show expected features
    with st.expander("ℹ️ Expected Features"):
        st.markdown("""
        The uploaded CSV should contain the following 13 features:

        1. **age** - Age in years
        2. **sex** - Sex (1 = male; 0 = female)
        3. **cp** - Chest pain type (0-3)
        4. **trestbps** - Resting blood pressure (mm Hg)
        5. **chol** - Serum cholesterol (mg/dl)
        6. **fbs** - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        7. **restecg** - Resting ECG results (0-2)
        8. **thalach** - Maximum heart rate achieved
        9. **exang** - Exercise induced angina (1 = yes; 0 = no)
        10. **oldpeak** - ST depression induced by exercise
        11. **slope** - Slope of peak exercise ST segment (0-2)
        12. **ca** - Number of major vessels colored by fluoroscopy (0-3)
        13. **thal** - Thalassemia (0-3)
        14. **target** - Target variable (0 = no disease; 1 = disease) [Optional for predictions]
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Heart Disease Classification System</strong></p>
    <p>BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2</p>
    <p>© 2026 SAIRAM GAJABINKAR (2025AA05138)</p>
</div>
""", unsafe_allow_html=True)
