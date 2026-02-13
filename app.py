"""
Heart Disease Classification - Streamlit Web Application
BITS Pilani ML Assignment 2
Student: SAIRAM GAJABINKAR (2025AA05138)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #ff4b4b 0%, #ff8080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .student-info {
        text-align: center;
        color: #333;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 8px;
    }

    /* Gradient cards common style */
    .upload-gradient-bg, .model-gradient-bg, .performance-gradient-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem 0.5rem 2rem;
        border-radius: 12px 12px 0 0;
        margin: 1rem 0 0 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .performance-gradient-bg {
        margin: 1.5rem 0 0 0;
    }
    .upload-gradient-bg h2, .model-gradient-bg h2, .performance-gradient-bg h2 {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0 0 0.8rem 0;
    }
    .model-gradient-bg h2 {
        margin: 0;
    }
    .performance-gradient-bg h2 {
        margin: 0 0 1rem 0;
    }
    .upload-gradient-bg p, .performance-gradient-bg h3 {
        color: white;
        font-size: 1.1rem;
        margin: 0;
        opacity: 0.95;
    }
    .performance-gradient-bg h3 {
        font-size: 1.2rem;
        font-weight: bold;
        opacity: 1;
    }

    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 0 2rem 2rem 2rem !important;
        margin: 0 0 1rem 0 !important;
        border-radius: 0 0 12px 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stFileUploader"] > div,
    div[data-testid="stFileUploader"] > div > div,
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInput"] {
        background-color: rgba(45, 55, 72, 0.9) !important;
        border: 2px dashed rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] div {
        color: white !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    /* Model selectbox styling - only for main content, not settings */
    div.stSelectbox:not(dialog div.stSelectbox):not([role="dialog"] div.stSelectbox) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 0 2rem 2rem 2rem !important;
        margin: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    div.stSelectbox:not(dialog div.stSelectbox):not([role="dialog"] div.stSelectbox) label {
        color: white !important;
        font-size: 1rem !important;
    }
    div.stSelectbox:not(dialog div.stSelectbox):not([role="dialog"] div.stSelectbox) > div {
        background-color: rgba(45, 55, 72, 0.9) !important;
        border-radius: 8px !important;
    }
    div.stSelectbox:not(dialog div.stSelectbox):not([role="dialog"] div.stSelectbox) input {
        color: white !important;
    }

    /* Model description */
    .model-description {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 2rem 2rem 2rem;
        border-radius: 0 0 12px 12px;
        margin: -2rem 0 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .model-description p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0;
        font-style: italic;
    }

    /* Metrics display */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 1rem 2rem 2rem 2rem !important;
        border-radius: 0 0 12px 12px !important;
        margin: 0 0 1rem 0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetricLabel"],
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stat-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .stat-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Ensure consistent metric card heights */
    [data-testid="stMetric"] {
        min-height: 100px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* Info boxes */
    .info-box, .success-box, .warning-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .info-box {
        background-color: #e3f2fd;
        border-color: #2196f3;
        color: #000;
    }
    .success-box {
        background-color: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
        font-weight: 500;
    }
    .warning-box {
        background-color: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
        font-weight: 500;
    }

    /* Feature list */
    .feature-item {
        background-color: rgba(102, 126, 234, 0.15);
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #667eea;
        font-size: 0.9rem;
    }

    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

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
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
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
        prob_col = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2 else y_pred_proba
        metrics['AUC'] = roc_auc_score(y_true, prob_col)
    else:
        metrics['AUC'] = 0.0

    return metrics

def plot_confusion_matrix_plotly(cm, title):
    """Create interactive confusion matrix using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Disease', 'Disease'],
        y=['No Disease', 'Disease'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500,
        font=dict(size=14)
    )

    return fig

def plot_metrics_bar_chart(metrics):
    """Create bar chart for metrics using Plotly"""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker=dict(
                color=metric_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=[f'{v:.4f}' for v in metric_values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1]),
        height=400,
        showlegend=False
    )

    return fig

def plot_roc_curve_plotly(y_true, y_pred_proba):
    """Create ROC curve using Plotly"""
    if y_pred_proba is None:
        return None

    prob_col = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2 else y_pred_proba
    fpr, tpr, _ = roc_curve(y_true, prob_col)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#667eea', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True,
        hovermode='closest'
    )

    return fig

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">❤️ Heart Disease Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Model ML Classifier with Interactive Evaluation</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. Upload CSV file with patient data
    2. Select a machine learning model
    3. View predictions and metrics
    4. Download reports for analysis
    """)

    st.divider()

    st.markdown("### 📊 Expected CSV Features")
    st.markdown("Your uploaded file should contain these 13 features:")

    features_info = {
        "age": "Age in years",
        "sex": "Gender (0=Female, 1=Male)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar",
        "restecg": "Resting ECG results",
        "thalach": "Max heart rate",
        "exang": "Exercise induced angina",
        "oldpeak": "ST depression",
        "slope": "Slope of peak exercise",
        "ca": "Major vessels (0-3)",
        "thal": "Thalassemia (0-3)"
    }

    with st.expander("📝 View Feature Details"):
        for feature, description in features_info.items():
            st.markdown(f'<div class="feature-item"><strong>{feature}:</strong> {description}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🤖 About Models")
    st.markdown("""
    This application uses **6 trained models**:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest
    - XGBoost

    All models are trained on heart disease dataset with 1,025 patient records.
    """)

# ==================== MAIN CONTENT ====================

# Upload Section
st.markdown("""
    <div class='upload-gradient-bg'>
        <h2>📤 Upload Patient Data</h2>
        <p>Choose a CSV file containing heart disease features</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=['csv'],
    label_visibility="collapsed",
    help="Upload test data with 13 features"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown('<div class="success-box">✅ <strong>File uploaded successfully!</strong></div>', unsafe_allow_html=True)

        # Display statistics in cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f'<div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">'
                f'<h3>{df.shape[0]}</h3>'
                f'<p>Total Samples</p></div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f'<div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">'
                f'<h3>{df.shape[1]}</h3>'
                f'<p>Features</p></div>',
                unsafe_allow_html=True
            )

        with col3:
            if 'target' in df.columns:
                disease_pct = (df['target'].sum() / len(df) * 100)
                st.markdown(
                    f'<div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">'
                    f'<h3>{disease_pct:.1f}%</h3>'
                    f'<p>Disease Cases</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="stat-card" style="background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);">'
                    f'<h3>N/A</h3>'
                    f'<p>No Labels</p></div>',
                    unsafe_allow_html=True
                )

        with col4:
            if 'target' in df.columns:
                no_disease_pct = ((len(df) - df['target'].sum()) / len(df) * 100)
                st.markdown(
                    f'<div class="stat-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">'
                    f'<h3>{no_disease_pct:.1f}%</h3>'
                    f'<p>Healthy Cases</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="stat-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">'
                    f'<h3>N/A</h3>'
                    f'<p>No Labels</p></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Data preview
        with st.expander("📊 View Data Preview (First 10 Rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        # Check for target column
        has_labels = 'target' in df.columns
        if has_labels:
            X_test = df.drop('target', axis=1)
            y_test = df['target']
        else:
            X_test = df
            y_test = None
            st.markdown('<div class="warning-box">⚠️ <strong>Note:</strong> No target column found. Predictions will be made without evaluation metrics.</div>', unsafe_allow_html=True)

        st.divider()

        # ==================== MODEL SELECTION ====================
        st.markdown('<div class="model-gradient-bg"><h2>🤖 Select Machine Learning Model</h2></div>', unsafe_allow_html=True)

        selected_model = st.selectbox(
            "Choose a classification model",
            list(MODEL_INFO.keys()),
            help="Select one of the 6 trained models for prediction"
        )

        st.markdown(
            f'<div class="model-description">'
            f'<p>{MODEL_INFO[selected_model]["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Load model and make predictions
        model = load_model(MODEL_INFO[selected_model]['file'])
        scaler = load_scaler()

        if model is not None:
            X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test.values
            if scaler is None:
                st.warning("⚠️ Scaler not found. Using raw data.")

            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

            st.divider()

            # ==================== RESULTS ====================
            if has_labels:
                st.markdown(
                    '<div class="performance-gradient-bg">'
                    '<h2>📊 Model Performance & Evaluation</h2>'
                    '<h3>🎯 Performance Metrics</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )

                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

                # Display metrics in cards
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                metrics_list = [
                    (col1, "Accuracy", metrics['Accuracy'], "🎯"),
                    (col2, "AUC Score", metrics['AUC'], "📈"),
                    (col3, "Precision", metrics['Precision'], "🎪"),
                    (col4, "Recall", metrics['Recall'], "🔍"),
                    (col5, "F1 Score", metrics['F1 Score'], "⚖️"),
                    (col6, "MCC Score", metrics['MCC'], "🧮")
                ]

                for col, name, value, icon in metrics_list:
                    with col:
                        st.metric(label=f"{icon} {name}", value=f"{value:.4f}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics Bar Chart
                st.plotly_chart(plot_metrics_bar_chart(metrics), use_container_width=True)

                st.divider()

                # ==================== MODEL COMPARISON ====================
                st.markdown("### 🏆 Model Comparison on Current Dataset")

                with st.spinner("Evaluating all models on your dataset..."):
                    all_models_metrics = {}

                    for model_name, model_info in MODEL_INFO.items():
                        try:
                            temp_model = load_model(model_info['file'])
                            if temp_model is not None:
                                temp_pred = temp_model.predict(X_test_scaled)
                                temp_pred_proba = temp_model.predict_proba(X_test_scaled) if hasattr(temp_model, 'predict_proba') else None
                                all_models_metrics[model_name] = calculate_metrics(y_test, temp_pred, temp_pred_proba)
                        except Exception:
                            continue

                if all_models_metrics:
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(all_models_metrics).T
                    comparison_df = comparison_df.round(4)

                    # Find best model based on F1 Score
                    best_model_name = comparison_df['F1 Score'].idxmax()
                    best_f1_score = comparison_df.loc[best_model_name, 'F1 Score']

                    # Display best model info
                    st.markdown(
                        f'<div class="success-box">🏆 <strong>Best Performing Model:</strong> {best_model_name} '
                        f'(F1 Score: {best_f1_score:.4f})</div>',
                        unsafe_allow_html=True
                    )

                    # If selected model is not the best, show side-by-side comparison
                    if selected_model != best_model_name:
                        st.markdown("#### 🔄 Selected vs Best Model Comparison")
                        st.markdown("<br>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        selected_metrics = comparison_df.loc[selected_model]
                        best_metrics = comparison_df.loc[best_model_name]

                        with col1:
                            st.markdown(f'<div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem;">📌 {selected_model}</div>', unsafe_allow_html=True)
                            for metric_name in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']:
                                delta = selected_metrics[metric_name] - best_metrics[metric_name]
                                st.metric(
                                    label=metric_name,
                                    value=f"{selected_metrics[metric_name]:.4f}",
                                    delta=f"{delta:+.4f}" if delta != 0 else "0.0000",
                                    delta_color="normal"
                                )

                        with col2:
                            st.markdown(f'<div style="text-align: center; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem;">🏆 {best_model_name}</div>', unsafe_allow_html=True)
                            for metric_name in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']:
                                delta = best_metrics[metric_name] - selected_metrics[metric_name]
                                st.metric(
                                    label=metric_name,
                                    value=f"{best_metrics[metric_name]:.4f}",
                                    delta=f"{delta:+.4f}" if delta != 0 else "0.0000",
                                    delta_color="normal"
                                )
                    else:
                        st.markdown(
                            '<div class="success-box">✅ You have selected the best performing model for this dataset!</div>',
                            unsafe_allow_html=True
                        )

                st.divider()

                # ==================== DETAILED EVALUATION ====================
                st.markdown("### 📈 Detailed Evaluation")

                # Confusion Matrix
                st.markdown("#### 🎯 Confusion Matrix")
                st.markdown(
                    '<div class="info-box">'
                    '<strong>Understanding Confusion Matrix:</strong><br>'
                    '• <strong>True Negatives (Top-Left):</strong> Correctly predicted healthy patients<br>'
                    '• <strong>False Positives (Top-Right):</strong> Healthy patients incorrectly predicted as diseased<br>'
                    '• <strong>False Negatives (Bottom-Left):</strong> Diseased patients incorrectly predicted as healthy<br>'
                    '• <strong>True Positives (Bottom-Right):</strong> Correctly predicted diseased patients<br>'
                    '</div>',
                    unsafe_allow_html=True
                )

                cm = confusion_matrix(y_test, y_pred)
                st.plotly_chart(plot_confusion_matrix_plotly(cm, f"Confusion Matrix - {selected_model}"), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ True Negatives", cm[0, 0])
                    st.metric("❌ False Positives", cm[0, 1])
                with col2:
                    st.metric("❌ False Negatives", cm[1, 0])
                    st.metric("✅ True Positives", cm[1, 1])

                st.divider()

                # Classification Report
                st.markdown("#### 📊 Classification Report")
                st.markdown(
                    '<div class="info-box">'
                    '<strong>Understanding Classification Report:</strong><br>'
                    '• <strong>Precision:</strong> Of all positive predictions, how many were correct?<br>'
                    '• <strong>Recall:</strong> Of all actual positives, how many did we catch?<br>'
                    '• <strong>F1-Score:</strong> Harmonic mean of precision and recall<br>'
                    '• <strong>Support:</strong> Number of actual occurrences in the dataset<br>'
                    '</div>',
                    unsafe_allow_html=True
                )

                report = classification_report(
                    y_test, y_pred,
                    target_names=['No Disease', 'Disease'],
                    output_dict=True
                )

                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

                st.divider()

                # ROC Curve
                st.markdown("#### 📉 ROC Curve")
                st.markdown(
                    '<div class="info-box">'
                    '<strong>Understanding ROC Curve:</strong><br>'
                    '• <strong>AUC (Area Under Curve):</strong> Measures the model\'s ability to distinguish between classes<br>'
                    '• <strong>Perfect Model:</strong> AUC = 1.0 (curve hugs the top-left corner)<br>'
                    '• <strong>Random Model:</strong> AUC = 0.5 (diagonal line)<br>'
                    '• <strong>Interpretation:</strong> Higher AUC = Better model performance<br>'
                    '</div>',
                    unsafe_allow_html=True
                )

                fig_roc = plot_roc_curve_plotly(y_test, y_pred_proba)
                if fig_roc:
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC curve not available for this model.")

                st.divider()

                # ==================== DOWNLOAD SECTION ====================
                st.markdown('<div class="section-header">📥 Download Reports</div>', unsafe_allow_html=True)

                # Create predictions report
                predictions_report_df = X_test.copy()
                predictions_report_df['Actual'] = ['Disease' if val == 1 else 'No Disease' for val in y_test]
                predictions_report_df['Predicted'] = ['Disease' if val == 1 else 'No Disease' for val in y_pred]
                predictions_report_df['Correct'] = ['✓' if actual == pred else '✗' for actual, pred in zip(y_test, y_pred)]

                if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    predictions_report_df['Probability_No_Disease'] = y_pred_proba[:, 0]
                    predictions_report_df['Probability_Disease'] = y_pred_proba[:, 1]

                st.markdown("#### 📋 Predictions Report Preview")
                st.dataframe(predictions_report_df.head(10), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Full Predictions Report",
                        data=predictions_report_df.to_csv(index=False),
                        file_name=f"predictions_report_{selected_model.replace(' ', '_').lower()}.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Classification Report",
                        data=report_df.to_csv(),
                        file_name=f"classification_report_{selected_model.replace(' ', '_').lower()}.csv",
                        mime="text/csv"
                    )

            else:
                # Predictions only (no labels)
                st.markdown('<div class="section-header">🔮 Predictions</div>', unsafe_allow_html=True)
                st.markdown(f"#### Predictions using **{selected_model}**")

                predictions_df = X_test.copy()
                predictions_df['Prediction'] = ['Disease' if p == 1 else 'No Disease' for p in y_pred]
                predictions_df['Prediction_Code'] = y_pred

                if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    predictions_df['Probability_No_Disease'] = y_pred_proba[:, 0]
                    predictions_df['Probability_Disease'] = y_pred_proba[:, 1]

                st.dataframe(predictions_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔴 Predicted Disease", sum(y_pred == 1))
                with col2:
                    st.metric("🟢 Predicted No Disease", sum(y_pred == 0))

                st.divider()

                st.markdown("### 📥 Download Predictions")
                st.download_button(
                    label="📥 Download Predictions Report",
                    data=predictions_df.to_csv(index=False),
                    file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )

        else:
            st.error("❌ Failed to load model. Please check if model files exist.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Please ensure your CSV has the correct format with the same features as training data.")

else:
    st.markdown('<div class="info-box">👆 <strong>Get Started:</strong> Upload a CSV file above to begin predictions and model evaluation.</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem; border-radius: 10px;'>
    <p style='margin: 0.3rem 0;'>BITS Pilani M.Tech (AIML/DSE) - Machine Learning - Assignment 2</p>
    <p style='margin: 0.3rem 0;'>© 2026 | SAIRAM GAJABINKAR (2025AA05138)</p>
</div>
""", unsafe_allow_html=True)
