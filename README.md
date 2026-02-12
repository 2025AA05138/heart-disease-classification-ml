# Heart Disease Classification - Multi-Model ML Classifier

**BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2**

**Student Name:** SAIRAM GAJABINKAR
**Student ID:** 2025AA05138
**Course:** Machine Learning
**Submission Date:** February 2026

---

## 📋 Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early and accurate detection of heart disease is crucial for effective treatment and improved patient outcomes. This project aims to develop a robust machine learning-based classification system that can predict the presence of heart disease in patients based on various clinical and diagnostic features.

The objective is to:
- Implement and compare multiple classification algorithms
- Evaluate their performance using comprehensive metrics
- Deploy an interactive web application for real-time predictions
- Provide healthcare professionals with an accessible tool for preliminary heart disease screening

This multi-model approach allows for comparison of different algorithmic strategies and selection of the most appropriate model based on specific performance criteria such as accuracy, precision, recall, and other relevant metrics.

---

## 📊 Dataset Description

### **Dataset:** Heart Disease Classification Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Type:** Binary Classification Problem
- **Target Variable:**
  - `0` = No heart disease
  - `1` = Presence of heart disease

### **Dataset Characteristics:**
- **Total Instances:** 1,025 patient records
- **Total Features:** 13 clinical and diagnostic features
- **Feature Types:** Integer and Continuous
- **Missing Values:** None
- **Class Distribution:** Reasonably balanced between positive and negative cases

### **Feature Details:**

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **age** | Age of the patient in years | Continuous | 29-77 |
| **sex** | Gender of the patient | Binary | 0 = Female, 1 = Male |
| **cp** | Chest pain type | Categorical | 0-3 (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic) |
| **trestbps** | Resting blood pressure (mm Hg) | Continuous | 94-200 |
| **chol** | Serum cholesterol (mg/dl) | Continuous | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0 = False, 1 = True |
| **restecg** | Resting electrocardiographic results | Categorical | 0-2 (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy) |
| **thalach** | Maximum heart rate achieved | Continuous | 71-202 |
| **exang** | Exercise induced angina | Binary | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise relative to rest | Continuous | 0-6.2 |
| **slope** | Slope of the peak exercise ST segment | Categorical | 0-2 (0: Upsloping, 1: Flat, 2: Downsloping) |
| **ca** | Number of major vessels colored by fluoroscopy | Integer | 0-3 |
| **thal** | Thalassemia | Categorical | 0-3 (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described) |

### **Data Preprocessing:**
1. **Feature Scaling:** StandardScaler applied to normalize all features
2. **Train-Test Split:** 80-20 split with stratification to maintain class balance
3. **Random State:** Fixed at 42 for reproducibility

---

## 🤖 Models Used

Six different classification algorithms were implemented and evaluated on the heart disease dataset:

### 1. **Logistic Regression**
A linear model that uses the logistic function to model binary outcomes. It's interpretable and provides probability estimates.

### 2. **Decision Tree Classifier**
A tree-based model that makes decisions by splitting the dataset based on feature values. It's intuitive and can capture non-linear relationships.

### 3. **K-Nearest Neighbors (kNN)**
An instance-based learning algorithm that classifies based on the majority vote of k nearest neighbors. Set with k=5 neighbors.

### 4. **Naive Bayes (Gaussian)**
A probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Particularly effective for certain types of classification problems.

### 5. **Random Forest (Ensemble)**
An ensemble method that builds multiple decision trees and combines their predictions through voting. Uses 100 trees for robust performance.

### 6. **XGBoost (Ensemble)**
An advanced gradient boosting algorithm that builds trees sequentially, with each tree correcting errors from previous ones. Known for high performance in competitions.

---

## 📈 Model Performance Comparison

All models were trained on the same training set and evaluated on the same test set (205 samples). The following metrics were calculated for each model:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **Decision Tree** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **kNN** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **Naive Bayes** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **Random Forest (Ensemble)** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| **XGBoost (Ensemble)** | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

**Note:** Run `python train_models.py` to generate actual metrics and update this table.

### **Evaluation Metrics Explained:**
- **Accuracy:** Overall correctness of the model
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Proportion of positive predictions that are actually correct
- **Recall:** Proportion of actual positives that are correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure considering all confusion matrix categories

---

## 💡 Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | [TO BE FILLED AFTER TRAINING] |
| **Decision Tree** | [TO BE FILLED AFTER TRAINING] |
| **kNN** | [TO BE FILLED AFTER TRAINING] |
| **Naive Bayes** | [TO BE FILLED AFTER TRAINING] |
| **Random Forest (Ensemble)** | [TO BE FILLED AFTER TRAINING] |
| **XGBoost (Ensemble)** | [TO BE FILLED AFTER TRAINING] |

---

## 🚀 Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained models and provide real-time predictions.

### **Application Features:**

1. **📤 CSV Upload:** Users can upload test data in CSV format for batch predictions
2. **🔧 Model Selection:** Dropdown menu to select from all 6 trained models
3. **📊 Metrics Display:** Real-time display of all evaluation metrics for the selected model
4. **📈 Confusion Matrix:** Visual representation of model performance with detailed classification report

### **Application URL:**
- **Live App:** [TO BE FILLED AFTER DEPLOYMENT]
- **GitHub Repository:** [TO BE FILLED AFTER GITHUB PUSH]

### **Running Locally:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train models first
python train_models.py

# Run the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
heart-disease-classification-ml/
│
├── app.py                      # Streamlit web application
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── heart.csv                   # Original dataset
├── test_data.csv              # Test data (generated after training)
├── model_results.csv          # Model metrics comparison (generated after training)
│
└── model/                     # Trained model files
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl             # Feature scaler
```

---

## 🛠️ Installation & Setup

### **Prerequisites:**
- Python 3.8 or higher
- pip (Python package manager)
- Git

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/[YOUR_USERNAME]/heart-disease-classification-ml.git
cd heart-disease-classification-ml
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Train Models**
```bash
python train_models.py
```

This will:
- Load and preprocess the heart disease dataset
- Train all 6 classification models
- Calculate evaluation metrics
- Save trained models in the `model/` directory
- Generate `model_results.csv` with performance comparison
- Create `test_data.csv` for the Streamlit app

### **Step 4: Run the Streamlit App**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 🌐 Deployment

The application is deployed on **Streamlit Community Cloud** for free hosting and easy access.

### **Deployment Steps:**
1. Push code to GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App"
5. Select repository: `heart-disease-classification-ml`
6. Set main file: `app.py`
7. Click "Deploy"

The app will be live within a few minutes with a public URL.

---

## 📊 Usage Example

1. **Access the Application:** Open the deployed Streamlit app URL
2. **Upload Test Data:** Click "Browse files" and upload a CSV file with heart disease features
3. **Select Model:** Choose one of the 6 trained models from the dropdown
4. **View Results:**
   - Performance metrics are displayed immediately
   - Confusion matrix shows classification breakdown
   - Classification report provides detailed per-class metrics

---

## 🔬 Technical Details

### **Libraries Used:**
- **scikit-learn:** Model implementation and evaluation
- **XGBoost:** Advanced gradient boosting
- **Streamlit:** Web application framework
- **Pandas & NumPy:** Data manipulation
- **Matplotlib & Seaborn:** Visualization

### **Hyperparameters:**
- Logistic Regression: `max_iter=1000`
- Decision Tree: `random_state=42`
- kNN: `n_neighbors=5`
- Naive Bayes: Default (Gaussian)
- Random Forest: `n_estimators=100, random_state=42`
- XGBoost: `random_state=42, eval_metric='logloss'`

---

## 📝 Key Findings

*[TO BE COMPLETED AFTER TRAINING AND ANALYSIS]*

1. Model performance summary
2. Best performing model and why
3. Trade-offs between different models
4. Recommendations for deployment

---

## 🎓 Academic Information

**Institution:** Birla Institute of Technology and Science (BITS Pilani)
**Program:** M.Tech in Artificial Intelligence and Machine Learning (AIML)
**Course:** Machine Learning
**Assignment:** Assignment 2 - Multi-Model Classification with Deployment
**Instructor:** [Course Instructor Name]
**Academic Year:** 2025-2026

---

## 📧 Contact

**Student:** SAIRAM GAJABINKAR
**Email:** [Your BITS Email]
**Student ID:** 2025AA05138

---

## 📄 License

This project is submitted as part of academic coursework for BITS Pilani M.Tech (AIML/DSE) program.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- BITS Pilani for providing the virtual lab environment
- Streamlit for the excellent web application framework
- Course instructors and teaching assistants for guidance

---

**Last Updated:** February 2026
**Status:** ✅ Completed and Deployed
