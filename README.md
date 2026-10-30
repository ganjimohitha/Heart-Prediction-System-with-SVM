# Heart Disease Prediction System Using Support Vector Machine (SVM)

## Project Overview
The Heart Disease Prediction System is a machine learning-based application that predicts the possibility of heart disease using patient medical information.

This project uses Support Vector Machine (SVM) classification to analyze patient data and classify the result into two categories:
- Heart Disease Detected
- No Heart Disease Detected

The main objective is to develop a simple and user-friendly prediction system that can assist in the early identification of potential heart disease cases.

## Objectives
- To analyze heart disease data using machine learning.
- To implement different SVM kernels.
- To compare model performance using evaluation metrics.
- To identify the best-performing SVM model.
- To develop a patient prediction application.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Algorithm:** Support Vector Machine (SVM)
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
- **Web Application:** Streamlit
- **Development Environment:** Visual Studio Code
- **Dataset:** UCI Heart Disease Dataset

## SVM Kernels Implemented
Three SVM kernels were implemented and compared:
1. Linear Kernel
2. RBF Kernel
3. Polynomial Kernel

The RBF kernel was selected as the final model based on its performance.

## Model Performance

The final tuned RBF SVM model achieved the following results on the test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 84.24% |
| Precision | 81.74% |
| Recall | 92.16% |
| F1-Score | 86.64% |

**Best Hyperparameters:**
- C = 1
- Gamma = 0.1
- Kernel = RBF

## Application Features
- Accepts 13 patient medical parameters.
- Processes the input using the trained RBF SVM model.
- Predicts the heart disease class.
- Displays the prediction result through a user-friendly interface.

## Project Files
- `heart_disease_svm.py` – Model training, evaluation, and hyperparameter tuning.
- `heart_disease_model.pkl` – Saved trained machine learning model.
- `heart_disease_uci.csv` – Dataset used for model development.
- Streamlit application file – Patient prediction interface.

## How to Run the Project

**Step 1: Clone the repository**
```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

**Step 2: Open the project folder**
```bash
cd YOUR_PROJECT_FOLDER
```

**Step 3: Install the required libraries**
```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

**Step 4: Run the application**

If your Streamlit application file is named `app.py`, run:
```bash
streamlit run app.py
```

## Conclusion
The project successfully implements a Heart Disease Prediction System using Support Vector Machine. Among the three kernels, the tuned RBF SVM model was selected based on its performance and integrated into a patient prediction application.

## Disclaimer
This project is developed for academic and educational purposes only. Its predictions are not a substitute for professional medical advice or diagnosis.

## Dataset Reference
UCI Machine Learning Repository – Heart Disease Dataset

https://archive.ics.uci.edu/dataset/45/heart+disease
