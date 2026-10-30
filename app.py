# import streamlit as st
# import pandas as pd
# import joblib


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Heart Disease Prediction System",
#     page_icon="❤️",
#     layout="wide"
# )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = joblib.load("heart_disease_model.pkl")


# # ============================================================
# # HEADER
# # ============================================================

# st.title("❤️ Heart Disease Prediction System")

# st.write(
#     "Machine Learning based heart disease prediction using "
#     "Support Vector Machine (SVM)."
# )

# st.divider()


# # ============================================================
# # MODEL PERFORMANCE
# # ============================================================

# st.subheader("📊 Model Performance")

# col1, col2, col3, col4 = st.columns(4)

# col1.metric("Accuracy", "84.24%")
# col2.metric("Precision", "81.74%")
# col3.metric("Recall", "92.16%")
# col4.metric("F1-Score", "86.64%")

# st.info(
#     "Final Model: Tuned RBF SVM  |  C = 1  |  Gamma = 0.1"
# )

# st.divider()


# # ============================================================
# # PATIENT DETAILS
# # ============================================================

# st.subheader("👤 Enter Patient Details")

# left, right = st.columns(2)


# # ============================================================
# # LEFT COLUMN
# # ============================================================

# with left:

#     age = st.number_input(
#         "Age",
#         min_value=1,
#         max_value=120,
#         value=50,
#         step=1
#     )

#     sex_option = st.selectbox(
#         "Sex",
#         ["Male", "Female"]
#     )

#     cp_option = st.selectbox(
#         "Chest Pain Type",
#         [
#             "Typical Angina",
#             "Atypical Angina",
#             "Non-anginal Pain",
#             "Asymptomatic"
#         ]
#     )

#     trestbps = st.number_input(
#         "Resting Blood Pressure (mm Hg)",
#         min_value=50.0,
#         max_value=250.0,
#         value=120.0,
#         step=1.0
#     )

#     chol = st.number_input(
#         "Cholesterol (mg/dl)",
#         min_value=50.0,
#         max_value=700.0,
#         value=200.0,
#         step=1.0
#     )

#     fbs_option = st.selectbox(
#         "Fasting Blood Sugar",
#         [
#             "≤ 120 mg/dl",
#             "> 120 mg/dl"
#         ]
#     )

#     restecg_option = st.selectbox(
#         "Resting ECG",
#         [
#             "Normal",
#             "ST-T Wave Abnormality",
#             "Left Ventricular Hypertrophy"
#         ]
#     )


# # ============================================================
# # RIGHT COLUMN
# # ============================================================

# with right:

#     thalch = st.number_input(
#         "Maximum Heart Rate Achieved",
#         min_value=50.0,
#         max_value=250.0,
#         value=150.0,
#         step=1.0
#     )

#     exang_option = st.selectbox(
#         "Exercise-Induced Angina",
#         [
#             "No",
#             "Yes"
#         ]
#     )

#     oldpeak = st.number_input(
#         "Oldpeak",
#         min_value=0.0,
#         max_value=10.0,
#         value=1.0,
#         step=0.1
#     )

#     slope_option = st.selectbox(
#         "Slope of Peak Exercise ST Segment",
#         [
#             "Upsloping",
#             "Flat",
#             "Downsloping"
#         ]
#     )

#     ca = st.number_input(
#         "Number of Major Vessels (0–3)",
#         min_value=0.0,
#         max_value=3.0,
#         value=0.0,
#         step=1.0
#     )

#     thal_option = st.selectbox(
#         "Thalassemia",
#         [
#             "Normal",
#             "Fixed Defect",
#             "Reversible Defect"
#         ]
#     )


# # ============================================================
# # CONVERT INPUTS
# # ============================================================

# sex = sex_option


# cp_values = {
#     "Typical Angina": "typical angina",
#     "Atypical Angina": "atypical angina",
#     "Non-anginal Pain": "non-anginal",
#     "Asymptomatic": "asymptomatic"
# }

# cp = cp_values[cp_option]


# fbs = "False" if fbs_option == "≤ 120 mg/dl" else "True"


# restecg_values = {
#     "Normal": "normal",
#     "ST-T Wave Abnormality": "st-t abnormality",
#     "Left Ventricular Hypertrophy": "lv hypertrophy"
# }

# restecg = restecg_values[restecg_option]


# exang = "False" if exang_option == "No" else "True"


# slope_values = {
#     "Upsloping": "upsloping",
#     "Flat": "flat",
#     "Downsloping": "downsloping"
# }

# slope = slope_values[slope_option]


# thal_values = {
#     "Normal": "normal",
#     "Fixed Defect": "fixed defect",
#     "Reversible Defect": "reversable defect"
# }

# thal = thal_values[thal_option]


# # ============================================================
# # PATIENT DATA
# # ============================================================

# patient_data = pd.DataFrame({
#     "age": [age],
#     "sex": [sex],
#     "cp": [cp],
#     "trestbps": [trestbps],
#     "chol": [chol],
#     "fbs": [fbs],
#     "restecg": [restecg],
#     "thalch": [thalch],
#     "exang": [exang],
#     "oldpeak": [oldpeak],
#     "slope": [slope],
#     "ca": [ca],
#     "thal": [thal]
# })


# # ============================================================
# # PREDICTION BUTTON
# # ============================================================

# st.divider()

# if st.button(
#     "🔍 Predict Heart Disease",
#     type="primary",
#     use_container_width=True
# ):

#     prediction = model.predict(patient_data)

#     st.subheader("📋 Prediction Result")

#     if prediction[0] == 1:

#         st.error(
#             "⚠️ HEART DISEASE DETECTED"
#         )

#         st.write(
#             "The model predicts the **Disease** class."
#         )

#     else:

#         st.success(
#             "✅ NO HEART DISEASE DETECTED"
#         )

#         st.write(
#             "The model predicts the **No Disease** class."
#         )


# # ============================================================
# # MODEL INFORMATION
# # ============================================================

# st.divider()

# st.subheader("🤖 Model Information")

# info1, info2 = st.columns(2)

# with info1:
#     st.write("**Algorithm:** Support Vector Machine")
#     st.write("**Kernel:** RBF")
#     st.write("**C:** 1")
#     st.write("**Gamma:** 0.1")

# with info2:
#     st.write("**Cross-Validation:** 5-Fold")
#     st.write("**Accuracy:** 84.24%")
#     st.write("**Recall:** 92.16%")
#     st.write("**F1-Score:** 86.64%")


# # ============================================================
# # DISCLAIMER
# # ============================================================

# st.divider()

# st.caption(
#     "⚠️ This system is developed for academic and demonstration "
#     "purposes only. It is not a medical diagnostic tool."
# )




import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("heart_disease_model.pkl")


# ============================================================
# HEADER
# ============================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Machine Learning based heart disease prediction using "
    "Support Vector Machine (SVM)."
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "84.24%")
col2.metric("Precision", "81.74%")
col3.metric("Recall", "92.16%")
col4.metric("F1-Score", "86.64%")

st.info(
    "Final Model: Tuned RBF SVM  |  C = 1  |  Gamma = 0.1"
)

st.divider()


# ============================================================
# PATIENT DETAILS
# ============================================================

st.subheader("👤 Enter Patient Details")



left, right = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with left:

    # ---------------- AGE ----------------

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=None,
        step=1,
        placeholder="Enter age"
    )




    # ---------------- SEX ----------------

    sex_option = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )


    # ---------------- CHEST PAIN ----------------

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )


    # ---------------- BLOOD PRESSURE ----------------

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50.0,
        max_value=250.0,
        value=None,
        step=1.0,
        placeholder="e.g. 120"
    )




    # ---------------- CHOLESTEROL ----------------

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=50.0,
        max_value=700.0,
        value=None,
        step=1.0,
        placeholder="e.g. 180"
    )


   


    # ---------------- FASTING BLOOD SUGAR ----------------

    fbs_option = st.selectbox(
        "Fasting Blood Sugar",
        [
            "≤ 120 mg/dL",
            "> 120 mg/dL"
        ]
    )


    # ---------------- RESTING ECG ----------------

    restecg_option = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:

    # ---------------- MAX HEART RATE ----------------

    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50.0,
        max_value=250.0,
        value=None,
        step=1.0,
        placeholder="e.g. 150"
    )




    # ---------------- EXERCISE ANGINA ----------------

    exang_option = st.selectbox(
        "Exercise-Induced Angina",
        [
            "No",
            "Yes"
        ]
    )


    # ---------------- OLDPEAK ----------------

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=None,
        step=0.1,
        placeholder="e.g. 1.0"
    )

    

    # ---------------- SLOPE ----------------

    slope_option = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )


    # ---------------- MAJOR VESSELS ----------------

    ca = st.number_input(
        "Number of Major Vessels (0–3)",
        min_value=0.0,
        max_value=3.0,
        value=None,
        step=1.0,
        placeholder="Enter 0–3"
    )

   


    # ---------------- THALASSEMIA ----------------

    thal_option = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )


# ============================================================
# CONVERT INPUTS
# ============================================================

sex = sex_option


cp_values = {
    "Typical Angina": "typical angina",
    "Atypical Angina": "atypical angina",
    "Non-anginal Pain": "non-anginal",
    "Asymptomatic": "asymptomatic"
}

cp = cp_values[cp_option]


fbs = "False" if fbs_option == "≤ 120 mg/dL" else "True"


restecg_values = {
    "Normal": "normal",
    "ST-T Wave Abnormality": "st-t abnormality",
    "Left Ventricular Hypertrophy": "lv hypertrophy"
}

restecg = restecg_values[restecg_option]


exang = "False" if exang_option == "No" else "True"


slope_values = {
    "Upsloping": "upsloping",
    "Flat": "flat",
    "Downsloping": "downsloping"
}

slope = slope_values[slope_option]


thal_values = {
    "Normal": "normal",
    "Fixed Defect": "fixed defect",
    "Reversible Defect": "reversable defect"
}

thal = thal_values[thal_option]


# ============================================================
# PATIENT DATA
# ============================================================

patient_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalch": [thalch],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
):

    # Check required numerical inputs
    required_values = {
        "Age": age,
        "Resting Blood Pressure": trestbps,
        "Cholesterol": chol,
        "Maximum Heart Rate": thalch,
        "Oldpeak": oldpeak,
        "Number of Major Vessels": ca
    }

    missing_values = [
        name for name, value in required_values.items()
        if value is None
    ]

    if missing_values:

        st.warning(
            "Please enter the following values before prediction: "
            + ", ".join(missing_values)
        )

    else:

        prediction = model.predict(patient_data)

        st.subheader("📋 Prediction Result")

        if prediction[0] == 1:

            st.error(
                "⚠️ HEART DISEASE DETECTED"
            )

            st.write(
                "The model predicts the **Disease** class."
            )

        else:

            st.success(
                "✅ NO HEART DISEASE DETECTED"
            )

            st.write(
                "The model predicts the **No Disease** class."
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")

info1, info2 = st.columns(2)

with info1:

    st.write("**Algorithm:** Support Vector Machine")
    st.write("**Kernel:** RBF")
    st.write("**C:** 1")
    st.write("**Gamma:** 0.1")


with info2:

    st.write("**Cross-Validation:** 5-Fold")
    st.write("**Accuracy:** 84.24%")
    st.write("**Recall:** 92.16%")
    st.write("**F1-Score:** 86.64%")


# ============================================================
# DEVELOPED BY
# ============================================================

st.divider()

st.subheader("👩‍💻 Developed By")

team_col1, team_col2 = st.columns(2)

with team_col1:
    st.write("**J Ganga Bhavani**")
    st.write("**G Mohitha**")

with team_col2:
    st.write("**Y Jahnavi**")
    st.write("**Y Kavya**")


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

# # streamlit run app.py