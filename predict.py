import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("heart_disease_model.pkl")

MODEL_ACCURACY = 0.8424
MODEL_PRECISION = 0.8174
MODEL_RECALL = 0.9216
MODEL_F1 = 0.8664

# ============================================================
# DISPLAY TITLE
# ============================================================

print("=" * 60)
print("HEART DISEASE PREDICTION SYSTEM")
print("=" * 60)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print("Accuracy : 84.24%")
print("Precision: 81.74%")
print("Recall   : 92.16%")
print("F1-Score : 86.64%")
print("✅ Model Accuracy: 84.24%")

print("\nEnter the following patient details:\n")


# ============================================================
# 1. AGE
# ============================================================

age = float(input("Age: "))


# ============================================================
# 2. SEX
# ============================================================

print("\nSex:")
print("1. Male")
print("2. Female")

sex_choice = int(input("Enter choice (1/2): "))

if sex_choice == 1:
    sex = "Male"
else:
    sex = "Female"


# ============================================================
# 3. CHEST PAIN TYPE
# ============================================================

print("\nChest Pain Type:")
print("1. Typical Angina")
print("2. Atypical Angina")
print("3. Non-anginal Pain")
print("4. Asymptomatic")

cp_choice = int(input("Enter choice (1-4): "))

cp_values = {
    1: "typical angina",
    2: "atypical angina",
    3: "non-anginal",
    4: "asymptomatic"
}

cp = cp_values[cp_choice]


# ============================================================
# 4. RESTING BLOOD PRESSURE
# ============================================================

trestbps = float(input("\nResting Blood Pressure (mm Hg): "))


# ============================================================
# 5. CHOLESTEROL
# ============================================================

chol = float(input("Cholesterol (mg/dl): "))


# ============================================================
# 6. FASTING BLOOD SUGAR
# ============================================================

print("\nFasting Blood Sugar:")
print("1. <= 120 mg/dl")
print("2. > 120 mg/dl")

fbs_choice = int(input("Enter choice (1/2): "))

if fbs_choice == 1:
    fbs = "False"
else:
    fbs = "True"


# ============================================================
# 7. RESTING ECG
# ============================================================

print("\nResting ECG:")
print("1. Normal")
print("2. ST-T Wave Abnormality")
print("3. Left Ventricular Hypertrophy")

restecg_choice = int(input("Enter choice (1-3): "))

restecg_values = {
    1: "normal",
    2: "st-t abnormality",
    3: "lv hypertrophy"
}

restecg = restecg_values[restecg_choice]


# ============================================================
# 8. MAXIMUM HEART RATE
# ============================================================

thalch = float(input("\nMaximum Heart Rate Achieved: "))


# ============================================================
# 9. EXERCISE-INDUCED ANGINA
# ============================================================

print("\nExercise-Induced Angina:")
print("1. No")
print("2. Yes")

exang_choice = int(input("Enter choice (1/2): "))

if exang_choice == 1:
    exang = "False"
else:
    exang = "True"


# ============================================================
# 10. OLDPEAK
# ============================================================

oldpeak = float(input("\nOldpeak: "))


# ============================================================
# 11. SLOPE
# ============================================================

print("\nSlope of Peak Exercise ST Segment:")
print("1. Upsloping")
print("2. Flat")
print("3. Downsloping")

slope_choice = int(input("Enter choice (1-3): "))

slope_values = {
    1: "upsloping",
    2: "flat",
    3: "downsloping"
}

slope = slope_values[slope_choice]


# ============================================================
# 12. CA
# ============================================================

print("\nNumber of Major Vessels:")
print("0, 1, 2, or 3")

ca = float(input("Enter number of vessels (0-3): "))


# ============================================================
# 13. THAL
# ============================================================

print("\nThalassemia:")
print("1. Normal")
print("2. Fixed Defect")
print("3. Reversible Defect")

thal_choice = int(input("Enter choice (1-3): "))

thal_values = {
    1: "normal",
    2: "fixed defect",
    3: "reversable defect"
}

thal = thal_values[thal_choice]


# ============================================================
# CREATE PATIENT DATA
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
# PREDICTION
# ============================================================

prediction = model.predict(patient_data)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

if prediction[0] == 1:

    print("\n⚠ HEART DISEASE DETECTED")

else:

    print("\n✓ NO HEART DISEASE DETECTED")

print("\n" + "=" * 60)