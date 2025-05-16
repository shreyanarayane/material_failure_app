# 🔧 SheetMetal AI: Predicting Failure Modes in Forming Processes

Welcome to **SheetMetal AI**, an advanced machine learning-powered web application designed to assist engineers and researchers in predicting **failure modes in sheet metal forming processes**. Whether you're designing automotive components or optimizing material flow, this tool empowers you to make informed decisions about potential failure modes such as **Cracking**, **Wrinkling**, or **Buckling**.

By leveraging synthetic yet industry-inspired data, **SheetMetal AI** integrates **Finite Element Analysis (FEA)** principles with **Machine Learning (ML)** to offer early-stage failure prediction—eliminating the need for costly trials or simulations.

---
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
---

## Objective

The primary goals of this project are to:
- Develop a **robust and interpretable machine learning model** to predict failure types in sheet metal forming.
- Combine material properties and forming parameters to mimic real-world scenarios.
- Deliver an intuitive, interactive web interface built with **Streamlit** for seamless use in industrial and academic applications.

---

## Problem Background

In sheet metal forming operations, **failure** often arises from improper process parameters or unsuitable material selection. Common failure types include:
- **Cracking**: Caused by excessive tensile stress.
- **Buckling**: Resulting from compressive instability.
- **Wrinkling**: Due to inadequate blank holder force.

Traditional failure detection methods, like FEA simulations, are time-intensive and expensive. **SheetMetal AI** offers a faster, cost-effective alternative using a trained **ML classifier** to predict likely failure types based on key input parameters.

---

## Tech Stack

- **Frontend**: Streamlit (interactive UI)
- **Backend**: Python
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Data Handling**: Pandas, NumPy
- **Model Type**: Random Forest Classifier
- **Balancing Technique**: SMOTENC (for mixed categorical and numerical features)

---

## Input Parameters

The following input parameters are used for failure prediction:

| Parameter                | Type         | Description                                   |
|--------------------------|--------------|-----------------------------------------------|
| `Material Type`          | Categorical  | Type of sheet metal (e.g., Aluminum, Steel)   |
| `Thickness`              | Continuous   | Thickness of the sheet (mm)                   |
| `Yield Strength`         | Continuous   | Material yield strength (MPa)                 |
| `Punch Speed`            | Continuous   | Punch velocity during forming (mm/s)          |
| `Die Radius`             | Continuous   | Radius of the forming die (mm)                |
| `Blank Holder Force`     | Continuous   | Force applied to prevent wrinkling (kN)       |
| `Friction Coefficient`   | Continuous   | Friction value between punch and sheet        |

---

## Labels / Output Classes

The model predicts one of the following failure modes:
- **Cracking**
- **Buckling**
- **Wrinkling**

These predictions enable CAE engineers to proactively adjust forming parameters, reducing the risk of production issues.

---

## Machine Learning Pipeline

### 1. Data Preprocessing
- Seamlessly handle mixed data types (categorical and numerical).
- Standardize numerical features for uniform scaling.
- Encode categorical values like `Material Type` using **LabelEncoder**.

### 2. Handling Class Imbalance
- Leverages **SMOTENC** to generate synthetic samples for minority classes.
- Ensures the model remains unbiased toward majority classes.

### 3. Model Training
- Utilizes a **Random Forest Classifier** with:
  - 100 estimators for robust predictions.
  - Balanced class weights to handle imbalanced data.
  - Optimized max depth to prevent overfitting.
- Train/test split: **80/20**.

### 4. Evaluation Metrics
- **Accuracy**: Measures overall performance.
- **Precision/Recall/F1-Score**: Evaluates class-specific performance.
- **Confusion Matrix**: Visualized for detailed error analysis.

---

## Features of the App

- **Streamlit-Based UI**: A simple, interactive interface for user-friendly operation.
- **Real-Time Predictions**: Enter forming parameters via sliders and dropdowns to get instant failure predictions.
- **Extensibility**: Future capabilities include:
  - Prediction confidence scores.
  - Visual explanations using tools like **SHAP**.

---

## How to Run Locally

Follow these steps to run the app on your local machine:

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/SheetMetal-AI.git
cd SheetMetal-AI

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
streamlit run app/streamlit_app.py
```

---

## Requirements

Ensure you have the following installed:

- **Python**: Version 3.7 or higher
- **Packages**:
  - `streamlit>=1.22`
  - `numpy>=1.21`
  - `pandas>=1.3`
  - `scikit-learn>=1.0`
  - `imbalanced-learn>=0.9`
  - `matplotlib>=3.5`

---

## Future Improvements

- Extend support for additional failure modes.
- Integrate real-world experimental data.
- Add advanced visualization tools for parameter sensitivity analysis.

---

#### Developed by  
 **Shreya Narayane** |
 **Data Science, COEP** 
