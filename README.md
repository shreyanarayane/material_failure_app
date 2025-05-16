# Sheet Metal Forming Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A machine learning-powered web application that predicts potential forming defects in sheet metal manufacturing processes based on material properties and process parameters.

## 📌 Overview

This application helps manufacturing engineers and material scientists:
- Predict potential forming defects in sheet metal operations
- Optimize process parameters to prevent defects
- Understand key factors influencing forming quality
- Compare different materials for forming applications

The system uses a Random Forest classifier trained on synthetic data simulating various sheet metal forming scenarios.

## ✨ Features

- **Material Database**: Includes 7 common sheet metals with mechanical properties
- **Defect Prediction**: Predicts forming defects (Wrinkling, Cracking, Buckling) or Successful Forming
- **Interactive Process Simulation**: Adjust process parameters and see predictions in real-time
- **Probability Analysis**: View defect probability distribution
- **Feature Importance**: Understand key factors affecting forming quality
- **Performance Metrics**: Detailed model evaluation metrics

## 🏭 Supported Materials

| Material    | Type        | Yield Strength (MPa) | Young's Modulus (GPa) | r-value | Typical Applications |
|-------------|-------------|----------------------|-----------------------|---------|----------------------|
| AA6061-T6   | Aluminum    | 275                  | 69                    | 0.8     | Aircraft components, automotive parts |
| AA7075-T6   | Aluminum    | 503                  | 71.7                  | 0.6     | Aerospace structures |
| DP600       | Steel       | 350                  | 210                   | 1.0     | Automotive body panels |
| DP780       | Steel       | 500                  | 210                   | 0.9     | Structural components |
| SS304       | Stainless   | 215                  | 193                   | 1.2     | Kitchen equipment, chemical containers |
| Ti-6Al-4V   | Titanium    | 830                  | 113.8                 | 2.5     | Aerospace, medical implants |
| AZ31B       | Magnesium   | 160                  | 45                    | 3.0     | Lightweight components |

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sheet-metal-forming-predictor.git
cd sheet-metal-forming-predictor
