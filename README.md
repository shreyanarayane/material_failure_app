# Material Failure Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A machine learning-powered web application that predicts potential failure modes in materials based on their properties and loading conditions.

## Features

- **Material Database**: Includes 7 common engineering materials with their mechanical properties
- **Failure Mode Prediction**: Predicts between Buckling, Wrinkling, Crack, or No Failure
- **Interactive Interface**: User-friendly Streamlit web interface with sliders and selectors
- **Model Insights**: Displays prediction probabilities and feature importance
- **Performance Metrics**: Shows classification report for model evaluation

## Supported Materials

| Material    | Type        | Yield Strength (MPa) | Young's Modulus (GPa) | r-value |
|-------------|-------------|----------------------|-----------------------|---------|
| AA6061-T6   | Aluminum    | 275                  | 69                    | 0.8     |
| AA7075-T6   | Aluminum    | 503                  | 71.7                  | 0.6     |
| DP600       | Steel       | 350                  | 210                   | 1.0     |
| DP780       | Steel       | 500                  | 210                   | 0.9     |
| SS304       | Stainless   | 215                  | 193                   | 1.2     |
| Ti-6Al-4V   | Titanium    | 830                  | 113.8                 | 2.5     |
| AZ31B       | Magnesium   | 160                  | 45                    | 3.0     |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/material-failure-predictor.git
cd material-failure-predictor
