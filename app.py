import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ======================
# SET UP THE APP
# ======================
st.set_page_config(page_title="Material Failure Predictor", layout="wide")
st.title("Material Failure Prediction System")
st.write("""
This tool predicts potential failure modes in materials based on their properties and loading conditions.
""")

# ======================
# MATERIAL DATABASE (MOVED TO TOP FOR UI)
# ======================
materials_db = {
    "AA6061-T6": {"type": "Aluminum", "yield": 275, "E": 69e3, "r": 0.8},
    "AA7075-T6": {"type": "Aluminum", "yield": 503, "E": 71.7e3, "r": 0.6},
    "DP600": {"type": "Steel", "yield": 350, "E": 210e3, "r": 1.0},
    "DP780": {"type": "Steel", "yield": 500, "E": 210e3, "r": 0.9},
    "SS304": {"type": "Stainless", "yield": 215, "E": 193e3, "r": 1.2},
    "Ti-6Al-4V": {"type": "Titanium", "yield": 830, "E": 113.8e3, "r": 2.5},
    "AZ31B": {"type": "Magnesium", "yield": 160, "E": 45e3, "r": 3.0}
}

# ======================
# USER INPUT SECTION
# ======================
with st.sidebar:
    st.header("Material Parameters")
    
    # Material selection
    material = st.selectbox("Select Material", list(materials_db.keys()))
    
    # Get default values from database
    defaults = materials_db[material]
    
    # Input fields with material-specific defaults
    thickness = st.number_input("Thickness (mm)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    strain = st.number_input("Strain (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    stress = st.number_input("Stress (MPa)", min_value=0.0, value=defaults["yield"]*1.2, step=10.0)
    compression = st.number_input("Compression Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    friction = st.number_input("Friction Coefficient", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# ======================
# DATA GENERATION AND MODEL TRAINING (CACHED)
# ======================
@st.cache_data
def generate_and_train():
    np.random.seed(42)
    n_samples = 1500
    df_list = []

    for mat_name, props in materials_db.items():
        n = n_samples // len(materials_db)

        data = {
            'material': [mat_name] * n,
            'type': [props["type"]] * n,
            'yield': np.clip(np.random.normal(props["yield"], 0.05*props["yield"], n), 
                           0.8*props["yield"], 1.2*props["yield"]),
            'E': [props["E"]] * n,
            'r': np.clip(np.random.normal(props["r"], 0.1, n), 
                 0.7*props["r"], 1.3*props["r"]),
            'thickness': np.random.uniform(1.0, 3.0, n),
            'strain': np.clip(np.random.normal(0.04, 0.01, n), 0.02, 0.06),
            'stress': np.clip(np.random.normal(1.2*props["yield"], 0.2*props["yield"], n), 
                     0.9*props["yield"], 1.5*props["yield"]),
            'compression': np.random.uniform(0.3, 0.7, n),
            'friction': np.random.uniform(0.1, 0.3, n)
        }

        temp_df = pd.DataFrame(data)

        # Failure rules
        temp_df['failure'] = 'None'
        temp_df.loc[
            (temp_df['stress'] > 0.9*temp_df['E']*(temp_df['thickness']**2)) &
            (temp_df['strain'] > 0.03),
            'failure'
        ] = 'Buckling'
        temp_df.loc[
            (temp_df['compression'] > 0.6) &
            (temp_df['friction'] < 0.2),
            'failure'
        ] = 'Wrinkling'
        temp_df.loc[
            (temp_df['stress'] > 1.5*temp_df['yield']),
            'failure'
        ] = 'Crack'

        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    df['stiffness_ratio'] = df['E'] / df['yield']
    df['slenderness'] = df['thickness'] / df['strain']

    # Preprocessing
    X = df[['type', 'yield', 'E', 'r', 'thickness', 'strain', 'stress', 'compression', 'friction', 'stiffness_ratio', 'slenderness']]
    y = df['failure']

    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X[['type']])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['type']))
    X_processed = pd.concat([X_encoded_df, X.drop('type', axis=1)], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    # SMOTE-NC balancing
    smote_nc = SMOTENC(
        categorical_features=list(range(X_encoded_df.shape[1])),
        sampling_strategy='not majority',
        random_state=42,
        k_neighbors=5
    )
    X_res, y_res = smote_nc.fit_resample(X_train, y_train)

    # Model training
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_res, y_res)
    
    return model, encoder, X_test, y_test

# Load model, encoder, and test data
model, encoder, X_test, y_test = generate_and_train()

# ======================
# PREDICTION SECTION
# ======================
if st.sidebar.button("Predict Failure"):
    # Prepare input data
    mat_props = materials_db[material]
    
    input_data = {
        'type': mat_props["type"],
        'yield': mat_props["yield"],
        'E': mat_props["E"],
        'r': mat_props["r"],
        'thickness': thickness,
        'strain': strain/100,  # Convert % to decimal
        'stress': stress,
        'compression': compression,
        'friction': friction
    }
    
    # Feature engineering
    input_data['stiffness_ratio'] = input_data['E'] / input_data['yield']
    input_data['slenderness'] = input_data['thickness'] / (input_data['strain'] if input_data['strain'] != 0 else 0.0001)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode
    X_encoded = encoder.transform(input_df[['type']])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['type']))
    X_processed = pd.concat([X_encoded_df, input_df.drop('type', axis=1)], axis=1)
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Failure Mode", prediction)
        
        # Show probabilities
        st.write("Probability Distribution:")
        proba_df = pd.DataFrame({
            'Failure Mode': model.classes_,
            'Probability': proba
        }).sort_values('Probability', ascending=False)
        st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))
    
    with col2:
        # Show input parameters
        st.write("Input Parameters:")
        st.json({
            "Material": material,
            "Thickness (mm)": thickness,
            "Strain (%)": strain,
            "Stress (MPa)": stress,
            "Compression Ratio": compression,
            "Friction Coefficient": friction
        })
    
    # Show feature importance plot
    st.subheader("Key Influencing Factors")
    importances = pd.DataFrame({
        'Feature': X_processed.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(importances['Feature'], importances['Importance'])
    ax.set_title('Top Influencing Factors for This Prediction')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

# ======================
# DATA EXPLORATION SECTION
# ======================
if st.checkbox("Show Material Database"):
    st.subheader("Material Properties Database")
    db_df = pd.DataFrame.from_dict(materials_db, orient='index')
    st.dataframe(db_df)

if st.checkbox("Show Model Performance Metrics"):
    st.subheader("Model Performance")
    st.write("""
    The model was trained on synthetic data with the following performance:
    """)
    
    # Generate sample evaluation (in a real app, you'd cache this)
    y_pred = model.predict(X_test)
    
    st.text(classification_report(y_test, y_pred))