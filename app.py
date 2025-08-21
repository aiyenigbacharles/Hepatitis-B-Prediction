# ==============================================================================
# 1. IMPORTING NECESSARY LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================================================================
# 2. LOADING THE SAVED MODEL AND IMPUTERS
# ==============================================================================

# This function loads the trained model and preprocessing objects (imputers)
# The @st.cache_resource decorator ensures this is loaded only once, improving performance.
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Load the dictionary containing the model and imputers
        model_data = joblib.load('hepatitis_prediction_model.joblib')
        return model_data
    except FileNotFoundError:
        st.error("Model file not found. Please run the `train_model.py` script first.")
        return None

# Load the model data
model_data = load_model()

# If model loading fails, stop the app
if model_data is None:
    st.stop()

# Extract the model and imputers from the loaded dictionary
model = model_data['model']
imputers = model_data['imputers']
median_imputer = imputers['median']
mode_imputer = imputers['mode']
numerical_cols = imputers['num_cols']
categorical_cols = imputers['cat_cols']


# 3. SETTING UP THE STREAMLIT USER INTERFACE

# Set the title and a description for the web app
st.title("🩺 Hepatitis B Prediction System")
st.write("""
This app uses an Ensemble Machine Learning model to predict the outcome (Negative or Positive) for a patient with Hepatitis B. 
Kindly provide the patient's data in the sidebar to get a prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Data Input")

# Create a function to collect user inputs
def user_input_features():
    # Create sliders and select boxes for each feature
    # The values are set to reasonable defaults or common values from the dataset
    age = st.sidebar.slider('Age', 10, 80, 40)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    steroid = st.sidebar.selectbox('Steroid Treatment', ('No', 'Yes'))
    antivirals = st.sidebar.selectbox('Antivirals', ('No', 'Yes'))
    fatigue = st.sidebar.selectbox('Fatigue', ('No', 'Yes'))
    malaise = st.sidebar.selectbox('Malaise', ('No', 'Yes'))
    anorexia = st.sidebar.selectbox('Anorexia', ('No', 'Yes'))
    liver_big = st.sidebar.selectbox('Liver Big', ('No', 'Yes'))
    liver_firm = st.sidebar.selectbox('Liver Firm', ('No', 'Yes'))
    spleen_palpable = st.sidebar.selectbox('Spleen Palpable', ('No', 'Yes'))
    spiders = st.sidebar.selectbox('Spiders', ('No', 'Yes'))
    ascites = st.sidebar.selectbox('Ascites', ('No', 'Yes'))
    varices = st.sidebar.selectbox('Varices', ('No', 'Yes'))
    bilirubin = st.sidebar.slider('Bilirubin (mg/dL)', 0.1, 8.0, 1.0, 0.1)
    alk_phosphate = st.sidebar.slider('Alkaline Phosphate (IU/L)', 20, 300, 100)
    sgot = st.sidebar.slider('SGOT (IU/L)', 10, 650, 100)
    albumin = st.sidebar.slider('Albumin (g/dL)', 2.0, 6.0, 4.0, 0.1)
    protime = st.sidebar.slider('Prothrombin Time (seconds)', 0, 100, 50)
    histology = st.sidebar.selectbox('Histology', ('No', 'Yes'))
    
    # Convert categorical inputs from text ('Yes'/'No') to numbers (1/2 or 1/0)
    # The original dataset uses 1 for 'No' and 2 for 'Yes' for most binary features. Let's stick to that.
    # Sex: Male=1, Female=2
    sex_val = 1 if sex == 'Male' else 2
    
    # Map Yes/No to the dataset's format (usually 1=No, 2=Yes)
    def map_yes_no(val):
        return 2 if val == 'Yes' else 1

    # Create a dictionary to hold the input data
    data = {
        'Age': age,
        'Sex': sex_val,
        'Steroid': map_yes_no(steroid),
        'Antivirals': map_yes_no(antivirals),
        'Fatigue': map_yes_no(fatigue),
        'Malaise': map_yes_no(malaise),
        'Anorexia': map_yes_no(anorexia),
        'LiverBig': map_yes_no(liver_big),
        'LiverFirm': map_yes_no(liver_firm),
        'SpleenPalpable': map_yes_no(spleen_palpable),
        'Spiders': map_yes_no(spiders),
        'Ascites': map_yes_no(ascites),
        'Varices': map_yes_no(varices),
        'Bilirubin': bilirubin,
        'AlkPhosphate': alk_phosphate,
        'SGOT': sgot,
        'Albumin': albumin,
        'Protime': protime,
        'Histology': map_yes_no(histology)
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get the user input
input_df = user_input_features()

# 4. PREDICTION AND DISPLAYING THE OUTPUT

# Display the user's input data in the main area
st.subheader('Patient Input Parameters')
st.write(input_df)

# Create a button to trigger the prediction
if st.button('Predict Outcome'):
    # The user input is a single row, so we don't need to impute missing values.
    # However, we must ensure the column order is the same as the training data.
    # The column names used in the DataFrame creation above match the original dataset.

    # Make the prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the result
    st.subheader('Prediction Result')
    
    # Get the outcome string ('Live' or 'Die') based on the prediction (0 or 1)
    outcome = 'NEGATIVE' if prediction[0] == 1 else 'POSITIVE'
    st.write(f"The model predicts that the patient's Hepatitis-B result is **{outcome}**.")

    # Display the prediction probability
    st.subheader('Prediction Probability')
    probability_df = pd.DataFrame({
        'Positive Probability': [f"{prediction_proba[0][0]*100:.2f}%"],
        'Negative Probability': [f"{prediction_proba[0][1]*100:.2f}%"]
    })
    st.write(probability_df)