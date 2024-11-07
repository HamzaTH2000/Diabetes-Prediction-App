import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
loaded_model = pickle.load(open("trained_model.sav", 'rb'))
loaded_scaler = pickle.load(open("scaler.sav", "rb"))

# Title and description
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction App</h1>
    <p style='text-align: center; color: gray;'>Enter the required health details to check the risk of diabetes.</p>
""", unsafe_allow_html=True)


# Display the My name under the title
st.markdown("---")
st.markdown("<p style='text-align: center; color: #1E90FF; font-size: 18px;'>Made with ❤️ by : Hamza Thamlaoui 👨‍💻</p>", unsafe_allow_html=True)



# Add a brief explanation about the app and the context
st.markdown("""
    <h2 style='color: #FF6347;'>About This App</h2>
    <p>This app uses a <strong>Support Vector Machine (SVM)</strong> classifier trained on health data to predict whether an individual is at risk of diabetes. 
    It takes key health metrics like glucose levels, blood pressure, and BMI, and analyzes them to give a risk assessment.</p>

    <p>The model has been trained with a dataset that includes data on pregnancies, insulin levels, and other factors that contribute to diabetes risk. 
    The app provides an easy-to-use interface to help individuals assess their risk quickly.</p>

    <p><strong>Model Accuracy</strong>: The current model has achieved an accuracy of <strong>77.3%</strong> in predicting diabetes based on the available features.</p>
""", unsafe_allow_html=True)

st.divider()

# Organize input fields in columns for better aesthetics
col1, col2, col3 = st.columns(3)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=2, help="Number of times pregnant (optimal: 1-2)")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, step=1, value=120, help="Optimal range: 90-120 mmHg")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, value=0.3, help="Genetic risk of diabetes (optimal: <0.3)")
with col2:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, step=1, value=90, help="Optimal range: 70-99 mg/dL (fasting)")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, step=1, value=20, help="Ideal value for health: ~20 mm")
    age = st.number_input("Age", min_value=0, step=1, value=35, help="Age in years")
with col3:
    insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, step=1, value=15, help="Optimal range: 2-25 μU/mL")
    bmi = st.number_input("Body Mass Index", min_value=0.0, step=0.1, value=22.0, help="Optimal BMI range: 18.5-24.9")

st.divider()

# Gather input data
input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

# Custom button styling using HTML
button_html = """
    <style>
        .stButton>button {
            background-color: #FF6347; 
            color: white; 
            font-size: 18px; 
            padding: 12px 25px; 
            border-radius: 8px; 
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #FF4500;
        }
    </style>
"""
st.markdown(button_html, unsafe_allow_html=True)

# Prediction button
predictbutton = st.button("🔍 Predict Diabetes Risk")

st.divider()

# Prediction and display with color
if predictbutton:  # if predicted button is pressed
    with st.spinner("Analyzing..."):
        # Standardize and predict
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        standardized_data = loaded_scaler.transform(input_data_reshaped)
        prediction = loaded_model.predict(standardized_data)[0]

    # Display result
    if prediction == 0:
        st.success("This person is not diabetic. 👍", icon="✅")
    else:
        st.warning("This person is at risk of diabetes. ⚠️", icon="⚠️")
else:
    st.info("Fill in the details and press the button to predict.", icon="ℹ️")  # Prompt message

# streamlit run streamlt_code.py