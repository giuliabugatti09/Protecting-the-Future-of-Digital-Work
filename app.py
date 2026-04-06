import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Impact Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Definindo os caminhos corretos conforme sua imagem
    model_path = os.path.join('models', 'ai_impact_model.pkl')
    encoder_path = os.path.join('models', 'label_encoder.pkl')

    # Carrega o Pipeline completo (Preprocessor + XGBoost)
    model_pipeline = joblib.load(model_path)
    
    # Carrega o Label Encoder
    le = joblib.load(encoder_path)
    
    return model_pipeline, le

# Agora chamamos apenas os dois
model, le = load_assets()
# --- HEADER ---
st.title("🤖 AI Occupational Impact Predictor")
st.markdown("""
This application uses a Machine Learning pipeline to forecast the intensity of Artificial Intelligence impact across different professions based on market data (2024-2030).
---
""")

# --- INPUT OPTIONS (VALIDATED BY DATASET) ---
options_industry = ['Education', 'Entertainment', 'Finance', 'Healthcare', 'IT', 'Manufacturing', 'Retail', 'Transportation']
options_job_status = ['Decreasing', 'Increasing', 'Stable']
options_education = ["Bachelor's", "Master's", "PhD", "High School", "Associate's"]
options_location = ['Australia', 'Canada', 'Germany', 'India', 'UK', 'USA']

# --- INPUT FORM ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Quantitative Features")
    
    salary = st.number_input("Median Salary (USD)", min_value=0, value=70000, step=5000)
    experience = st.slider("Experience Required (Years)", 0, 30, 5)
    openings_2024 = st.number_input("Job Openings (2024)", min_value=0, value=5000)
    openings_2030 = st.number_input("Projected Openings (2030)", min_value=0, value=5500)
    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 40)
    gender_div = st.slider("Gender Diversity (%)", 0, 100, 50)
    auto_risk = st.slider("Automation Risk (%)", 0, 100, 50)

with col2:
    st.subheader("🏷️ Categorical Features")
    
    industry = st.selectbox("Industry Sector", options_industry)
    status = st.selectbox("Market Status", options_job_status)
    education = st.selectbox("Required Education Level", options_education)
    location = st.selectbox("Geographical Location", options_location)

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("🔍 Run Impact Analysis", type="primary"):
    
    # Building the input DataFrame with exact feature names from the notebook
    input_df = pd.DataFrame({
        'Industry': [industry],
        'Median Salary (USD)': [salary],
        'Experience Required (Years)': [experience],
        'Required Education': [education],
        'Job Openings (2024)': [openings_2024],
        'Projected Openings (2030)': [openings_2030],
        'Remote Work Ratio (%)': [remote_ratio],
        'Automation Risk (%)': [auto_risk],
        'Location': [location],
        'Gender Diversity (%)': [gender_div],
        'Job Status': [status]
    })
    
    try:
        # 1. Pipeline transformation & Prediction
        # Since the model saved is the full Pipeline, it handles scaling/encoding internally
        prediction_encoded = model.predict(input_df)
        result = le.inverse_transform(prediction_encoded)[0]
        
        # 2. Display Result
        st.subheader("Analysis Result:")
        
        if result == 'High':
            st.error(f"### AI Impact Level: **{result}**")
            st.warning("⚠️ High exposure to automation. Reskilling and AI-adaptation are recommended.")
        elif result == 'Moderate':
            st.warning(f"### AI Impact Level: **{result}**")
            st.info("📈 AI will likely augment productivity. Focus on integrating AI tools into daily workflows.")
        else:
            st.success(f"### AI Impact Level: **{result}**")
            st.write("✅ Low direct impact. This role relies heavily on human-centric or strategic skills.")
            
        # 3. TRANSPARENCY & ETHICS NOTE
        st.markdown("---")
        st.info(
            """
            **ℹ️ Transparency & Ethics Note:**
            
            This model was trained on a specific 2024-2030 job market dataset. Technical validation revealed that 
            demographic variables (Salary, Location, Experience) have low linear correlation with AI impact.
            
            This tool is for **educational and demonstration purposes only**, showcasing a Machine Learning 
            pipeline (EDA -> Clustering -> Classification). It should not be the sole basis for real-world career decisions.
            """
        )

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Giulia Bugatti | FIAP AI")