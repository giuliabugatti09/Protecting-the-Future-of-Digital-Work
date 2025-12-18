import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================================
# 1. CARREGAMENTO DOS ARTEFATOS
# ===============================================
try:
    # Carrega os arquivos gerados pelo notebook
    model = joblib.load('modelo_final.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Recupera os nomes das features esperadas pelo preprocessor
    numerical_features = preprocessor.transformers_[0][2] 
    categorical_features = preprocessor.transformers_[1][2]
    
except FileNotFoundError:
    st.error("🚨 Erro Crítico: Arquivos .pkl não encontrados.")
    st.warning("Certifique-se de que 'modelo_final.pkl', 'preprocessor.pkl' e 'label_encoder.pkl' estão na mesma pasta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

# ===============================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ===============================================
st.set_page_config(
    page_title="AI Impact Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Previsor de Impacto da IA nas Profissões")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para prever o nível de impacto da Inteligência Artificial em diferentes ocupações.
***
""")

# ===============================================
# 3. DEFINIÇÃO DE OPÇÕES (VALIDADAS PELO DATASET)
# ===============================================
options_industry = [
    'Education', 'Entertainment', 'Finance', 'Healthcare', 
    'IT', 'Manufacturing', 'Retail', 'Transportation'
]

options_job_status = [
    'Decreasing', 'Increasing'
]

options_education = [
    'Associate Degree', 'Bachelor’s Degree', 'High School', 'Master’s Degree', 'PhD'
]

options_location = [
    'Australia', 'Brazil', 'Canada', 'China', 'Germany', 'India', 'UK', 'USA'
]

# ===============================================
# 4. FORMULÁRIO DE ENTRADA
# ===============================================

col1, col2 = st.columns(2)

with col1:
    st.header("📊 Características Quantitativas")
    
    # Adicionei o argumento 'key' em todos os inputs para evitar o erro de ID duplicado
    median_salary = st.number_input(
        label=numerical_features[0],
        min_value=0, 
        value=50000,
        step=1000,
        key="salary_input"
    )
    experience = st.number_input(
        label=numerical_features[1],
        min_value=0, 
        value=5,
        key="experience_input"
    )
    job_openings_2024 = st.number_input(
        label=numerical_features[2],
        min_value=0, 
        value=1000,
        key="openings24_input"
    )
    projected_openings_2030 = st.number_input(
        label=numerical_features[3],
        min_value=0, 
        value=1200,
        key="openings30_input"
    )
    remote_ratio = st.slider(
        label=numerical_features[4],
        min_value=0.0, 
        max_value=100.0, 
        value=20.0, 
        step=0.1,
        key="remote_input"
    )
    gender_diversity = st.slider(
        label=numerical_features[5],
        min_value=0.0, 
        max_value=100.0, 
        value=40.0, 
        step=0.1,
        key="gender_input"
    )
    automation_risk = st.slider(
        label=numerical_features[6],
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=0.1,
        key="risk_input"
    )

with col2:
    st.header("🏷️ Características Categóricas")
    
    industry = st.selectbox(
        label=categorical_features[0],
        options=options_industry,
        key="industry_input"
    )
    job_status = st.selectbox(
        label=categorical_features[1],
        options=options_job_status,
        key="status_input"
    )
    education = st.selectbox(
        label=categorical_features[2],
        options=options_education,
        key="education_input"
    )
    location = st.selectbox(
        label=categorical_features[3],
        options=options_location,
        key="location_input"
    )

st.markdown("---")

# ===============================================
# 5. BOTÃO DE PREVISÃO E LÓGICA
# ===============================================

if st.button("🔍 Prever Impacto da IA", type="primary", key="predict_btn"):
    
    # Coletar os dados em um dicionário
    input_data = {
        numerical_features[0]: [median_salary],
        numerical_features[1]: [experience],
        numerical_features[2]: [job_openings_2024],
        numerical_features[3]: [projected_openings_2030],
        numerical_features[4]: [remote_ratio],
        numerical_features[5]: [gender_diversity],
        numerical_features[6]: [automation_risk],
        
        categorical_features[0]: [industry],
        categorical_features[1]: [job_status],
        categorical_features[2]: [education],
        categorical_features[3]: [location],
    }
    
    # Criar DataFrame
    input_df = pd.DataFrame(input_data)
    
    try:
        # 1. Processar os dados
        input_processed = preprocessor.transform(input_df)
        
        # 2. Fazer a previsão
        prediction_encoded = model.predict(input_processed)
        
        # 3. Decodificar a previsão
        prediction_text = le.inverse_transform(prediction_encoded)
        result = prediction_text[0]
        
        # 4. Exibir Resultado
        st.subheader("Resultado da Análise:")
        
        if result == 'High':
            st.error(f"🚨 Nível de Impacto Previsto: **{result}**")
        elif result == 'Moderate':
            st.warning(f"⚠️ Nível de Impacto Previsto: **{result}**")
        else:
            st.success(f"✅ Nível de Impacto Previsto: **{result}**")
            
        # 5. AVISO ÉTICO E TÉCNICO
        st.markdown("---")
        st.info(
            """
            **ℹ️ Nota de Transparência e Ética:**
            
            Este modelo foi treinado utilizando dados públicos de mercado. Durante a fase de validação técnica, 
            observou-se que as variáveis disponíveis (Salário, Localização, Experiência) possuem baixa correlação 
            linear com o impacto da IA, resultando em uma acurácia preditiva limitada (~33%).
            
            Portanto, esta ferramenta deve ser utilizada para fins educacionais e de demonstração técnica do pipeline 
            de Machine Learning, e **não como base única para decisões reais de carreira**.
            """
        )

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")