# 🛡️ AI & The Future of Work: A Predictive & Cluster Analysis

> **Strategic Machine Learning pipeline** investigating the impact of AI on global job markets (2024–2030). This project goes beyond simple prediction to uncover the underlying structures of automation risk.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📋 Executive Overview
As AI reshapes the global economy, understanding which roles are most vulnerable is critical. This project analyzes a dataset of global occupations through a dual lens: **Unsupervised Learning** (to find hidden risk groups) and **Supervised Learning** (to test the predictability of AI impact).

---

## 🚀 Strategic Pipeline

### 1. Exploratory Data Analysis (EDA) 📊
* **Counter-Intuitive Discovery:** We identified a **near-zero correlation (~0.01)** between salary/experience and automation risk. 
* **The "Safety Myth":** Contrary to popular belief, high-paying roles are not inherently safer from AI disruption; the nature of tasks (routine vs. strategic) is the true driver.



### 2. Unsupervised Learning: Defining Risk Profiles 🧩
Using **K-Means Clustering (k=10)**, we moved beyond job titles to identify behavioral clusters:
* **Resilient Strategists:** High-creativity, low-automation risk profiles.
* **Transitioning Tech Profiles:** High-salary roles that face significant automation due to routine-based technical tasks.
* **Vulnerable Operational Roles:** High-risk, high-routine clusters requiring urgent reskilling.



### 3. Supervised Learning: The Prediction Challenge 🤖
We implemented a **Random Forest Classifier** (optimized via GridSearchCV).
* **The Result:** ~33% Accuracy.
* **Technical Diagnosis:** The low accuracy empirically proves that traditional demographic/financial metrics are **insufficient predictors** for AI impact. This finding shifts the focus toward qualitative task-analysis rather than quantitative job-metrics.

### 4. Ethical Deployment (Streamlit) 🌐
A web interface was developed to visualize these risks, featuring a **Transparency & Ethics Layer**. It informs users that AI impact is non-linear and cannot be predicted by salary alone.

---

## 🧠 Key Strategic Insights
1. **Universal Risk:** Automation impact is sector-agnostic, affecting Healthcare and Transportation with similar intensity.
2. **Reskilling Urgency:** The identification of "Transitioning Profiles" highlights where corporate training budget should be prioritized.
3. **Data Integrity:** Responsible AI means acknowledging when data lacks predictive power—transparency over "perfect" but false metrics.

---

## ⚙️ Installation & Deployment
```bash
# Clone and install
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn

# Launch the dashboard
streamlit run app.py

