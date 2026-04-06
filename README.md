# 🛡️ AI & The Future of Work: A Predictive & Cluster Analysis

> **Strategic Machine Learning pipeline** investigating the impact of AI on global job markets (2024–2030). This project goes beyond simple prediction to uncover the underlying structures of automation risk and professional resilience.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📋 Executive Summary
As Artificial Intelligence reshapes the global economy, understanding which professional roles are most vulnerable is critical. This project analyzes global occupations through a dual-lens approach: **Unsupervised Learning** (to identify hidden risk archetypes) and **Supervised Learning** (to test the predictability of AI impact based on traditional metrics).

---

## 🚀 The Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA) 📊
* **Counter-Intuitive Discovery:** We identified a **near-zero correlation (~0.01)** between salary/experience and automation risk. 
* **The "Safety Myth":** Contrary to popular belief, high-paying roles are not inherently safer from AI disruption; the nature of tasks (routine vs. strategic) is the true driver of automation risk.

### 2. Unsupervised Learning: Occupational Archetypes 🧩
Using **K-Means Clustering (k=10)** and **PCA Visualization**, we moved beyond job titles to identify behavioral clusters:
* **The Resilient Strategists:** High-creativity, low-automation risk profiles (Avg. Risk: 29%).
* **The Vulnerable Operationals:** Roles focused on repetitive tasks, regardless of salary level (Avg. Risk: 74%).
* **The Technicians in Transition:** High-demand technical roles facing a "reskilling race" against evolving AI capabilities.

### 3. Supervised Learning: The Prediction Challenge 🤖
We implemented an **XGBoost Classifier** optimized via **GridSearchCV**.
* **Diagnostic Result:** ~33% Accuracy (Matching the random baseline for 3 classes).
* **Technical Conclusion:** This empirical failure proves that traditional demographic/financial metrics are **insufficient predictors** for AI impact. It highlights the need for qualitative task-analysis in future workforce modeling.

---

## 🧠 Key Strategic Insights
1. **Sector-Agnostic Risk:** Automation intensity is similar across Healthcare, Education, and Transportation, proving that AI impact is now generalized.
2. **Reskilling Priority:** The identification of "Transitioning Profiles" indicates where corporate training budgets should be strategically prioritized.
3. **Ethical Transparency:** Responsible AI involves acknowledging when data lacks predictive power—prioritizing transparency over "perfect" but misleading metrics.

---

## 📂 Project Structure
```text
├── models/
│   ├── ai_impact_model.pkl   # Trained XGBoost Pipeline
│   └── label_encoder.pkl     # Target class decoder
├── notebooks/
│   └── ai_impact_analysis.ipynb  # Full EDA and Clustering logic
├── data/
│   └── ai_job_trends_dataset (1).csv # Dataset
├── app.py                    # Streamlit Interactive Dashboard
└── requirements.txt          # Project dependencies
└── LICENSE         # License Project

```

---

## ⚙️ Installation & Deployment
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/giuliabugatti09/AI-Impact-Professions.git](https://github.com/giuliabugatti09/AI-Impact-Professions.git)
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

---
**Developed by Giulia Bugatti** *Artificial Intelligence Student at FIAP* | Expected Graduation: **Dec 2026**
```
