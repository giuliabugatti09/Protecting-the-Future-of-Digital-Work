# 🛡️🤖 Protecting the Future of Digital Work

**A Machine Learning pipeline analyzing the impact of AI on global occupations through clustering and predictive modeling.**

📌 **Global Solution 2025.2 – Machine Learning & Modeling**

---

## 📋 Overview

This project leverages **Machine Learning** to analyze global job market data (2024–2030), aiming to understand and predict the impact of **Artificial Intelligence on occupations**.

Rather than relying solely on prediction, the solution emphasizes **exploratory analysis and clustering**, uncovering hidden patterns, risk profiles, and career trajectories in the evolving digital workforce.

---

## 🚀 Project Pipeline

The solution was developed in **four strategic stages**:

---

### 1️⃣ Exploratory Data Analysis (EDA) 📊

We explored relationships between key market variables such as:

* Salary
* Experience
* Location
* Automation Risk

🔍 **Key Discovery**
A near-zero correlation (~0.01) was found between numerical features and automation risk, **challenging the assumption** that higher salaries automatically imply lower automation risk.

---

### 2️⃣ Unsupervised Learning (Clustering) 🧩

After confirming that salary-based prediction was ineffective, we applied **K-Means Clustering (k = 10)** to group professions by similarity.

✅ **Result**
Distinct professional profiles emerged, including:

* **Resilient Strategists** → Low risk, creative and strategic roles
* **Vulnerable Roles** → High risk, routine-based occupations

This step proved to be one of the most insightful parts of the project.

---

### 3️⃣ Supervised Learning (Prediction) 🤖

A **Random Forest Classifier**, optimized with **GridSearchCV**, was trained to predict the **AI Impact Level**.

📈 **Outcome**

* Test Accuracy: ~33%

🧪 **Diagnosis**
Since the target variable has three classes (High, Moderate, Low), this accuracy is equivalent to random guessing.
This result empirically validated the EDA findings: **the available features lack sufficient predictive signal** for this target.

---

### 4️⃣ Interactive Interface (Streamlit) 🌐

A **Streamlit web application** allows users to:

* Input job characteristics
* Receive an AI impact analysis

⚠️ The interface includes a **clear transparency warning**, highlighting model limitations to promote **ethical and responsible use**.

---

## 📂 Project Structure

```
├── app.py                   # Streamlit Application (Main Interface)
├── analysis_notebook.ipynb  # EDA, K-Means Clustering & Model Training
├── requirements.txt         # Project Dependencies
├── modelo_final.pkl         # Trained Random Forest Model
├── preprocessor.pkl         # Data Preprocessing Pipeline
├── label_encoder.pkl        # Target Variable Encoder
└── README.md                # Project Documentation
```

---

## ⚙️ Installation & Usage

### 🔹 Clone the repository (or download the files)

Make sure all `.pkl` files and `app.py` are located in the same directory.

### 🔹 Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

### 🔹 Run the application

```bash
streamlit run app.py
```

---

## 🧠 Key Insights & Conclusions

* **Risk Is Generalized**
  Automation impacts both **Transportation** and **Healthcare** sectors at similar levels.

* **The Plot Twist**
  Salary and experience are **not reliable predictors** of AI safety.
  The **nature of the task** (routine vs. creative) plays a far more critical role.

* **Cluster Profiles Matter**
  The most valuable insights came from clustering, especially identifying
  **“Transitioning Profiles”** (High Tech, High Risk), which demand **urgent reskilling strategies**.

---

## ⚠️ Ethical & Transparency Note

This tool uses **public job market data** strictly for educational purposes.

During validation, we identified that the dataset has **low predictive power** for the target variable *“AI Impact Level”*.
As a result:

* Predictions are **demonstrative**, not prescriptive
* The application should **not be used as the sole basis for career decisions**

Transparency and responsible AI usage were core design principles of this project.

---

## 👨‍💻 Authors

**Global Solution Team – 1TIAP**

✍️ Giulia Bugatti

