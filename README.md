<div align="center">

# 🦺 AI-Powered Workplace Incident Outcome Prediction
### Predicting Workplace Injury Severity Using Machine Learning & NLP on OSHA ITA Case Detail Data

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end Machine Learning project that predicts workplace incident outcomes using structured OSHA records and unstructured incident narratives. The solution combines data preprocessing, feature engineering, Natural Language Processing, and ensemble learning to support proactive workplace safety decisions.

</div>

---

# 📑 Table of Contents

- Project Overview
- Business Problem
- Objectives
- Dataset
- Tech Stack
- Project Workflow
- Repository Structure
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning Pipeline
- Models Implemented
- Model Performance
- Key Business Insights
- Installation
- Running the Project
- Future Improvements
- Contributors
- Acknowledgements

---

# 📌 Project Overview

Workplace injuries continue to impose significant operational, financial, and human costs across industries. Although organizations collect extensive incident records, these datasets are often used only for compliance reporting rather than predictive decision-making.

This project develops an AI-powered classification system capable of predicting workplace incident outcomes using historical OSHA Injury Tracking Application (ITA) Case Detail Data. By integrating structured organizational information with Natural Language Processing (NLP) features extracted from incident narratives, the model helps organizations identify high-risk incidents before they escalate.

The project follows a complete Data Science lifecycle including business understanding, exploratory data analysis, data preprocessing, feature engineering, machine learning, model evaluation, and business interpretation.

---

# 🎯 Business Problem

Traditional workplace safety management is reactive.

Organizations typically investigate incidents only after they occur, resulting in:

- Increased workplace injuries
- Higher compensation costs
- Productivity loss
- Regulatory penalties
- Delayed safety interventions

This project addresses these challenges by building a predictive system that estimates the likely outcome of a workplace incident using historical OSHA data.

---

# 🎯 Objectives

- Predict workplace incident outcomes into four severity classes
- Analyze industry-level safety trends
- Extract insights from structured and textual OSHA data
- Compare multiple Machine Learning algorithms
- Select the most suitable production-ready model
- Support proactive workplace safety planning

---

# 📂 Dataset

**Source**

OSHA Injury Tracking Application (ITA) Case Detail Data

### Dataset Statistics

| Metric | Value |
|---------|--------|
| Original Records | 686,806 |
| Final Records | 683,962 |
| Original Features | 39 |
| Engineered Features | 586 |
| Text Columns | 7 |
| Target Classes | 4 |

### Target Classes

- Days Away From Work
- Job Transfer / Restriction
- Other Recordable Case
- Death (Rare Class)

---

# 🛠 Tech Stack

## Programming

- Python

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-Learn
- LightGBM
- XGBoost
- Random Forest
- Logistic Regression

## NLP

- TF-IDF Vectorization

## Feature Engineering

- Frequency Encoding
- One-Hot Encoding
- Log Transformation
- Geospatial Encoding
- Temporal Feature Extraction

---

# 🏗 Project Workflow

```
OSHA Dataset
      │
      ▼
Business Understanding
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Natural Language Processing
(TF-IDF)
      │
      ▼
Feature Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Business Insights
```

---

# 📁 Repository Structure

```
AI-Workplace-Incident-Prediction
│
├── data/
│
├── notebooks/
│   └── Final_Capstone_GRP_5_2026.ipynb
│
├── reports/
│   ├── Interim_Report.pdf
│   ├── Final_Report.pdf
│   ├── Synopsis.pdf
│
├── presentation/
│   └── Capstone_Presentation.pptx
│
├── images/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# 📊 Exploratory Data Analysis

The project includes extensive exploratory analysis to understand workplace injury patterns.

### Key Findings

- California recorded the highest number of workplace incidents.
- Transportation and Healthcare sectors exhibited the highest safety risk.
- Physical injuries accounted for the majority of incidents.
- Death cases represented only **0.03%** of the dataset, creating severe class imbalance.
- Most incidents occurred during regular working hours.
- Employee count and total hours worked showed strong positive correlation.

---

# ⚙ Data Preprocessing

The preprocessing pipeline included:

- Duplicate removal
- Missing value treatment
- Identifier removal
- Outlier capping
- Datetime conversion
- Data type correction
- Feature scaling
- Class balancing

---

# 🧠 Feature Engineering

The project generated 586 predictive features.

### Engineered Features

- Incident Day
- Work Shift
- Season
- Weekend Indicator
- Hours After Shift Start
- Industry Sector
- SOC Group
- Working Hour
- Size Category

### Encoding Techniques

- Frequency Encoding
- One-Hot Encoding
- Label Encoding

### NLP Features

Incident narratives from seven OSHA text columns were combined and transformed using TF-IDF Vectorization to generate 500 textual features.

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Evaluation Metrics

- Accuracy
- ROC-AUC
- Matthews Correlation Coefficient
- Cohen's Kappa
- Precision
- Recall
- F1 Score

---

# 🏆 Model Performance

| Model | Accuracy | ROC-AUC | MCC |
|---------|----------|---------|------|
| Logistic Regression | 71.7% | 0.892 | 0.580 |
| Random Forest | 77.0% | 0.919 | 0.656 |
| XGBoost | 83.1% | 0.952 | 0.746 |
| ⭐ LightGBM | **82.7%** | **0.953** | **0.742** |

---

# ⭐ Final Model

## LightGBM Classifier

### Performance

- Accuracy: **82.7%**
- ROC-AUC: **0.953**
- MCC: **0.742**
- Cohen's Kappa: **0.741**

The LightGBM model was selected due to its strong overall performance, efficient handling of high-dimensional sparse features, and improved detection of the rare "Death" class compared to other evaluated models.

---

# 📈 Key Business Insights

- Transportation and Healthcare industries represent the highest workplace safety risk.
- Physical injuries dominate OSHA incident records.
- Seasonal trends indicate increased workplace incidents during mid-year months.
- NLP-derived incident narratives significantly improve predictive performance.
- Machine Learning enables proactive safety planning instead of reactive reporting.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Workplace-Incident-Prediction.git
```

Move into the project

```bash
cd AI-Workplace-Incident-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Notebook

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
Final_Capstone_GRP_5_2026.ipynb
```

Run all cells sequentially.

---

# 💡 Future Improvements

- Deploy the model using Streamlit or FastAPI
- Integrate transformer-based NLP models (BERT)
- Experiment with SMOTE and advanced imbalance learning techniques
- Real-time OSHA incident prediction dashboard
- Explainable AI using SHAP
- Cloud deployment using Azure or AWS
- Continuous model retraining with new OSHA records

---

# 📊 Project Presentation

The complete project presentation explains the business problem, data analysis, feature engineering, model development, evaluation, and business recommendations.

📥 **Download Presentation**

➡️ [Capstone Presentation (PPTX)](./capstone_final2.pptx)

---

## 📚 Additional Project Documents

- 📄 [Final Project Report](./Capstone_Final_Report.docx)
- 📓 [Jupyter Notebook](./Final_Capstone_GRP_5_2026.ipynb)

# 👥 Contributors

- **Vikash Basfore**
- Kavya Radheshwar
- Mumtaz Khan
- Mangasamudram Lokeswari
- Divyasree C.
- R. Haridharan

---

# 🙏 Acknowledgements

This project was successfully completed as part of the **PGP in Data Science with Specialization in Generative AI** Capstone Project.

Special thanks to:

- **Great Learning** for providing mentorship, guidance, and the capstone project framework.
- **Occupational Safety and Health Administration (OSHA)** for making the Injury Tracking Application (ITA) Case Detail Dataset publicly available.
- The **Scikit-learn**, **LightGBM**, **XGBoost**, **Pandas**, and **NumPy** open-source communities for developing powerful machine learning and data analysis libraries.
- My capstone teammates for their collaboration, discussions, and contributions throughout the project.
- All open-source contributors whose tools and documentation made this project possible.

This project was developed solely for educational and research purposes to demonstrate the practical application of Machine Learning and Natural Language Processing in workplace safety analytics.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

</div>
