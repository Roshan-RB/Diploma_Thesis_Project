
# 📌 Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-orange)](https://en.wikipedia.org/wiki/Supervised_learning)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MLP%20%7C%20CNN-green)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-red)](#)

## 📖 Overview
This repository contains code and notebooks for **Customer Churn Prediction**, where the goal is to forecast whether a subscriber will **renew their membership or churn** based on historical data. The dataset used in this project comes from **KKBOX**, a leading music streaming platform in Asia.

## 🎯 Objective
- Predict customer churn based on transactional, demographic, and activity data.
- Use various **machine learning (Logistic Regression, Random Forest, XGB Classifier)** and **deep learning (MLP, CNN)** models.
- Identify key **features** that influence churn behavior.

## 📁 Repository Structure
```
├── notebooks/                  # Jupyter notebooks for EDA, preprocessing, modeling
│   ├── 1_EDA.ipynb             # Exploratory Data Analysis
│   ├── 2_Data_Preprocessing.ipynb # Data Cleaning & Feature Engineering
│   ├── 3_Model_Training.ipynb  # ML Model Training (Logistic Regression, RF, XGB)
│   ├── 4_Advanced_Modeling.ipynb # Deep Learning Models (MLP, CNN)
│   ├── 5_Evaluation.ipynb      # Model Performance Evaluation
├── data/                       # Sample dataset (if applicable)
├── scripts/                    # Python scripts for data processing & model training
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── LICENSE                     # Open-source license (if applicable)
```

## 🗂️ Dataset
The data is sourced from **[KKBOX Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)** and consists of:
1. **train_v2.csv** - Churn labels for training (970K records).
2. **transactions_v2.csv** - Subscription & payment history (1.4M records).
3. **user_logs_v2.csv** - User activity data (18.3M records).
4. **members_v3.csv** - Demographic details of users (6.7M records).

### 🏆 Evaluation Metric
- **Log Loss** (Binary Classification)
- **F1-Score**

## 📊 Exploratory Data Analysis (EDA)
Key insights:
- **City 1** has the highest number of users, but **City 21** has the highest churn rate (~14.7%).
- **Subscription plans of 30 days** are the most common.
- **Auto-renewal significantly reduces churn**.
- **Cancellation behavior** is a strong churn indicator.

## 🏗️ Model Training & Results
| Model                | Log Loss  | Accuracy  | Key Findings |
|----------------------|----------|-----------|--------------|
| Logistic Regression (L1/L2) | 0.153 | ~91%  | Baseline Model |
| Random Forest       | **0.104** | **94.5%** | Best ML model |
| XGB Classifier      | 0.120 | 93.4% | Tuning required |
| MLP (Deep Learning) | 0.123 | 94.5% | Overfitting issue |
| CNN (Deep Learning) | 0.166 | 93.4% | Not the best fit |

- **Random Forest performed best** with the lowest log loss.
- **Deep Learning models (MLP, CNN) did not outperform traditional models** in this case.

## 🛠️ Installation & Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks:
   ```bash
   jupyter notebook
   ```

## 🚀 Deployment Plan (Future Work)
A **Streamlit web app** will be created where users can input customer details to predict churn probability. Stay tuned! 

---

## 📚 References
- [KKBOX Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
- [Stanford CS229 - Customer Churn](http://cs229.stanford.edu/proj2017/final-posters/5147439.pdf)
- [Oracle - Churn Prediction](https://blogs.oracle.com/datascience/introduction-to-churn-prediction-in-python)

---

## 🙌 Acknowledgment
This project was developed as part of the **AI & ML Diploma** under **Applied Roots with University of Hyderabad**, guided by **Mentor: Harichandhana K**.
