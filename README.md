# Fraud_detection

# 🚀 Fraud Detection using Logistic Regression and XGBoost

Welcome to this project!  
In this repository, we build a **Fraud Detection Model** for a financial company using machine learning.

The goal is to **predict fraudulent transactions** and **create an actionable business plan** based on model insights.

---

## 📚 Project Overview

- **Dataset**: Real-world transactional data (6,362,620 rows × 10 columns).
- **Problem**: Identify whether a transaction is fraudulent (`is_fraud = 1`) or not (`is_fraud = 0`).
- **Approach**:  
  - Clean and explore the dataset.
  - Train and evaluate two models:
    - Logistic Regression
    - XGBoost Classifier
  - Compare results and suggest business actions based on findings.

---

## 🛠️ Technologies Used

- Python 3
- Jupyter Notebook
- Pandas, NumPy (data handling)
- Matplotlib, Seaborn (data visualization)
- Scikit-learn (machine learning models and preprocessing)
- XGBoost (advanced model)

---

## 📈 Project Pipeline (Step-by-Step)

1. **Load Data**  
   Import the dataset and inspect basic structure (rows, columns, types).

2. **Data Cleaning**  
   - Handle missing values.
   - Remove duplicate records if any.

3. **Exploratory Data Analysis (EDA)**  
   - Check fraud vs non-fraud distribution.
   - Visualize patterns using graphs like boxplots.

4. **Preprocessing**  
   - Separate features (X) and target (y).
   - Handle categorical variables with one-hot encoding.
   - Scale numerical features.
   - Split into training and testing sets.

5. **Model Building**
   - **Logistic Regression**:  
     A simple and interpretable model that acts as a good baseline.
   - **XGBoost Classifier**:  
     A powerful, optimized gradient boosting model that handles imbalanced data well.

6. **Evaluation**
   - Use Classification Report (Precision, Recall, F1-Score).
   - Calculate ROC-AUC Score for model comparison.

7. **Feature Importance**
   - Identify top features influencing fraud detection for both models.

8. **Actionable Business Plan**
   - Based on model insights, suggest practical ways to reduce fraud.

---

## 🔥 Key Highlights

- Logistic Regression is **simple to explain** to business teams.
- XGBoost offers **higher accuracy** and **better handling of imbalanced datasets**.
- Focus is not just on accuracy, but also **Recall** (catching fraud cases).
- **Feature importance** explains where the company should focus its attention.

---

## 📝 How to Run This Project

1. Clone this repository:
   ```bash
   git clone https://github.com/Mighty2Skiddie/fraud-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd fraud-detection
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run the cells step-by-step and follow the instructions.

---

## 📊 Final Results

| Model                | ROC-AUC Score | Key Notes                            |
|----------------------|---------------|--------------------------------------|
| Logistic Regression  | Moderate      | Easy to understand, fast training    |
| XGBoost Classifier   | High           | Best for real-world fraud detection  |

---

## 🧠 Future Improvements

- Try ensemble techniques like Random Forests or LightGBM.
- Tune hyperparameters using GridSearchCV or Optuna.
- Deploy the final model using Flask/FastAPI as a real-time fraud detection API.

---

## 🤝 Let's Connect!

If you like this project or want to collaborate, feel free to reach out!  
**Happy Learning and Building 🚀**

---

# ⭐ Thank you for visiting this project!
