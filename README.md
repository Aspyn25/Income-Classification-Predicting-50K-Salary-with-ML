# Income Classification: Predicting $50K+ Salary with ML

## 📌 Project Overview
This project aims to predict whether a person earns more than **$50,000 per year** using machine learning models. Various classification algorithms were trained and evaluated to determine the most effective model.

✔ **Binary Classification Problem** → Predict if income is **≤50K or >50K**  
✔ **Comparing Multiple ML Models** → Logistic Regression, Decision Tree, Random Forest, SVM, KNN  
✔ **Evaluating Model Performance** → Accuracy, Precision, Recall, F1-score, ROC-AUC  
✔ **Handling Class Imbalance** → SMOTE (Synthetic Minority Over-sampling Technique) applied  

## 📂 Dataset
- **Source:** UCI Adult Income Dataset
- **Features:** Age, Education, Occupation, Work Hours, etc.
- **Target Variable:** `Income` (Binary: ≤50K or >50K)
- **Preprocessing Steps:**
  - Handling missing values (workclass, occupation, native_country)
  - Encoding categorical features (Label Encoding & One-Hot Encoding)
  - Handling outliers (capital_gain, hours_per_week)
  - Feature Scaling (RobustScaler for linear models)
  - Dealing with class imbalance (SMOTE applied)

## ⚡ Machine Learning Models Used
The following models were trained and compared **before and after applying SMOTE**:

### **Model Performance Before SMOTE**
| Model                  | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 82.0%    | 76.5%     | 69.5%  | 71.5%    | 0.84    |
| K-Nearest Neighbors   | 84.5%    | 76.5%     | 72.9%  | 75.1%    | 0.89    |
| Decision Tree         | 84.0%    | 78.5%     | 73.0%  | 75.5%    | 0.88    |
| Support Vector Machine | 73.0%    | 61.0%     | 61.0%  | 62.5%    | 0.83    |
| Random Forest         | **88.0%** | **81.0%** | **76.0%** | **78.5%** | **0.90** |

### **Model Performance After SMOTE**
| Model                  | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 80.0%    | 80.0%     | 80.0%  | 80.0%    | 0.88    |
| K-Nearest Neighbors   | 88.5%    | 84.5%     | 84.5%  | 85.0%    | 0.91    |
| Decision Tree         | 85.0%    | 84.0%     | 84.5%  | 84.0%    | 0.91    |
| Support Vector Machine | 80.0%    | 79.5%     | 79.5%  | 79.0%    | 0.88    |
| Random Forest         | **90.5%** | **87.0%** | **87.0%** | **87.0%** | **0.95** |

📌 **Best Model After SMOTE:** **Random Forest** achieved the highest accuracy and AUC-ROC score. The recall score improved significantly for most models, showing that SMOTE helped balance class predictions.

## 📊 Feature Importance
The most influential features in predicting high-income individuals:<br>
1️⃣ **Education Level**  
2️⃣ **Hours Worked per Week**  
3️⃣ **Occupation Type**  
4️⃣ **Age**  

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- **Data Visualization**: Matplotlib, Seaborn
- **Jupyter Notebook** for development

## 🚀 How to Run the Project
Clone the repository:  
````bash
git clone https://github.com/your-username/income-classification-ml.git
cd income-classification-ml
````

## 📌 Key Takeaways
- Random Forest performed best in predicting high-income individuals.
-	SMOTE improved class balance and recall for high-income earners.
-	Feature Engineering (log transformation, categorical encoding) significantly improved model performance.

## 📩 Contact

For any questions or collaborations, feel free to reach out!

📧 Email: jeongfree25@gmail.com <br>
🔗 LinkedIn: [https://www.linkedin.com/in/jeonghyun-song-809457327/] <br>
🔗 GitHub: [https://github.com/Aspyn25]
