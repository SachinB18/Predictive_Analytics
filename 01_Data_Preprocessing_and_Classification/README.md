
# Income Classification Using Multiple Machine Learning Algorithms

## 📌 Project Overview

This project performs a complete Machine Learning pipeline on the **Adult Census Income Dataset (48,842 records)**.

The objective is to predict whether a person's income exceeds \$50K per year based on demographic and employment attributes.

The project includes:

- Data Preprocessing
- Handling Missing and Noisy Values
- Correlation Analysis (Numerical & Categorical)
- Implementation of Multiple Classifiers
- Confusion Matrix and Performance Metrics
- Comparative Analysis using Graphs
- Final Inference

---

## 📊 Dataset Description

**Dataset Name:** Adult Census Income Dataset  
**Records:** 48,842  
**Target Variable:** income (<=50K, >50K)

### Features:
- Numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- Categorical: workclass, education, marital-status, occupation, relationship, race, gender, native-country

---

## ⚙️ Data Preprocessing Steps

1. Replaced missing values (`?`) with NaN.
2. Filled:
   - Categorical features → Mode
   - Numerical features → Median
3. Removed duplicate records.
4. Encoded categorical variables using One-Hot Encoding.
5. Scaled features using StandardScaler (for KNN & SVM).

---

## 🔎 Correlation Analysis

- Pearson Correlation used for numerical features.
- Heatmap visualization generated.
- Strong predictors identified:
  - education-num
  - capital-gain
  - hours-per-week

---

## 🤖 Machine Learning Models Implemented

### 1️⃣ Decision Tree
- ID3 (Entropy)
- CART (Gini Index)

### 2️⃣ Naive Bayes
- Gaussian Naive Bayes
- Remedy: Laplace smoothing

### 3️⃣ K-Nearest Neighbors
- k = 5
- Euclidean Distance

### 4️⃣ Support Vector Machine
- Linear Kernel
- RBF Kernel

---

## 📈 Evaluation Metrics

For each classifier, the following were computed:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve

---

## 📊 Comparative Analysis

Bar graphs and ROC curves were generated to compare:

- Accuracy of all models
- Precision, Recall, F1-score comparison
- KNN accuracy vs K-value
- Confusion Matrix heatmaps

---

## 🏆 Results Summary (Typical Performance Range)

| Model | Accuracy |
|-------|----------|
| Decision Tree (Entropy) | ~84% |
| Decision Tree (Gini) | ~86% |
| Naive Bayes | ~82% |
| KNN | ~85% |
| SVM (Linear) | ~87% |
| SVM (RBF) | ~89% |

---

## 📌 Final Inference

After performing preprocessing and applying multiple classification algorithms:

- Decision Trees provided interpretable results.
- Naive Bayes was computationally efficient but slightly less accurate.
- KNN performed well but required scaling.
- SVM with RBF kernel achieved the highest accuracy due to its ability to handle non-linear decision boundaries.

Therefore, **SVM (RBF Kernel)** was the best performing classifier for this dataset.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## 📚 Learning Outcomes

- Applied complete ML workflow on real-world dataset
- Understood model comparison techniques
- Performed statistical correlation analysis
- Implemented multiple classification algorithms
- Interpreted confusion matrix and performance metrics

---

