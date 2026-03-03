# Comparative Analysis of Clustering Algorithms  
(Mall Customer Segmentation Dataset)

## 📌 Objective

The objective of this assignment is to perform a comparative analysis of different clustering algorithms:

- K-Means
- K-Medoids
- Hierarchical Clustering (AGNES)
- EM Algorithm (Gaussian Mixture Model)

The comparison is based on clustering performance metrics and visualization.

---

## 📊 Dataset Used

**Dataset:** Mall Customers Dataset  

**Features:**
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

The dataset contains customer demographic and spending behavior information.

---

## 🛠 Data Preprocessing Steps

1. Removed `CustomerID` column (not useful for clustering).
2. Encoded `Gender` column:
   - Male → 0
   - Female → 1
3. Selected numerical features.
4. Standardized the dataset using `StandardScaler`.

---

## 🔍 Determining Optimal Number of Clusters

Two methods were used:

### 1️⃣ Elbow Method
- Plotted WCSS (Within-Cluster Sum of Squares) for K = 1 to 10.
- Observed elbow point at **K = 5**.

### 2️⃣ Silhouette Score
- Computed Silhouette Score for K = 2 to 10.
- Best separation observed around **K = 5**.

Therefore, **K = 5** was selected for all algorithms.

---

## 🤖 Algorithms Applied

### 1️⃣ K-Means

- Clusters data by minimizing intra-cluster variance.
- Sensitive to outliers.
- Fast and computationally efficient.

**Results:**
- Silhouette Score: 0.2719  
- Davies-Bouldin Index: 1.1811  

---

### 2️⃣ K-Medoids

- Similar to K-Means but uses actual data points as cluster centers.
- More robust to outliers.

**Results:**
- Silhouette Score: 0.3133  
- Davies-Bouldin Index: 1.1497  

---

### 3️⃣ Hierarchical Clustering (AGNES)

- Agglomerative approach (bottom-up).
- Builds dendrogram to show cluster hierarchy.
- Does not produce explicit centroids.

**Results:**
- Silhouette Score: 0.2870  
- Davies-Bouldin Index: 1.2198  

Note: DIANA (Divisive method) was not implemented as it is not directly available in sklearn.

---

### 4️⃣ EM Algorithm (Gaussian Mixture Model)

- Probabilistic clustering approach.
- Assumes Gaussian distribution.
- Handles overlapping and elliptical clusters better.

Clusters were visualized using:
- Annual Income
- Spending Score

---

## 📈 Performance Comparison

| Algorithm   | Silhouette Score | Davies-Bouldin Index | Observation |
|------------|------------------|----------------------|-------------|
| K-Means    | 0.2719           | 1.1811               | Moderate performance |
| K-Medoids  | 0.3133           | 1.1497               | Best separation |
| AGNES      | 0.2870           | 1.2198               | Good hierarchical insight |
| EM (GMM)   | (Add Your Value) | (Add Your Value)     | Handles overlapping clusters |

---

## 📌 Final Conclusion

Based on the performance metrics:

- K-Medoids achieved the highest Silhouette Score and lowest Davies-Bouldin Index.
- K-Means performed moderately but is computationally efficient.
- Hierarchical clustering provided structural insight but had slightly lower performance.
- EM algorithm is useful when clusters overlap and follow Gaussian distributions.

Overall, **K-Medoids performed best for this dataset**, making it more suitable for customer segmentation when robustness is required.

---

## 🧠 Key Learnings

- Feature scaling is essential before clustering.
- Choosing optimal K is critical.
- Different algorithms perform differently depending on data distribution.
- Evaluation metrics help in selecting the best model objectively.

---

## 🛠 Technologies Used

- Python
- Google Colab
- Scikit-learn
- Scikit-learn-extra
- Matplotlib
- Pandas
- NumPy

---


