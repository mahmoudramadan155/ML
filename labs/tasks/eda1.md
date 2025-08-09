## **EDA Script Tasks**

### **1. Setup & Preparation**

* [ ] **Install dependencies**
  Run:

  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels plotly umap-learn
  ```
* [ ] **Create project folder structure**

  ```
  project/
  ├── Comprehensive_EDA_step_by_step.py
  ├── data/
  │   └── dataset.csv   # Replace with your dataset
  ├── eda_outputs/      # auto-created for plots
  ```

---

### **2. Data Import & Overview**

* [ ] Load dataset (CSV, Excel, SQL, or seaborn sample)
* [ ] Run **`overview()`** to get:

  * Shape (rows/columns)
  * Data types
  * Memory usage
  * Descriptive statistics (numeric & categorical)
* [ ] Save this information for documentation

---

### **3. Missing Values Analysis**

* [ ] Run **`missing_values_table()`**
* [ ] Identify high-missingness columns
* [ ] Save missing value bar chart
* [ ] Decide on **drop vs impute** strategy

---

### **4. Categorical Feature Analysis**

* [ ] Run **`unique_value_counts()`**
* [ ] For key categorical variables, run **`plot_value_counts()`**
* [ ] Detect potential ID-like columns (too many unique values)

---

### **5. Numerical Feature Analysis**

* [ ] Run **`distribution_plots_numeric()`**

  * Histogram + KDE for each numeric column
  * Boxplot for outlier detection
* [ ] Record skewness/kurtosis for transformations if needed

---

### **6. Correlation & Relationships**

* [ ] Run **`correlation_analysis()`** (Pearson/Spearman)
* [ ] Save heatmap & pairplot
* [ ] Identify **multicollinearity** via **`calculate_vif()`**
* [ ] Flag redundant features for potential removal

---

### **7. Outlier Detection**

* [ ] Apply **`detect_outliers_iqr()`** on each numeric column
* [ ] Apply **`detect_outliers_zscore()`**
* [ ] Decide on capping/removal/retaining

---

### **8. Missing Data Imputation**

* [ ] Try **`impute_simple()`** (mean/median)
* [ ] Try **`impute_knn()`** for more complex datasets
* [ ] Compare before/after distribution

---

### **9. Feature Encoding & Scaling**

* [ ] For categorical → **`encode_onehot()`** or **`label_encode_series()`**
* [ ] For numeric → **`scale_features()`** (Standard, MinMax, Robust)

---

### **10. Dimensionality Reduction**

* [ ] Run **`pca_analysis()`**

  * Save explained variance plot
* [ ] Run **`tsne_umap_reduction()`** for visualization

---

### **11. Clustering Analysis**

* [ ] Run **`elbow_kmeans()`** to find best k
* [ ] Test clustering with silhouette score

---

### **12. Time-Series EDA** *(if applicable)*

* [ ] Run **`timeseries_eda()`**:

  * Line plot
  * Rolling mean/std
  * Seasonal decomposition

---

### **13. Statistical Hypothesis Testing**

* [ ] If binary group & numeric feature → **`t_test_between_groups()`**
* [ ] If two categorical features → **`chi2_test_categorical()`**

---

### **14. Automated EDA Report**

* [ ] Run **`automated_report(df, target='column')`**
* [ ] Open **`eda_outputs/index.html`** for all plots in one place
* [ ] Review & summarize findings

---

### **15. Documentation & Insights**

* [ ] Write summary of EDA findings
* [ ] Suggest preprocessing steps for modeling
* [ ] Archive charts & tables
