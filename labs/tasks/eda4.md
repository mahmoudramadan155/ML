### Task 1: First Contact - Initial Data Inspection

**Objective:** Get a high-level overview of the dataset.

*   **1.1:** Run the code to create the sample DataFrame.
*   **1.2:** Use the `.head()` and `.tail()` functions. Are the introduced duplicate rows visible?
*   **1.3:** Use `.shape` to state the initial dimensions (rows and columns) of the dataset before any cleaning.
*   **1.4:** Use the `.info()` method. Which columns have missing values? What are the data types of the `income` and `gender` columns?

### Task 2: The Numbers Game - Descriptive Statistics

**Objective:** Summarize the core characteristics of the data.

*   **2.1:** Generate the descriptive statistics for the **numerical columns**.
    *   What is the average income?
    *   What is the age range (min and max) of the individuals in this dataset?
    *   What is the median `satisfaction_score`?
*   **2.2:** Generate the descriptive statistics for the **categorical columns**.
    *   How many unique cities are there?
    *   Which gender is the most frequent (top) in the dataset?

### Task 3: Clean Up Duty - Data Cleaning

**Objective:** Identify and resolve data quality issues like missing values, duplicates, and outliers.

*   **3.1: Missing Values:**
    *   Generate the count and the heatmap of missing values. Confirm that `income` and `satisfaction_score` have nulls.
    *   Run the imputation code.
    *   Verify that there are no more missing values in the dataset by running `.isnull().sum()` again.
*   **3.2: Duplicate Data:**
    *   Identify the number of duplicate rows in the dataset.
    *   Run the code to remove the duplicates.
    *   Report the new shape of the DataFrame. How many rows were removed?
*   **3.3: Outlier Identification:**
    *   Generate a box plot for the `age` column. Based on the plot, are there any apparent outliers in age?

### Task 4: One at a Time - Univariate Analysis

**Objective:** Analyze each variable individually to understand its distribution.

*   **4.1:** Generate a histogram for the `satisfaction_score`. Describe the shape of its distribution (e.g., is it uniform, skewed left, skewed right, or normal?).
*   **4.2:** Create a pie chart for the `city` column. Which city has the smallest representation in the dataset? Express its proportion as a percentage.

### Task 5: It Takes Two - Bivariate Analysis

**Objective:** Explore the relationships between pairs of variables.

*   **5.1: Numerical vs. Numerical:** Generate a scatter plot to visualize the relationship between `age` and `satisfaction_score`. Does there appear to be any correlation between them?
*   **5.2: Numerical vs. Categorical:** Create a violin plot to compare the `income` distribution across the different `gender` categories. What differences can you observe between the groups?
*   **5.3: Categorical vs. Categorical:**
    *   Create a contingency table (`pd.crosstab`) between `gender` and `city`.
    *   Which city has the highest number of 'Male' participants?
    *   Visualize this relationship using a grouped bar chart.

### Task 6: The Grand View - Multivariate Analysis

**Objective:** Visualize the interactions between multiple variables simultaneously.

*   **6.1:** Generate a `pairplot` for all numerical variables, but this time set the `hue` parameter to `'city'`.
*   **6.2:** Based on the pair plot, which two variables appear to have the most interesting or distinct relationship when viewed across different cities? Describe what you see.

### Task 7: Making More from Less - Feature Engineering

**Objective:** Create new, potentially more useful, features from the existing data.

*   **7.1:** Create a new feature named `income_group`. Categorize individuals into 'Low' (< $40000), 'Medium' ($40000 - $60000), and 'High' (> $60000) income brackets based on their `income`.
*   **7.2:** Display the value counts for your new `income_group` column.
*   **7.3:** Prepare the `city` column for a machine learning model by applying one-hot encoding. Show the first 5 rows of the resulting DataFrame.

### Task 8: Prove It! - Hypothesis Testing

**Objective:** Use statistical tests to validate the relationships observed during the visual analysis.

*   **8.1: Income and City:** In Task 5.2, you might have visually compared income across different groups. Now, perform an **ANOVA** test to determine if there is a statistically significant difference in `income` across the different `cities`. State your conclusion based on the p-value.
*   **8.2: Gender and City Independence:** Using the contingency table from Task 5.3, perform a **Chi-Square test**. Is the choice of `city` independent of an individual's `gender`? Interpret the resulting p-value to support your answer.

### Challenge Task: The Final Report

**Objective:** Synthesize and communicate your findings.

*   Based on all the steps above, write a brief summary (2-3 paragraphs) of your findings. Imagine you are presenting this to a non-technical manager. Your summary should cover:
    1.  A brief description of the data quality.
    2.  Key demographic characteristics of the individuals in the dataset.
    3.  At least two significant relationships or insights you discovered (e.g., "Income appears to be significantly higher in City X," or "There is no significant link between age and satisfaction").
