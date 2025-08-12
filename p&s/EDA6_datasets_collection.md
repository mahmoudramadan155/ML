# EDA Practice Datasets - Complete Collection

## 🎯 **Quick Start - Ready-to-Use Datasets**

### **Option 1: Built-in Python Datasets (Immediate Access)**
```python
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris, load_boston, load_wine

# 1. IRIS DATASET (Classic beginner dataset)
iris = sns.load_dataset('iris')
# Features: sepal_length, sepal_width, petal_length, petal_width, species
# Perfect for: Basic EDA, classification, correlation analysis

# 2. TITANIC DATASET (Survival analysis)
titanic = sns.load_dataset('titanic')
# Features: survived, pclass, sex, age, sibsp, parch, fare, embarked, etc.
# Perfect for: Missing value handling, categorical analysis, survival prediction

# 3. TIPS DATASET (Restaurant tips)
tips = sns.load_dataset('tips')
# Features: total_bill, tip, sex, smoker, day, time, size
# Perfect for: Bivariate analysis, group comparisons

# 4. FLIGHTS DATASET (Flight delays)
flights = sns.load_dataset('flights')
# Features: year, month, passengers
# Perfect for: Time series analysis, trend analysis

# 5. CAR CRASHES DATASET
crashes = sns.load_dataset('car_crashes')
# Features: total, speeding, alcohol, not_distracted, no_previous, ins_premium, ins_losses, abbrev
# Perfect for: State-wise analysis, correlation studies
```

### **Option 2: Download Links (CSV Format)**

#### **🔰 BEGINNER LEVEL**

**1. Iris Dataset**
- **URL**: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv`
- **Size**: 150 rows, 5 columns
- **Use Case**: Perfect first dataset for learning EDA basics
- **Features**: Numerical measurements, clear categories

**2. Tips Dataset**
- **URL**: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv`
- **Size**: 244 rows, 7 columns
- **Use Case**: Restaurant analysis, correlation studies
- **Features**: Mixed data types, interesting relationships

**3. Penguins Dataset**
- **URL**: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv`
- **Size**: 344 rows, 7 columns
- **Use Case**: Species analysis, missing value practice
- **Features**: Categorical and numerical, some missing values

#### **🎯 INTERMEDIATE LEVEL**

**4. House Prices Dataset**
- **URL**: `https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`
- **Size**: 506 rows, 14 columns
- **Use Case**: Real estate analysis, regression preparation
- **Features**: Multiple numerical features, outlier detection

**5. Wine Quality Dataset**
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- **Size**: 1,599 rows, 12 columns
- **Use Case**: Quality analysis, feature importance
- **Features**: Chemical properties, quality ratings

**6. Heart Disease Dataset**
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Size**: 303 rows, 14 columns
- **Use Case**: Medical data analysis, binary classification
- **Features**: Mixed data types, medical indicators

#### **🚀 ADVANCED LEVEL**

**7. Online Retail Dataset**
- **URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx`
- **Size**: 541,909 rows, 8 columns
- **Use Case**: Customer behavior, market basket analysis
- **Features**: Transactional data, time series elements

**8. Superstore Sales Dataset**
- **URL**: Available on Kaggle (requires account)
- **Alternative**: Create synthetic version (code provided below)
- **Size**: 9,994 rows, 21 columns
- **Use Case**: Business analytics, profit analysis

**9. COVID-19 Dataset**
- **URL**: `https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv`
- **Size**: 100,000+ rows, multiple columns
- **Use Case**: Pandemic analysis, global trends
- **Features**: Time series, geographical data

---

## 📥 **Easy Download Code**

### **Method 1: Direct Download Function**
```python
import pandas as pd
import requests
from io import StringIO

def download_dataset(dataset_name):
    """Download popular datasets directly"""
    
    urls = {
        'iris': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
        'titanic': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv',
        'tips': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
        'flights': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv',
        'penguins': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv',
        'car_crashes': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/car_crashes.csv',
        'boston_housing': 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
        'wine_quality': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'diamonds': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv'
    }
    
    if dataset_name not in urls:
        print(f"Available datasets: {list(urls.keys())}")
        return None
    
    try:
        response = requests.get(urls[dataset_name])
        if dataset_name == 'wine_quality':
            df = pd.read_csv(StringIO(response.text), sep=';')
        else:
            df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Successfully loaded {dataset_name}")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None

# Usage examples:
# iris_df = download_dataset('iris')
# titanic_df = download_dataset('titanic')
# tips_df = download_dataset('tips')
```

### **Method 2: Create Synthetic Datasets**
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_sample_ecommerce_data(n_rows=1000):
    """Create synthetic e-commerce dataset for EDA practice"""
    
    np.random.seed(42)
    
    # Customer data
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_rows + 1)]
    ages = np.random.normal(35, 12, n_rows).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows, p=[0.45, 0.50, 0.05])
    cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], 
                             n_rows, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15])
    
    # Purchase data
    categories = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports'], 
                                 n_rows, p=[0.25, 0.30, 0.15, 0.20, 0.10])
    
    # Price based on category
    base_prices = {'Electronics': 200, 'Clothing': 80, 'Books': 25, 'Home & Garden': 150, 'Sports': 120}
    prices = [base_prices[cat] * np.random.lognormal(0, 0.5) for cat in categories]
    prices = np.clip(prices, 10, 2000)
    
    # Purchase dates (last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    purchase_dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]
    
    # Satisfaction scores (correlated with price and category)
    satisfaction_base = np.random.normal(7, 1.5, n_rows)
    # Electronics and Books tend to have higher satisfaction
    satisfaction_boost = [0.5 if cat in ['Electronics', 'Books'] else 0 for cat in categories]
    satisfaction_scores = satisfaction_base + satisfaction_boost
    satisfaction_scores = np.clip(satisfaction_scores, 1, 10)
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    satisfaction_scores[missing_indices] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'city': cities,
        'product_category': categories,
        'purchase_amount': np.round(prices, 2),
        'purchase_date': purchase_dates,
        'satisfaction_score': satisfaction_scores
    })
    
    # Add some duplicates (for duplicate detection practice)
    duplicate_indices = np.random.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    print(f"✅ Created synthetic e-commerce dataset")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Features: {list(df.columns)}")
    print(f"🔍 Missing values: {df.isnull().sum().sum()}")
    print(f"🔄 Duplicate rows: {df.duplicated().sum()}")
    
    return df

def create_sample_sensor_data(n_rows=5000):
    """Create synthetic IoT sensor dataset"""
    
    np.random.seed(42)
    
    # Time series data (hourly for ~7 months)
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_rows)]
    
    # Sensor readings with realistic patterns
    base_temp = 20 + 10 * np.sin(np.arange(n_rows) * 2 * np.pi / (24 * 30))  # Monthly cycle
    daily_temp = 5 * np.sin(np.arange(n_rows) * 2 * np.pi / 24)  # Daily cycle
    noise = np.random.normal(0, 2, n_rows)
    temperature = base_temp + daily_temp + noise
    
    # Humidity (inversely correlated with temperature)
    humidity = 60 - 0.5 * (temperature - 20) + np.random.normal(0, 5, n_rows)
    humidity = np.clip(humidity, 20, 95)
    
    # Equipment status
    equipment_status = np.random.choice(['Normal', 'Warning', 'Critical'], 
                                       n_rows, p=[0.85, 0.12, 0.03])
    
    # Sensor locations
    locations = np.random.choice(['Warehouse_A', 'Warehouse_B', 'Office_Floor_1', 
                                 'Office_Floor_2', 'Production_Line'], n_rows)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'location': locations,
        'temperature_celsius': np.round(temperature, 1),
        'humidity_percent': np.round(humidity, 1),
        'equipment_status': equipment_status
    })
    
    # Introduce some anomalies
    anomaly_indices = np.random.choice(n_rows, size=int(0.01 * n_rows), replace=False)
    df.loc[anomaly_indices, 'temperature_celsius'] += np.random.choice([-20, 20], len(anomaly_indices))
    
    print(f"✅ Created synthetic sensor dataset")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Features: {list(df.columns)}")
    
    return df

# Usage:
# ecommerce_df = create_sample_ecommerce_data(1000)
# sensor_df = create_sample_sensor_data(5000)
```

---

## 🎯 **Dataset Recommendations by EDA Task**

### **For Missing Value Analysis**
- **Penguins Dataset**: Natural missing values
- **Titanic Dataset**: Missing ages and cabin info
- **Heart Disease**: Some missing attributes

### **For Outlier Detection**
- **Boston Housing**: Price outliers
- **Wine Quality**: Chemical composition extremes
- **Car Crashes**: State-level variations

### **For Correlation Analysis**
- **Iris Dataset**: Perfect correlations between measurements
- **Wine Quality**: Chemical properties vs quality
- **House Prices**: Multiple feature correlations

### **For Categorical Analysis**
- **Titanic**: Survival by class, gender
- **Tips**: Tip patterns by day, time
- **Penguins**: Species characteristics

### **For Time Series Analysis**
- **Flights Dataset**: Seasonal passenger trends
- **COVID-19 Data**: Pandemic progression
- **Stock Market Data**: Financial trends

### **For Advanced EDA**
- **Online Retail**: Customer segmentation
- **Sensor Data**: Anomaly detection
- **E-commerce Data**: Business analytics

---

## 🚀 **Quick Start Code**

```python
# Quick setup for any dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load any dataset
df = download_dataset('titanic')  # or any other dataset

# Quick EDA overview
print("=== QUICK EDA OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Numerical columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")

# Quick visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Missing values heatmap
sns.heatmap(df.isnull(), ax=axes[0,0], cbar=True, yticklabels=False)
axes[0,0].set_title('Missing Values')

# Correlation matrix (numerical only)
numerical_df = df.select_dtypes(include=[np.number])
if len(numerical_df.columns) > 1:
    sns.heatmap(numerical_df.corr(), ax=axes[0,1], annot=True, cmap='coolwarm', center=0)
    axes[0,1].set_title('Correlation Matrix')

# Distribution of first numerical column
if len(numerical_df.columns) > 0:
    first_num_col = numerical_df.columns[0]
    numerical_df[first_num_col].hist(ax=axes[1,0], bins=30)
    axes[1,0].set_title(f'Distribution: {first_num_col}')

# Categorical distribution
categorical_df = df.select_dtypes(include=['object', 'category'])
if len(categorical_df.columns) > 0:
    first_cat_col = categorical_df.columns[0]
    df[first_cat_col].value_counts().head(10).plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title(f'Categories: {first_cat_col}')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

---

## 📋 **Dataset Checklist**

### **Before Starting EDA:**
- [ ] Dataset downloaded and loaded successfully
- [ ] Basic shape and structure understood
- [ ] Column names and types identified
- [ ] Sample of data examined

### **Good EDA Practice Datasets Should Have:**
- [ ] Mixed data types (numerical + categorical)
- [ ] Some missing values (for practice)
- [ ] Interesting relationships to discover
- [ ] Real-world context for meaningful insights
- [ ] Appropriate size (not too small, not too large for learning)

### **Recommended Learning Path:**
1. **Start with**: Iris or Tips dataset
2. **Move to**: Titanic or Penguins (missing values)
3. **Advance to**: Wine Quality or Boston Housing
4. **Challenge with**: Online Retail or custom synthetic data

---

## 💡 **Pro Tips**

1. **Start Small**: Begin with datasets under 1000 rows
2. **Progress Gradually**: Add complexity with each new dataset
3. **Focus on Stories**: Choose data that interests you personally
4. **Practice Regularly**: Use different datasets for the same EDA techniques
5. **Document Everything**: Keep notes on insights and methods

**Happy Data Exploring! 🔍📊**