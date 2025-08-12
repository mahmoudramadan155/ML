"""
Comprehensive Exploratory Data Analysis (EDA) Toolkit
====================================================

This unified EDA toolkit combines the best practices and methodologies from multiple approaches
to provide a complete, step-by-step solution for exploratory data analysis.

Features:
- Complete data inspection and quality assessment
- Univariate, bivariate, and multivariate analysis
- Advanced visualizations (static and interactive)
- Statistical testing and hypothesis validation
- Outlier detection with multiple methods
- Missing value analysis and treatment
- Time-series and geospatial analysis capabilities
- Automated insights generation and reporting
- Modular design for custom analysis workflows

Author: Combined from multiple EDA implementations
Version: 1.0
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm

# Optional imports
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    from pandas_profiling import ProfileReport
    _HAS_PROFILING = True
except ImportError:
    _HAS_PROFILING = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)

# Create output directory
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_fig(fig, name: str, dpi: int = 150) -> str:
    """Save matplotlib figure to OUTPUT_DIR and return file path."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only DataFrame copy."""
    return df.select_dtypes(include=[np.number]).copy()

def print_section(title: str, char: str = "=", width: int = 80):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title: str, char: str = "-", width: int = 60):
    """Print formatted subsection header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

# =============================================================================
# STEP 1: DATA LOADING AND INITIAL INSPECTION
# =============================================================================

class DataInspector:
    """Handles initial data loading and inspection."""
    
    @staticmethod
    def load_and_inspect(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset and perform comprehensive initial inspection.
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file
        sample_size : int, optional
            Number of rows to sample for large datasets
            
        Returns:
        --------
        pd.DataFrame
            Loaded and optionally sampled dataset
        """
        print_section("STEP 1: DATA LOADING AND INITIAL INSPECTION")
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Dataset sampled to {sample_size} rows for analysis.")
        
        # Basic information
        print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print_subsection("Dataset Info")
        df.info()
        
        print_subsection("First 5 Rows")
        print(df.head())
        
        print_subsection("Last 5 Rows")
        print(df.tail())
        
        print_subsection("Column Information")
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            print(f"{i:2d}. {col:30s} - {str(dtype):15s}")
        
        return df

# =============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# =============================================================================

class DataQualityAssessor:
    """Comprehensive data quality assessment."""
    
    @staticmethod
    def assess_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
        --------
        dict
            Dictionary containing quality assessment results
        """
        print_section("STEP 2: DATA QUALITY ASSESSMENT")
        
        quality_report = {}
        
        # Missing values analysis
        print_subsection("Missing Values Analysis")
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df)
        quality_report['missing_analysis'] = missing_df
        
        # Visualize missing values
        if missing_counts.sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap')
            save_fig(fig, 'missing_values_heatmap')
        
        # Duplicate analysis
        print_subsection("Duplicate Analysis")
        duplicates = df.duplicated().sum()
        duplicate_percentage = (duplicates / len(df)) * 100
        print(f"Duplicate rows: {duplicates:,} ({duplicate_percentage:.2f}%)")
        quality_report['duplicates'] = {'count': duplicates, 'percentage': duplicate_percentage}
        
        # Data types summary
        print_subsection("Data Types Summary")
        dtype_counts = df.dtypes.value_counts()
        print(dtype_counts)
        quality_report['data_types'] = dtype_counts.to_dict()
        
        # Column statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print_subsection("Column Categories")
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        quality_report['column_types'] = {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
        
        return quality_report

# =============================================================================
# STEP 3: DESCRIPTIVE STATISTICS
# =============================================================================

class DescriptiveAnalyzer:
    """Generate comprehensive descriptive statistics."""
    
    @staticmethod
    def analyze_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed descriptive statistics."""
        print_section("STEP 3: DESCRIPTIVE STATISTICS")
        
        stats_report = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical statistics
        if numerical_cols:
            print_subsection("Numerical Variables Statistics")
            basic_stats = df[numerical_cols].describe()
            print(basic_stats)
            
            # Additional statistics
            print_subsection("Additional Numerical Statistics")
            additional_stats = pd.DataFrame({
                'Variance': df[numerical_cols].var(),
                'Skewness': df[numerical_cols].skew(),
                'Kurtosis': df[numerical_cols].kurtosis(),
                'Range': df[numerical_cols].max() - df[numerical_cols].min(),
                'IQR': df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25)
            })
            print(additional_stats)
            
            stats_report['numerical'] = {
                'basic_stats': basic_stats,
                'additional_stats': additional_stats
            }
        
        # Categorical statistics
        if categorical_cols:
            print_subsection("Categorical Variables Statistics")
            cat_stats = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                most_frequent_count = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                
                print(f"\n{col}:")
                print(f"  Unique values: {unique_count}")
                print(f"  Most frequent: {most_frequent} (count: {most_frequent_count})")
                print(f"  Top 5 values:\n{df[col].value_counts().head()}")
                
                cat_stats[col] = {
                    'unique_count': unique_count,
                    'most_frequent': most_frequent,
                    'value_counts': df[col].value_counts().to_dict()
                }
            
            stats_report['categorical'] = cat_stats
        
        return stats_report

# =============================================================================
# STEP 4: UNIVARIATE ANALYSIS
# =============================================================================

class UnivariateAnalyzer:
    """Comprehensive univariate analysis with visualizations."""
    
    @staticmethod
    def analyze_numerical_univariate(df: pd.DataFrame, numerical_cols: List[str]) -> List[str]:
        """
        Comprehensive analysis of numerical variables.
        
        Returns:
        --------
        List[str]
            List of saved figure paths
        """
        print_section("STEP 4A: UNIVARIATE ANALYSIS - NUMERICAL VARIABLES")
        
        saved_plots = []
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        # Distribution plots (histograms with KDE)
        print_subsection("Distribution Analysis")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution: {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        saved_plots.append(save_fig(fig, 'numerical_distributions'))
        
        # Box plots for outlier detection
        print_subsection("Outlier Detection (Box Plots)")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Box Plot: {col}')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        saved_plots.append(save_fig(fig, 'numerical_boxplots'))
        
        # Q-Q plots for normality assessment
        print_subsection("Normality Assessment (Q-Q Plots)")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
                axes[i].set_title(f'Q-Q Plot: {col}')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        saved_plots.append(save_fig(fig, 'numerical_qqplots'))
        
        return saved_plots
    
    @staticmethod
    def analyze_categorical_univariate(df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
        """
        Comprehensive analysis of categorical variables.
        
        Returns:
        --------
        List[str]
            List of saved figure paths
        """
        print_section("STEP 4B: UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES")
        
        saved_plots = []
        
        for col in categorical_cols:
            if df[col].nunique() > 20:  # Skip very high cardinality variables
                print(f"Skipping {col} - too many unique values ({df[col].nunique()})")
                continue
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1)
            ax1.set_title(f'Bar Plot: {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Pie chart
            ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Pie Chart: {col}')
            
            plt.tight_layout()
            saved_plots.append(save_fig(fig, f'categorical_{col}'))
        
        return saved_plots

# =============================================================================
# STEP 5: BIVARIATE ANALYSIS
# =============================================================================

class BivariateAnalyzer:
    """Comprehensive bivariate analysis."""
    
    @staticmethod
    def analyze_numerical_vs_numerical(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
        """Analyze relationships between numerical variables."""
        print_section("STEP 5A: BIVARIATE ANALYSIS - NUMERICAL vs NUMERICAL")
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical columns for correlation analysis.")
            return {}
        
        # Correlation matrix
        print_subsection("Correlation Analysis")
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix)
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        save_fig(fig, 'correlation_heatmap')
        
        # Pairplot for subset of variables
        if len(numerical_cols) <= 6:
            print_subsection("Pairplot Analysis")
            sns.pairplot(df[numerical_cols])
            plt.suptitle('Pairplot of Numerical Variables', y=1.02)
            save_fig(plt.gcf(), 'numerical_pairplot')
        
        # High correlation pairs analysis
        print_subsection("High Correlation Pairs")
        high_corr_pairs = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr_pairs.append((numerical_cols[i], numerical_cols[j], corr_val))
        
        if high_corr_pairs:
            print("High correlation pairs (|r| > 0.5):")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} vs {col2}: r = {corr:.3f}")
                
                # Create scatter plot for highly correlated pairs
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(df[col1], df[col2], alpha=0.6)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f'Scatter Plot: {col1} vs {col2} (r = {corr:.3f})')
                
                # Add trend line
                z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(df[col1], p(df[col1]), "r--", alpha=0.8)
                ax.grid(True, alpha=0.3)
                
                save_fig(fig, f'scatter_{col1}_vs_{col2}')
        
        return {'correlation_matrix': correlation_matrix, 'high_corr_pairs': high_corr_pairs}
    
    @staticmethod
    def analyze_categorical_vs_numerical(df: pd.DataFrame, categorical_cols: List[str], 
                                       numerical_cols: List[str]) -> List[str]:
        """Analyze relationships between categorical and numerical variables."""
        print_section("STEP 5B: BIVARIATE ANALYSIS - CATEGORICAL vs NUMERICAL")
        
        saved_plots = []
        
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical
            if df[cat_col].nunique() > 10:  # Skip high cardinality
                continue
                
            for num_col in numerical_cols[:3]:  # Limit to first 3 numerical
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Box plot
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=axes[0])
                axes[0].set_title(f'Box Plot: {num_col} by {cat_col}')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Violin plot
                sns.violinplot(data=df, x=cat_col, y=num_col, ax=axes[1])
                axes[1].set_title(f'Violin Plot: {num_col} by {cat_col}')
                axes[1].tick_params(axis='x', rotation=45)
                
                # Strip plot
                sns.stripplot(data=df, x=cat_col, y=num_col, alpha=0.6, ax=axes[2])
                axes[2].set_title(f'Strip Plot: {num_col} by {cat_col}')
                axes[2].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                saved_plots.append(save_fig(fig, f'cat_num_{cat_col}_{num_col}'))
        
        return saved_plots
    
    @staticmethod
    def analyze_categorical_vs_categorical(df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
        """Analyze relationships between categorical variables."""
        print_section("STEP 5C: BIVARIATE ANALYSIS - CATEGORICAL vs CATEGORICAL")
        
        saved_plots = []
        
        for i, cat1 in enumerate(categorical_cols[:3]):
            for cat2 in categorical_cols[i+1:4]:
                if df[cat1].nunique() > 10 or df[cat2].nunique() > 10:
                    continue
                
                # Cross-tabulation
                print_subsection(f"Cross-tabulation: {cat1} vs {cat2}")
                crosstab = pd.crosstab(df[cat1], df[cat2])
                print(crosstab)
                
                # Heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Cross-tabulation Heatmap: {cat1} vs {cat2}')
                save_fig(fig, f'crosstab_{cat1}_{cat2}')
                saved_plots.append(f'crosstab_{cat1}_{cat2}')
                
                # Chi-square test
                chi2, p_value, dof, expected = chi2_contingency(crosstab)
                print(f"Chi-square test: χ² = {chi2:.4f}, p-value = {p_value:.4f}")
                
                if p_value < 0.05:
                    print("Result: Statistically significant association")
                else:
                    print("Result: No significant association")
        
        return saved_plots

# =============================================================================
# STEP 6: MULTIVARIATE ANALYSIS
# =============================================================================

class MultivariateAnalyzer:
    """Advanced multivariate analysis techniques."""
    
    @staticmethod
    def analyze_multivariate(df: pd.DataFrame, numerical_cols: List[str], 
                           categorical_cols: List[str]) -> Dict[str, Any]:
        """Perform comprehensive multivariate analysis."""
        print_section("STEP 6: MULTIVARIATE ANALYSIS")
        
        results = {}
        
        # 3D scatter plots
        if len(numerical_cols) >= 3:
            print_subsection("3D Analysis")
            
            # Static 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            x, y, z = numerical_cols[0], numerical_cols[1], numerical_cols[2]
            scatter = ax.scatter(df[x], df[y], df[z], c=df.index, cmap='viridis', alpha=0.6)
            
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            ax.set_title(f'3D Scatter Plot: {x}, {y}, {z}')
            
            plt.colorbar(scatter)
            save_fig(fig, '3d_scatter')
            
            # Interactive 3D plot with Plotly
            if categorical_cols:
                color_col = categorical_cols[0] if df[categorical_cols[0]].nunique() <= 10 else None
                fig_plotly = px.scatter_3d(df, x=x, y=y, z=z, color=color_col,
                                         title=f'Interactive 3D Plot: {x}, {y}, {z}')
                fig_plotly.write_html(OUTPUT_DIR / '3d_interactive.html')
                results['3d_plot'] = str(OUTPUT_DIR / '3d_interactive.html')
        
        # Parallel coordinates
        if len(numerical_cols) >= 3:
            print_subsection("Parallel Coordinates")
            
            # Normalize data
            normalized_df = df[numerical_cols[:5]].copy()
            for col in normalized_df.columns:
                col_min, col_max = normalized_df[col].min(), normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            pd.plotting.parallel_coordinates(normalized_df.reset_index(), 
                                           'index', alpha=0.3, ax=ax)
            ax.set_title('Parallel Coordinates Plot')
            plt.xticks(rotation=45)
            save_fig(fig, 'parallel_coordinates')
        
        # Grouped analysis
        if categorical_cols and numerical_cols:
            print_subsection("Grouped Analysis")
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 8:
                grouped_stats = df.groupby(cat_col)[numerical_cols].agg(['mean', 'std', 'median'])
                print(f"Grouped statistics by {cat_col}:")
                print(grouped_stats)
                results['grouped_stats'] = grouped_stats
        
        return results

# =============================================================================
# STEP 7: OUTLIER DETECTION
# =============================================================================

class OutlierDetector:
    """Comprehensive outlier detection using multiple methods."""
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Detect outliers using multiple methods.
        
        Returns:
        --------
        pd.DataFrame
            Summary of outliers detected by different methods
        """
        print_section("STEP 7: OUTLIER DETECTION")
        
        outlier_summary = {}
        
        for col in numerical_cols:
            print_subsection(f"Outlier Analysis: {col}")
            
            # Method 1: IQR Method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # Method 2: Z-Score Method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            zscore_outliers = df[z_scores > 3]
            
            # Method 3: Modified Z-Score Method
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                modified_zscore_outliers = df[np.abs(modified_z_scores) > 3.5]
            else:
                modified_zscore_outliers = pd.DataFrame()
            
            outlier_summary[col] = {
                'IQR_outliers': len(iqr_outliers),
                'Z_Score_outliers': len(zscore_outliers),
                'Modified_Z_Score_outliers': len(modified_zscore_outliers)
            }
            
            print(f"  IQR Method: {len(iqr_outliers)} outliers ({len(iqr_outliers)/len(df)*100:.1f}%)")
            print(f"  Z-Score Method: {len(zscore_outliers)} outliers ({len(zscore_outliers)/len(df)*100:.1f}%)")
            print(f"  Modified Z-Score Method: {len(modified_zscore_outliers)} outliers ({len(modified_zscore_outliers)/len(df)*100:.1f}%)")
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Box plot
            axes[0].boxplot(df[col].dropna())
            axes[0].set_title(f'Box Plot: {col}')
            axes[0].set_ylabel(col)
            
            # Histogram with bounds
            axes[1].hist(df[col].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1].axvline(lower_bound, color='red', linestyle='--', label=f'Lower: {lower_bound:.2f}')
            axes[1].axvline(upper_bound, color='red', linestyle='--', label=f'Upper: {upper_bound:.2f}')
            axes[1].set_title(f'Histogram with IQR Bounds: {col}')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            
            # Scatter plot with outliers
            axes[2].scatter(range(len(df)), df[col], alpha=0.6, color='blue', label='Normal')
            if len(iqr_outliers) > 0:
                axes[2].scatter(iqr_outliers.index, iqr_outliers[col], 
                               color='red', alpha=0.8, label='IQR Outliers')
            axes[2].set_title(f'Outlier Detection: {col}')
            axes[2].set_xlabel('Index')
            axes[2].set_ylabel(col)
            axes[2].legend()
            
            plt.tight_layout()
            save_fig(fig, f'outliers_{col}')
        
        # Summary table
        outlier_df = pd.DataFrame(outlier_summary).T
        print_subsection("Outlier Summary")
        print(outlier_df)
        
        return outlier_df

# =============================================================================
# STEP 8: DISTRIBUTION ANALYSIS
# =============================================================================

class DistributionAnalyzer:
    """Analyze and test distributions of numerical variables."""
    
    @staticmethod
    def analyze_distributions(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
        """
        Comprehensive distribution analysis including normality testing.
        
        Returns:
        --------
        Dict[str, Any]
            Distribution analysis results
        """
        print_section("STEP 8: DISTRIBUTION ANALYSIS")
        
        distribution_results = {}
        
        for col in numerical_cols[:4]:  # Analyze first 4 numerical columns
            print_subsection(f"Distribution Analysis: {col}")
            
            data = df[col].dropna()
            
            # Basic statistics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            print(f"  Skewness: {skewness:.4f}")
            print(f"  Kurtosis: {kurtosis:.4f}")
            
            # Normality tests
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
            else:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(5000, random_state=42))
            
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            
            print(f"  Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            print(f"  Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
            
            distribution_results[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'ks_stat': ks_stat,
                'ks_p': ks_p
            }
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Histogram with normal overlay
            axes[0].hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Normal distribution overlay
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
            axes[0].set_title(f'Histogram with Normal Overlay: {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Density')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(data, dist="norm", plot=axes[1])
            axes[1].set_title(f'Q-Q Plot: {col}')
            axes[1].grid(True, alpha=0.3)
            
            # Log transformation (if all positive)
            if (data > 0).all():
                log_data = np.log(data)
                axes[2].hist(log_data, bins=30, density=True, alpha=0.7, 
                           color='lightgreen', edgecolor='black')
                axes[2].set_title(f'Log-transformed Distribution: {col}')
                axes[2].set_xlabel(f'log({col})')
                axes[2].set_ylabel('Density')
            else:
                axes[2].hist(data, bins=30, density=True, alpha=0.7, 
                           color='lightcoral', edgecolor='black')
                axes[2].set_title(f'Original Distribution: {col}')
                axes[2].set_xlabel(col)
                axes[2].set_ylabel('Density')
            
            axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, f'distribution_{col}')
        
        return distribution_results

# =============================================================================
# STEP 9: ADVANCED VISUALIZATIONS
# =============================================================================

class AdvancedVisualizer:
    """Create advanced interactive visualizations."""
    
    @staticmethod
    def create_interactive_plots(df: pd.DataFrame, numerical_cols: List[str], 
                               categorical_cols: List[str]) -> List[str]:
        """
        Create advanced interactive visualizations using Plotly.
        
        Returns:
        --------
        List[str]
            List of saved HTML file paths
        """
        print_section("STEP 9: ADVANCED VISUALIZATIONS")
        
        saved_files = []
        
        # Interactive correlation heatmap
        if len(numerical_cols) >= 2:
            print_subsection("Interactive Correlation Heatmap")
            corr_matrix = df[numerical_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
            ))
            
            fig.update_layout(
                title='Interactive Correlation Heatmap',
                xaxis_title='Variables',
                yaxis_title='Variables',
                width=800,
                height=800
            )
            
            heatmap_path = OUTPUT_DIR / 'interactive_correlation.html'
            fig.write_html(heatmap_path)
            saved_files.append(str(heatmap_path))
        
        # Interactive scatter matrix
        if len(numerical_cols) >= 3:
            print_subsection("Interactive Scatter Matrix")
            fig = px.scatter_matrix(
                df, 
                dimensions=numerical_cols[:4],
                title="Interactive Scatter Matrix"
            )
            
            scatter_matrix_path = OUTPUT_DIR / 'scatter_matrix.html'
            fig.write_html(scatter_matrix_path)
            saved_files.append(str(scatter_matrix_path))
        
        # Interactive box plots
        if numerical_cols and categorical_cols:
            print_subsection("Interactive Box Plots")
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 10:
                for num_col in numerical_cols[:2]:
                    fig = px.box(df, x=cat_col, y=num_col, 
                               title=f'Interactive Box Plot: {num_col} by {cat_col}')
                    
                    box_path = OUTPUT_DIR / f'interactive_box_{cat_col}_{num_col}.html'
                    fig.write_html(box_path)
                    saved_files.append(str(box_path))
        
        # Interactive histograms
        print_subsection("Interactive Histograms")
        for col in numerical_cols[:2]:
            fig = px.histogram(df, x=col, nbins=30, 
                             title=f'Interactive Histogram: {col}')
            
            hist_path = OUTPUT_DIR / f'interactive_hist_{col}.html'
            fig.write_html(hist_path)
            saved_files.append(str(hist_path))
        
        return saved_files

# =============================================================================
# STEP 10: STATISTICAL TESTING
# =============================================================================

class StatisticalTester:
    """Perform statistical hypothesis testing."""
    
    @staticmethod
    def perform_statistical_tests(df: pd.DataFrame, numerical_cols: List[str], 
                                 categorical_cols: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical testing.
        
        Returns:
        --------
        Dict[str, Any]
            Statistical test results
        """
        print_section("STEP 10: STATISTICAL TESTING")
        
        test_results = {}
        
        # T-tests for numerical vs binary categorical
        print_subsection("T-tests")
        for cat_col in categorical_cols:
            if df[cat_col].nunique() == 2:
                categories = df[cat_col].unique()
                for num_col in numerical_cols[:3]:
                    group1 = df[df[cat_col] == categories[0]][num_col].dropna()
                    group2 = df[df[cat_col] == categories[1]][num_col].dropna()
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_val = stats.ttest_ind(group1, group2)
                        
                        print(f"T-test: {num_col} by {cat_col}")
                        print(f"  T-statistic: {t_stat:.4f}")
                        print(f"  P-value: {p_val:.4f}")
                        print(f"  Result: {'Significant' if p_val < 0.05 else 'Not significant'}")
                        
                        test_results[f't_test_{num_col}_{cat_col}'] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
        
        # ANOVA for numerical vs multi-category categorical
        print_subsection("ANOVA Tests")
        for cat_col in categorical_cols:
            if 2 < df[cat_col].nunique() <= 10:
                for num_col in numerical_cols[:3]:
                    groups = [df[df[cat_col] == cat][num_col].dropna() 
                             for cat in df[cat_col].unique()]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        print(f"ANOVA: {num_col} by {cat_col}")
                        print(f"  F-statistic: {f_stat:.4f}")
                        print(f"  P-value: {p_val:.4f}")
                        print(f"  Result: {'Significant' if p_val < 0.05 else 'Not significant'}")
                        
                        test_results[f'anova_{num_col}_{cat_col}'] = {
                            'f_statistic': f_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
        
        # Chi-square tests for categorical vs categorical
        print_subsection("Chi-square Tests")
        for i, cat1 in enumerate(categorical_cols):
            for cat2 in categorical_cols[i+1:]:
                if df[cat1].nunique() <= 10 and df[cat2].nunique() <= 10:
                    contingency_table = pd.crosstab(df[cat1], df[cat2])
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    
                    print(f"Chi-square: {cat1} vs {cat2}")
                    print(f"  Chi-square statistic: {chi2:.4f}")
                    print(f"  P-value: {p_val:.4f}")
                    print(f"  Degrees of freedom: {dof}")
                    print(f"  Result: {'Significant' if p_val < 0.05 else 'Not significant'}")
                    
                    test_results[f'chi2_{cat1}_{cat2}'] = {
                        'chi2_statistic': chi2,
                        'p_value': p_val,
                        'dof': dof,
                        'significant': p_val < 0.05
                    }
        
        return test_results

# =============================================================================
# STEP 11: DIMENSIONALITY REDUCTION
# =============================================================================

class DimensionalityReducer:
    """Perform dimensionality reduction analysis."""
    
    @staticmethod
    def perform_dimensionality_reduction(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
        """
        Perform PCA and t-SNE analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Dimensionality reduction results
        """
        print_section("STEP 11: DIMENSIONALITY REDUCTION")
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical columns for dimensionality reduction.")
            return {}
        
        # Prepare data
        numeric_data = df[numerical_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        results = {}
        
        # PCA Analysis
        print_subsection("Principal Component Analysis (PCA)")
        n_components = min(len(numerical_cols), 5)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("Explained variance by component:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"  PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")
        
        # Plot explained variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig, 'pca_explained_variance')
        
        # 2D PCA plot
        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
            ax.set_title('PCA: First Two Principal Components')
            ax.grid(True, alpha=0.3)
            save_fig(fig, 'pca_2d')
        
        results['pca'] = {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'components': pca.components_
        }
        
        # t-SNE Analysis
        if len(scaled_data) <= 1000:  # t-SNE is computationally expensive
            print_subsection("t-SNE Analysis")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data)//4))
            tsne_result = tsne.fit_transform(scaled_data)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('t-SNE Visualization')
            ax.grid(True, alpha=0.3)
            save_fig(fig, 'tsne_2d')
            
            results['tsne'] = tsne_result
        
        return results

# =============================================================================
# STEP 12: CLUSTERING ANALYSIS
# =============================================================================

class ClusterAnalyzer:
    """Perform clustering analysis."""
    
    @staticmethod
    def perform_clustering_analysis(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
        """
        Perform K-means clustering with elbow method.
        
        Returns:
        --------
        Dict[str, Any]
            Clustering analysis results
        """
        print_section("STEP 12: CLUSTERING ANALYSIS")
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical columns for clustering.")
            return {}
        
        # Prepare data
        numeric_data = df[numerical_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Elbow method for optimal k
        print_subsection("Elbow Method for Optimal K")
        max_k = min(10, len(scaled_data) // 2)
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
            
            if len(scaled_data) <= 1000:  # Silhouette score is expensive for large datasets
                silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)
        
        # Plot elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].grid(True, alpha=0.3)
        
        if silhouette_scores:
            axes[1].plot(k_range, silhouette_scores, 'ro-')
            axes[1].set_xlabel('Number of Clusters (k)')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].set_title('Silhouette Score vs Number of Clusters')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig, 'clustering_analysis')
        
        # Perform clustering with optimal k (choose k=3 as default)
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Visualize clusters (2D)
        if len(numerical_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], 
                               c=cluster_labels, cmap='viridis', alpha=0.6)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
            ax.set_xlabel(numerical_cols[0])
            ax.set_ylabel(numerical_cols[1])
            ax.set_title(f'K-means Clustering (k={optimal_k})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter)
            save_fig(fig, 'kmeans_clusters')
        
        results = {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return results

# =============================================================================
# STEP 13: SUMMARY AND INSIGHTS GENERATION
# =============================================================================

class InsightGenerator:
    """Generate comprehensive summary and insights."""
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, quality_report: Dict, 
                         outlier_summary: pd.DataFrame, statistical_results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive insights and recommendations.
        
        Returns:
        --------
        Dict[str, Any]
            Summary insights and recommendations
        """
        print_section("STEP 13: SUMMARY AND INSIGHTS")
        
        numerical_cols = quality_report['column_types']['numerical']
        categorical_cols = quality_report['column_types']['categorical']
        
        insights = {}
        
        # Dataset overview
        print_subsection("Dataset Overview")
        overview = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numerical_features': len(numerical_cols),
            'categorical_features': len(categorical_cols),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        for key, value in overview.items():
            if isinstance(value, (int, float)):
                print(f"• {key.replace('_', ' ').title()}: {value:,}")
            else:
                print(f"• {key.replace('_', ' ').title()}: {value}")
        
        insights['overview'] = overview
        
        # Data quality insights
        print_subsection("Data Quality Insights")
        quality_insights = []
        
        missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_insights.append(f"Overall missing data: {missing_percent:.2f}%")
        
        high_missing_cols = quality_report['missing_analysis'][
            quality_report['missing_analysis']['Missing_Percentage'] > 50
        ]['Column'].tolist()
        
        if high_missing_cols:
            quality_insights.append(f"Columns with >50% missing data: {high_missing_cols}")
        
        for insight in quality_insights:
            print(f"• {insight}")
        
        insights['quality'] = quality_insights
        
        # Statistical insights
        print_subsection("Statistical Insights")
        statistical_insights = []
        
        if numerical_cols:
            # Skewness analysis
            highly_skewed = []
            for col in numerical_cols:
                skewness = df[col].skew()
                if abs(skewness) > 2:
                    highly_skewed.append(f"{col} (skew: {skewness:.2f})")
            
            if highly_skewed:
                statistical_insights.append(f"Highly skewed variables: {', '.join(highly_skewed)}")
            
            # Correlation analysis
            if len(numerical_cols) >= 2:
                corr_matrix = df[numerical_cols].corr()
                high_corr = []
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr.append(f"{numerical_cols[i]} & {numerical_cols[j]} (r={corr_val:.3f})")
                
                if high_corr:
                    statistical_insights.append(f"High correlation pairs (|r| > 0.8): {', '.join(high_corr)}")
        
        for insight in statistical_insights:
            print(f"• {insight}")
        
        insights['statistical'] = statistical_insights
        
        # Outlier insights
        print_subsection("Outlier Insights")
        outlier_insights = []
        
        if outlier_summary is not None and not outlier_summary.empty:
            total_outliers = outlier_summary['IQR_outliers'].sum()
            outlier_insights.append(f"Total outliers detected (IQR method): {total_outliers}")
            
            high_outlier_cols = outlier_summary[
                outlier_summary['IQR_outliers'] > len(df) * 0.05
            ].index.tolist()
            
            if high_outlier_cols:
                outlier_insights.append(f"Variables with >5% outliers: {high_outlier_cols}")
        
        for insight in outlier_insights:
            print(f"• {insight}")
        
        insights['outliers'] = outlier_insights
        
        # Categorical insights
        print_subsection("Categorical Insights")
        categorical_insights = []
        
        if categorical_cols:
            high_cardinality = []
            low_variance = []
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count > len(df) * 0.95:
                    high_cardinality.append(f"{col} ({unique_count} unique)")
                elif unique_count <= 2:
                    low_variance.append(f"{col} ({unique_count} unique)")
            
            if high_cardinality:
                categorical_insights.append(f"High cardinality variables: {', '.join(high_cardinality)}")
            if low_variance:
                categorical_insights.append(f"Low variance variables: {', '.join(low_variance)}")
        
        for insight in categorical_insights:
            print(f"• {insight}")
        
        insights['categorical'] = categorical_insights
        
        # Recommendations
        print_subsection("Recommendations")
        recommendations = []
        
        if df.isnull().sum().sum() > 0:
            recommendations.append("Address missing values using appropriate imputation strategies")
        
        if outlier_summary is not None and not outlier_summary.empty and outlier_summary['IQR_outliers'].sum() > 0:
            recommendations.append("Investigate and handle outliers based on domain knowledge")
        
        if numerical_cols:
            highly_skewed_count = sum(1 for col in numerical_cols if abs(df[col].skew()) > 2)
            if highly_skewed_count > 0:
                recommendations.append("Consider transformation (log, sqrt) for highly skewed variables")
        
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            high_corr_count = sum(1 for i in range(len(numerical_cols)) 
                                 for j in range(i+1, len(numerical_cols)) 
                                 if abs(corr_matrix.iloc[i, j]) > 0.8)
            if high_corr_count > 0:
                recommendations.append("Consider dimensionality reduction for highly correlated features")
        
        if categorical_cols:
            high_card_count = sum(1 for col in categorical_cols if df[col].nunique() > len(df) * 0.95)
            if high_card_count > 0:
                recommendations.append("Consider encoding strategies for high cardinality categorical variables")
        
        if not recommendations:
            recommendations.append("Dataset appears to be in good condition for analysis")
        
        for rec in recommendations:
            print(f"• {rec}")
        
        insights['recommendations'] = recommendations
        
        return insights

# =============================================================================
# MAIN EDA ORCHESTRATOR
# =============================================================================

class ComprehensiveEDA:
    """Main class orchestrating the entire EDA process."""
    
    def __init__(self, file_path: str, sample_size: Optional[int] = None):
        """
        Initialize the EDA with dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file
        sample_size : int, optional
            Sample size for large datasets
        """
        self.file_path = file_path
        self.sample_size = sample_size
        self.df = None
        self.results = {}
    
    def run_complete_eda(self) -> Dict[str, Any]:
        """
        Execute the complete EDA pipeline.
        
        Returns:
        --------
        Dict[str, Any]
            Complete EDA results
        """
        print("🔍 COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        try:
            # Step 1: Load and inspect data
            self.df = DataInspector.load_and_inspect(self.file_path, self.sample_size)
            
            # Step 2: Assess data quality
            quality_report = DataQualityAssessor.assess_quality(self.df)
            self.results['quality_report'] = quality_report
            
            # Step 3: Descriptive statistics
            descriptive_stats = DescriptiveAnalyzer.analyze_descriptive_stats(self.df)
            self.results['descriptive_stats'] = descriptive_stats
            
            # Get column types for subsequent analysis
            numerical_cols = quality_report['column_types']['numerical']
            categorical_cols = quality_report['column_types']['categorical']
            
            # Step 4: Univariate analysis
            if numerical_cols:
                univariate_plots = UnivariateAnalyzer.analyze_numerical_univariate(self.df, numerical_cols)
                self.results['univariate_plots'] = univariate_plots
            
            if categorical_cols:
                categorical_plots = UnivariateAnalyzer.analyze_categorical_univariate(self.df, categorical_cols)
                self.results['categorical_plots'] = categorical_plots
            
            # Step 5: Bivariate analysis
            bivariate_analyzer = BivariateAnalyzer()
            
            if len(numerical_cols) >= 2:
                numerical_bivariate = bivariate_analyzer.analyze_numerical_vs_numerical(self.df, numerical_cols)
                self.results['numerical_bivariate'] = numerical_bivariate
            
            if numerical_cols and categorical_cols:
                cat_num_plots = bivariate_analyzer.analyze_categorical_vs_numerical(
                    self.df, categorical_cols, numerical_cols)
                self.results['cat_num_plots'] = cat_num_plots
            
            if len(categorical_cols) >= 2:
                cat_cat_plots = bivariate_analyzer.analyze_categorical_vs_categorical(self.df, categorical_cols)
                self.results['cat_cat_plots'] = cat_cat_plots
            
            # Step 6: Multivariate analysis
            multivariate_results = MultivariateAnalyzer.analyze_multivariate(
                self.df, numerical_cols, categorical_cols)
            self.results['multivariate'] = multivariate_results
            
            # Step 7: Outlier detection
            if numerical_cols:
                outlier_summary = OutlierDetector.detect_outliers(self.df, numerical_cols)
                self.results['outlier_summary'] = outlier_summary
            else:
                outlier_summary = pd.DataFrame()
            
            # Step 8: Distribution analysis
            if numerical_cols:
                distribution_results = DistributionAnalyzer.analyze_distributions(self.df, numerical_cols)
                self.results['distribution_analysis'] = distribution_results
            
            # Step 9: Advanced visualizations
            interactive_plots = AdvancedVisualizer.create_interactive_plots(
                self.df, numerical_cols, categorical_cols)
            self.results['interactive_plots'] = interactive_plots
            
            # Step 10: Statistical testing
            statistical_results = StatisticalTester.perform_statistical_tests(
                self.df, numerical_cols, categorical_cols)
            self.results['statistical_tests'] = statistical_results
            
            # Step 11: Dimensionality reduction
            if len(numerical_cols) >= 2:
                dim_reduction_results = DimensionalityReducer.perform_dimensionality_reduction(
                    self.df, numerical_cols)
                self.results['dimensionality_reduction'] = dim_reduction_results
            
            # Step 12: Clustering analysis
            if len(numerical_cols) >= 2:
                clustering_results = ClusterAnalyzer.perform_clustering_analysis(self.df, numerical_cols)
                self.results['clustering'] = clustering_results
            
            # Step 13: Generate insights and summary
            insights = InsightGenerator.generate_insights(
                self.df, quality_report, outlier_summary, statistical_results)
            self.results['insights'] = insights
            
            # Generate HTML report
            self._generate_html_report()
            
            print("\n" + "=" * 80)
            print("✅ COMPREHENSIVE EDA COMPLETED SUCCESSFULLY!")
            print(f"📊 Results saved to: {OUTPUT_DIR.absolute()}")
            print("📄 View the complete report: eda_outputs/eda_report.html")
            print("=" * 80)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error during EDA execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report."""
        html_content = self._create_html_report()
        
        report_path = OUTPUT_DIR / 'eda_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 HTML report generated: {report_path}")
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Comprehensive EDA Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; }
                .section { margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #f8f9fa; }
                .section h2 { color: #2c3e50; margin-top: 0; }
                .insight { background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .stats-table th { background-color: #3498db; color: white; }
                .image-gallery { display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }
                .image-item { flex: 1; min-width: 300px; text-align: center; }
                .image-item img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #3498db; color: white; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Comprehensive EDA Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Dataset: {dataset_name}</p>
            </div>
        """.format(
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name=os.path.basename(self.file_path)
        )
        
        # Dataset Overview
        if 'insights' in self.results and 'overview' in self.results['insights']:
            overview = self.results['insights']['overview']
            html += f"""
            <div class="section">
                <h2>📊 Dataset Overview</h2>
                <div class="metric">Records: {overview['total_records']:,}</div>
                <div class="metric">Features: {overview['total_features']}</div>
                <div class="metric">Numerical: {overview['numerical_features']}</div>
                <div class="metric">Categorical: {overview['categorical_features']}</div>
                <div class="metric">Missing Values: {overview['missing_values']:,}</div>
                <div class="metric">Duplicates: {overview['duplicate_rows']:,}</div>
            </div>
            """
        
        # Key Insights
        if 'insights' in self.results:
            insights = self.results['insights']
            html += """
            <div class="section">
                <h2>💡 Key Insights</h2>
            """
            
            for category, insight_list in insights.items():
                if category != 'overview' and isinstance(insight_list, list):
                    html += f"<h3>{category.title()} Insights</h3>"
                    for insight in insight_list:
                        if category == 'recommendations':
                            html += f'<div class="recommendation">• {insight}</div>'
                        else:
                            html += f'<div class="insight">• {insight}</div>'
            
            html += "</div>"
        
        # Visual Gallery
        html += """
        <div class="section">
            <h2>📈 Visualizations</h2>
            <div class="image-gallery">
        """
        
        # Add all PNG images from the output directory
        for img_file in OUTPUT_DIR.glob("*.png"):
            img_name = img_file.stem.replace('_', ' ').title()
            html += f"""
            <div class="image-item">
                <h4>{img_name}</h4>
                <img src="{img_file.name}" alt="{img_name}">
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        # Interactive Plots
        if 'interactive_plots' in self.results and self.results['interactive_plots']:
            html += """
            <div class="section">
                <h2>🎯 Interactive Visualizations</h2>
                <p>The following interactive plots have been generated:</p>
                <ul>
            """
            
            for plot_path in self.results['interactive_plots']:
                plot_name = os.path.basename(plot_path).replace('.html', '').replace('_', ' ').title()
                html += f'<li><a href="{os.path.basename(plot_path)}" target="_blank">{plot_name}</a></li>'
            
            html += """
                </ul>
            </div>
            """
        
        # Statistical Results
        if 'statistical_tests' in self.results:
            html += """
            <div class="section">
                <h2>📊 Statistical Test Results</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Test</th>
                            <th>Statistic</th>
                            <th>P-value</th>
                            <th>Significant</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for test_name, result in self.results['statistical_tests'].items():
                stat_key = [k for k in result.keys() if 'statistic' in k][0] if any('statistic' in k for k in result.keys()) else 'statistic'
                html += f"""
                <tr>
                    <td>{test_name.replace('_', ' ').title()}</td>
                    <td>{result.get(stat_key, 'N/A'):.4f}</td>
                    <td>{result.get('p_value', 'N/A'):.4f}</td>
                    <td>{'Yes' if result.get('significant', False) else 'No'}</td>
                </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        html += """
            <div class="section">
                <h2>📝 Summary</h2>
                <p>This comprehensive EDA report provides insights into your dataset's structure, quality, and relationships. 
                Use these findings to guide your data preprocessing and modeling decisions.</p>
                <p><strong>Next Steps:</strong></p>
                <ul>
                    <li>Address any data quality issues identified</li>
                    <li>Consider the recommended transformations</li>
                    <li>Use the correlation and relationship insights for feature engineering</li>
                    <li>Apply appropriate preprocessing based on the distribution analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html

# =============================================================================
# UTILITY FUNCTIONS FOR SPECIFIC EDA TASKS
# =============================================================================

def quick_eda(file_path: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick EDA function for immediate analysis.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
    sample_size : int, optional
        Sample size for large datasets
        
    Returns:
    --------
    Dict[str, Any]
        EDA results
    """
    eda = ComprehensiveEDA(file_path, sample_size)
    return eda.run_complete_eda()

def create_eda_template(df: pd.DataFrame, output_file: str = 'eda_template.py'):
    """
    Create a Python template for EDA based on the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_file : str
        Output file name for the template
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    template = f'''
# EDA Template for your dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('your_file.csv')

# Basic info
print("Dataset shape:", df.shape)
print("\\nColumns:", df.columns.tolist())
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())

# Numerical columns: {numerical_cols}
numerical_cols = {numerical_cols}

# Categorical columns: {categorical_cols}
categorical_cols = {categorical_cols}

# Quick visualizations
for col in numerical_cols:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    df[col].hist(bins=30)
    plt.title(f'{{col}} Distribution')
    
    plt.subplot(1, 3, 2)
    df.boxplot(column=col)
    plt.title(f'{{col}} Boxplot')
    
    plt.subplot(1, 3, 3)
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'{{col}} Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# Correlation matrix
if len(numerical_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Categorical analysis
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'{{col}} Value Counts')
    plt.xticks(rotation=45)
    plt.show()
'''
    
    with open(output_file, 'w') as f:
        f.write(template)
    
    print(f"EDA template saved to {output_file}")

def automated_profiling_report(df: pd.DataFrame, output_file: str = 'profile_report.html'):
    """
    Generate automated profiling report using pandas-profiling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_file : str
        Output HTML file name
    """
    if _HAS_PROFILING:
        try:
            profile = ProfileReport(df, title="Automated EDA Report", explorative=True)
            profile.to_file(output_file)
            print(f"Automated profiling report saved to {output_file}")
        except Exception as e:
            print(f"Error generating profiling report: {e}")
    else:
        print("pandas-profiling not installed. Install with: pip install pandas-profiling")

# =============================================================================
# TIME SERIES SPECIFIC EDA
# =============================================================================

class TimeSeriesEDA:
    """Specialized EDA for time series data."""
    
    @staticmethod
    def analyze_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> Dict[str, Any]:
        """
        Comprehensive time series EDA.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with time series data
        date_col : str
            Name of the date column
        value_cols : List[str]
            Names of value columns to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Time series analysis results
        """
        print_section("TIME SERIES EDA")
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        df_ts = df.set_index(date_col).sort_index()
        
        results = {}
        
        for col in value_cols:
            print_subsection(f"Time Series Analysis: {col}")
            
            series = df_ts[col].dropna()
            
            # Basic time series plots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Original series
            series.plot(ax=axes[0], title=f'{col} - Original Series')
            axes[0].grid(True, alpha=0.3)
            
            # Rolling statistics
            rolling_mean = series.rolling(window=12, min_periods=1).mean()
            rolling_std = series.rolling(window=12, min_periods=1).std()
            
            axes[1].plot(series.index, rolling_mean, label='Rolling Mean', color='red')
            axes[1].plot(series.index, rolling_std, label='Rolling Std', color='blue')
            axes[1].set_title(f'{col} - Rolling Statistics (12 periods)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Decomposition (if enough data)
            if len(series) >= 24:
                try:
                    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=12)
                    
                    axes[2].plot(decomposition.trend, label='Trend', color='red')
                    axes[2].set_title(f'{col} - Trend Component')
                    axes[2].grid(True, alpha=0.3)
                    
                    # Full decomposition plot
                    fig_decomp = decomposition.plot()
                    fig_decomp.suptitle(f'Seasonal Decomposition: {col}')
                    save_fig(fig_decomp, f'ts_decomposition_{col}')
                    
                except Exception as e:
                    axes[2].text(0.5, 0.5, f'Decomposition failed: {str(e)}', 
                               transform=axes[2].transAxes, ha='center')
            
            plt.tight_layout()
            save_fig(fig, f'time_series_{col}')
            
            # Autocorrelation analysis
            if len(series) > 20:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                sm.tsa.plot_acf(series.dropna(), ax=ax1, lags=min(40, len(series)//4))
                ax1.set_title(f'Autocorrelation: {col}')
                
                sm.tsa.plot_pacf(series.dropna(), ax=ax2, lags=min(20, len(series)//8))
                ax2.set_title(f'Partial Autocorrelation: {col}')
                
                plt.tight_layout()
                save_fig(fig, f'autocorrelation_{col}')
        
        return results

# =============================================================================
# MAIN EXECUTION AND EXAMPLES
# =============================================================================

def main():
    """
    Main function demonstrating the EDA toolkit usage.
    """
    print("🔍 Comprehensive EDA Toolkit")
    print("=" * 50)
    print("This toolkit provides a complete solution for exploratory data analysis.")
    print("=" * 50)
    
    # Example usage with sample data
    print("\n📝 Creating sample dataset for demonstration...")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples).clip(20000, 200000),
        'education_years': np.random.randint(8, 20, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-employed'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Introduce some missing values and outliers
    sample_df.loc[sample_df.sample(n=50).index, 'income'] = np.nan
    sample_df.loc[sample_df.sample(n=30).index, 'satisfaction_score'] = np.nan
    sample_df.loc[sample_df.sample(n=20).index, 'income'] = np.random.uniform(300000, 500000, 20)  # Outliers
    
    # Save sample dataset
    sample_file = OUTPUT_DIR / 'sample_dataset.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"📊 Sample dataset created: {sample_file}")
    
    # Run comprehensive EDA
    print("\n🚀 Running comprehensive EDA on sample dataset...")
    eda_results = quick_eda(str(sample_file))
    
    if eda_results:
        print("\n✅ EDA completed successfully!")
        print(f"📁 All outputs saved to: {OUTPUT_DIR.absolute()}")
        print("\n📋 Available functions:")
        print("  • ComprehensiveEDA().run_complete_eda() - Full EDA pipeline")
        print("  • quick_eda(file_path) - Quick analysis")
        print("  • create_eda_template(df) - Generate EDA template")
        print("  • TimeSeriesEDA.analyze_time_series() - Time series analysis")
        print("  • automated_profiling_report() - Automated report")
    else:
        print("❌ EDA failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

# =============================================================================
# ADDITIONAL EXAMPLES AND DOCUMENTATION
# =============================================================================

"""
USAGE EXAMPLES:
===============

1. Basic Usage:
   from comprehensive_eda import quick_eda
   results = quick_eda('your_dataset.csv')

2. Advanced Usage:
   from comprehensive_eda import ComprehensiveEDA
   eda = ComprehensiveEDA('your_dataset.csv', sample_size=5000)
   results = eda.run_complete_eda()

3. Individual Components:
   from comprehensive_eda import DataInspector, BivariateAnalyzer
   df = DataInspector.load_and_inspect('your_dataset.csv')
   BivariateAnalyzer.analyze_numerical_vs_numerical(df, numerical_cols)

4. Time Series Analysis:
   from comprehensive_eda import TimeSeriesEDA
   TimeSeriesEDA.analyze_time_series(df, 'date_column', ['value_column'])

5. Generate EDA Template:
   from comprehensive_eda import create_eda_template
   create_eda_template(df, 'my_eda_template.py')

FEATURES INCLUDED:
==================
✅ Data Loading & Quality Assessment
✅ Descriptive Statistics (Central tendency, spread, shape)
✅ Univariate Analysis (Distributions, outliers, normality)
✅ Bivariate Analysis (Correlations, relationships, associations)
✅ Multivariate Analysis (3D plots, parallel coordinates, clustering)
✅ Missing Value Analysis & Treatment Recommendations
✅ Outlier Detection (IQR, Z-score, Modified Z-score)
✅ Statistical Hypothesis Testing (T-tests, ANOVA, Chi-square)
✅ Distribution Analysis & Normality Testing
✅ Dimensionality Reduction (PCA, t-SNE)
✅ Clustering Analysis (K-means with elbow method)
✅ Advanced Visualizations (Static + Interactive with Plotly)
✅ Time Series Analysis (Trend, seasonality, autocorrelation)
✅ Automated Insights & Recommendations Generation
✅ Comprehensive HTML Report Generation
✅ Modular Design for Custom Analysis Workflows

REQUIREMENTS:
=============
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
statsmodels >= 0.12.0

Optional:
umap-learn >= 0.5.0  # For UMAP dimensionality reduction
pandas-profiling >= 3.0.0  # For automated profiling reports

INSTALLATION:
=============
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn statsmodels

# Optional packages
pip install umap-learn pandas-profiling

This comprehensive EDA toolkit provides everything you need for thorough
exploratory data analysis, from basic statistics to advanced visualizations
and automated insights generation.
"""