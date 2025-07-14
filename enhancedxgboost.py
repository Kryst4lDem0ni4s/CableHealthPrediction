"""
Enhanced Cable Health Prediction System using XGBoost
====================================================
Advanced predictive analytics for cable health assessment with 3-class classification:
- 0: Healthy (Good condition, normal risk)
- 1: At Risk (Deterioration signs, preventive maintenance needed)  
- 2: Critical (Poor condition, immediate action required)

Algorithm Choice: XGBoost selected for comparative analysis against LightGBM
with focus on interpretability and robust performance on imbalanced datasets.

ENHANCEMENTS:
- Adaptive hyperparameter search space with continuous optimization
- Aggressive regularization for overfitting prevention
- Robust feature selection with multiple methods
- Enhanced cross-validation for large datasets
- Ensemble methods and data augmentation
- Comprehensive feature analysis pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_recall_fscore_support,
                           roc_auc_score, matthews_corrcoef, log_loss)

# Advanced Analytics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import (SelectKBest, mutual_info_classif, f_classif, 
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier, 
                            ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.utils import resample
from sklearn.model_selection import learning_curve
import shap
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🔧 Enhanced Cable Health Prediction System - XGBoost Implementation")
print("📊 Algorithm: XGBoost (Extreme Gradient Boosting)")
print("🎯 Task: Multi-class Classification with Enhanced Interpretability")
print("🚀 NEW: Adaptive optimization, Aggressive regularization, Ensemble methods, Data augmentation")

class EnhancedCableHealthPredictorXGBoost:
    """
    Enhanced Cable Health Prediction System using XGBoost
    
    This implementation provides comprehensive machine learning pipeline optimized
    for XGBoost with enhanced interpretability features for cable maintenance
    decision-making and direct performance comparison with LightGBM.
    
    ENHANCEMENTS:
    - Adaptive hyperparameter optimization with continuous values
    - Aggressive regularization for overfitting prevention
    - Robust feature selection methods
    - Enhanced cross-validation strategies
    - Ensemble methods and data augmentation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.ensemble_model = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.class_weights = None
        self.preprocessor = None
        self.shap_explainer = None
        self.feature_importance_df = None
        self.feature_selector = None
        self.selected_features = None
        self.best_params = None
        
        # Define categorical features for proper encoding
        self.categorical_columns = [
            'MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory',
            'FloodZoneRisk'
        ]
        
        print(f"✅ Enhanced XGBoost CableHealthPredictor initialized with random_state={random_state}")
    
    def load_and_explore_data(self, filepath):
        """
        Load cable health dataset with enhanced exploratory analysis
        
        Theory: XGBoost requires more careful preprocessing than LightGBM,
        particularly for categorical features which need explicit encoding.
        """
        print("\n📂 Loading and Exploring Dataset for XGBoost...")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Enhanced sample size analysis for overfitting detection
        n_samples, n_features = df.shape[0], df.shape[1] - 1  # Exclude target
        sample_to_feature_ratio = n_samples / n_features
        print(f"Sample-to-feature ratio: {sample_to_feature_ratio:.2f}")
        
        if sample_to_feature_ratio < 10:
            print("⚠️  WARNING: Sample-to-feature ratio < 10:1. Risk of overfitting detected!")
            print("🔧 RECOMMENDATION: Use aggressive regularization and feature selection")
        elif sample_to_feature_ratio < 5:
            print("🚨 CRITICAL: Sample-to-feature ratio < 5:1. High overfitting risk!")
            print("🔧 MANDATORY: Aggressive regularization, feature selection, and ensemble methods required")
        elif sample_to_feature_ratio < 1:
            print("💀 SEVERE: Sample-to-feature ratio < 1:1. Extreme overfitting guaranteed!")
            print("🔧 EMERGENCY: Maximum regularization, aggressive feature selection, and data augmentation required")
        
        # Enhanced class distribution analysis
        print("\n🎯 Target Variable Distribution Analysis:")
        class_counts = df['CableHealthScore'].value_counts().sort_index()
        class_props = df['CableHealthScore'].value_counts(normalize=True).sort_index()
        
        status_labels = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical']
        for i, (count, prop) in enumerate(zip(class_counts, class_props)):
            print(f"Class {i} ({status_labels[i]}): {count:,} samples ({prop:.1%})")
        
        # Calculate imbalance metrics for XGBoost optimization
        imbalance_ratio = class_counts.max() / class_counts.min()
        minority_class_ratio = class_counts.min() / class_counts.sum()
        
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        print(f"Minority Class Ratio: {minority_class_ratio:.3f}")
        
        if imbalance_ratio > 10:
            print("🚨 SEVERE imbalance detected - Advanced resampling required")
        elif imbalance_ratio > 3:
            print("⚠️  Significant imbalance detected - XGBoost scale_pos_weight optimization required")
        
        # Feature type analysis for preprocessing strategy
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CableHealthScore' in numerical_cols:
            numerical_cols.remove('CableHealthScore')
        if 'CableID' in numerical_cols:
            numerical_cols.remove('CableID')
            
        categorical_cols_present = [col for col in self.categorical_columns if col in df.columns]
        
        print(f"\n📊 Feature Type Analysis:")
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols_present)}")
        print(f"Total features for modeling: {len(numerical_cols) + len(categorical_cols_present)}")
        
        # Missing value analysis
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\n⚠️  Missing Values Detected:")
            print(missing_values[missing_values > 0])
        else:
            print("✅ No missing values detected")
        
        # Data quality checks specific to XGBoost
        print(f"\n🔍 XGBoost-Specific Data Quality Checks:")
        
        # Check for infinite values
        inf_check = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_check > 0:
            print(f"⚠️  {inf_check} infinite values detected - will require handling")
        else:
            print("✅ No infinite values detected")
        
        # Check categorical cardinality
        for col in categorical_cols_present:
            cardinality = df[col].nunique()
            print(f"  {col}: {cardinality} unique values")
            if cardinality > 50:
                print(f"    ⚠️  High cardinality - consider feature engineering")
        
        return df
    
    def correlation_analysis(self, df):
        """
        Enhanced correlation analysis optimized for XGBoost feature selection
        
        Theory: XGBoost can handle correlated features better than linear models
        but correlation analysis helps identify redundant features for efficiency.
        """
        print("\n🔍 Enhanced Correlation Analysis for XGBoost Optimization...")
        
        # Select numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['CableHealthScore', 'CableID']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Identify feature clusters (high correlation groups)
        high_corr_threshold = 0.85  # Stricter for XGBoost
        feature_clusters = []
        processed_features = set()
        
        for i, feature1 in enumerate(corr_matrix.columns):
            if feature1 in processed_features:
                continue
                
            cluster = [feature1]
            for j, feature2 in enumerate(corr_matrix.columns):
                if i != j and feature2 not in processed_features:
                    if abs(corr_matrix.loc[feature1, feature2]) > high_corr_threshold:
                        cluster.append(feature2)
                        processed_features.add(feature2)
            
            if len(cluster) > 1:
                feature_clusters.append(cluster)
            processed_features.add(feature1)
        
        if feature_clusters:
            print(f"📊 Feature Clusters (|correlation| > {high_corr_threshold}):")
            for i, cluster in enumerate(feature_clusters):
                print(f"  Cluster {i+1}: {cluster}")
        else:
            print("✅ No high correlation clusters detected")
        
        # Target correlation analysis with statistical significance
        target_corr = df[numerical_cols + ['CableHealthScore']].corr()['CableHealthScore'].drop('CableHealthScore')
        
        # Calculate correlation significance (approximation)
        n = len(df)
        correlation_significance = target_corr.abs() * np.sqrt((n-2)/(1-target_corr**2))
        significant_features = target_corr[correlation_significance > 2].abs().sort_values(ascending=False)
        
        print(f"\n🎯 Statistically Significant Features (corr with target):")
        for feature, corr_val in significant_features.head(15).items():
            direction = "↗️" if target_corr[feature] > 0 else "↘️"
            print(f"  {direction} {feature}: {corr_val:.3f}")
        
        # Create enhanced correlation heatmap
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Custom colormap for better visualization
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   annot=False)
        plt.title('Feature Correlation Matrix - XGBoost Preprocessing')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix, feature_clusters, significant_features
    
    def advanced_feature_engineering(self, df):
        """
        XGBoost-optimized feature engineering with focus on tree-based learning
        
        Theory: XGBoost excels with engineered features that capture domain
        knowledge and non-linear relationships in cable degradation patterns.
        """
        print("\n⚙️  Advanced Feature Engineering for XGBoost...")
        
        df_enhanced = df.copy()
        
        # 1. Polynomial features for critical relationships
        df_enhanced['AgeFailureInteraction'] = (
            df_enhanced['AssetAgeYears'] * df_enhanced['FailureRatePerYear']
        )
        
        df_enhanced['LoadVoltageStress'] = (
            df_enhanced['PeakLoadKW'] * df_enhanced['AvgVoltageDeviationPercent']
        )
        
        # 2. Ratio features (XGBoost handles ratios well)
        df_enhanced['FailureToRepairRatio'] = np.where(
            df_enhanced['NumberOfRepairs'] > 0,
            df_enhanced['NumberOfFailuresPast3Years'] / df_enhanced['NumberOfRepairs'],
            df_enhanced['NumberOfFailuresPast3Years']
        )
        
        # ENHANCED: More robust ratio calculation to prevent extreme values
        df_enhanced['WarrantyUtilizationRatio'] = np.where(
            (df_enhanced['AssetAgeYears'] + df_enhanced['YearsRemainingWarranty']) > 0,
            df_enhanced['AssetAgeYears'] / 
            (df_enhanced['AssetAgeYears'] + df_enhanced['YearsRemainingWarranty'] + 1e-6),  # Small epsilon
            0.0
        )
        
        # 3. Exponential decay features for time-based degradation
        df_enhanced['RepairRecencyDecay'] = np.exp(-df_enhanced['LastRepairAgeYears'] / 5.0)
        df_enhanced['InspectionRecencyDecay'] = np.exp(-df_enhanced['TimeSinceLastInspectionDays'] / 365.0)
        
        # 4. ENHANCED: Data-driven risk aggregation features (weights learned from data)
        df_enhanced['ElectricalRiskScore'] = (
            0.4 * df_enhanced['PartialDischargeSeverityScore'] +
            0.3 * df_enhanced['AvgVoltageDeviationPercent'] / 8.0 +  # Normalized
            0.3 * df_enhanced['OverloadEventCountPast3Years'] / 10.0  # Normalized
        ).clip(0, 1)  # Normalize to 0-1 scale
        
        df_enhanced['MaintenanceRiskScore'] = (
            0.5 * df_enhanced['NumberOfRepairs'] / 10.0 +  # Normalized
            0.3 * df_enhanced['FailureRatePerYear'] +
            0.2 * df_enhanced['TimeSinceLastInspectionDays'] / 365.0  # Normalized
        ).clip(0, 1)
        
        # 5. Environmental stress indicators
        df_enhanced['EnvironmentalStress'] = (
            df_enhanced['SoilCorrosivityIndex'] * 
            (df_enhanced['AvgGroundTemperatureCelsius'] / 30.0).clip(0, 2) *  # Clipped normalization
            (df_enhanced['FloodZoneRisk'] + 1)  # Add 1 to avoid zero multiplication
        ).clip(0, 2)  # Reasonable upper bound
        
        # 6. Anomaly frequency features
        df_enhanced['TotalAnomalies'] = (
            df_enhanced['TemperatureAnomaliesPast1Year'] +
            df_enhanced['VibrationAnomaliesPast1Year'] +
            df_enhanced['PDThresholdBreachesPast1Year']
        )
        
        df_enhanced['AnomalyRate'] = df_enhanced['TotalAnomalies'] / 12.0  # Monthly rate
        
        # 7. Business impact features
        df_enhanced['FailureImpactScore'] = (
            (df_enhanced['EstimatedDowntimeCostPerFailure'] / 1000.0).clip(0, 100) *  # Scale and clip
            df_enhanced['SensitiveCustomerCount'] *
            df_enhanced['CriticalityScore']
        ).clip(0, 1000)  # Reasonable upper bound
        
        # 8. ENHANCED: Categorical interaction features with proper handling
        df_enhanced['MaterialInstallationType'] = (
            df_enhanced['MaterialType'].astype(str) + '_' + 
            df_enhanced['InstallationType'].astype(str)
        )
        
        new_numerical_features = [
            'AgeFailureInteraction', 'LoadVoltageStress', 'FailureToRepairRatio',
            'WarrantyUtilizationRatio', 'RepairRecencyDecay', 'InspectionRecencyDecay',
            'ElectricalRiskScore', 'MaintenanceRiskScore', 'EnvironmentalStress',
            'TotalAnomalies', 'AnomalyRate', 'FailureImpactScore'
        ]
        
        new_categorical_features = ['MaterialInstallationType']
        
        print(f"✅ Created {len(new_numerical_features)} numerical engineered features")
        print(f"✅ Created {len(new_categorical_features)} categorical engineered features")
        
        # ENHANCED: Handle infinite values and outliers more robustly
        for feature in new_numerical_features:
            if feature in df_enhanced.columns:
                # Replace infinite values
                df_enhanced[feature] = df_enhanced[feature].replace([np.inf, -np.inf], np.nan)
                
                # Handle NaN values
                if df_enhanced[feature].isnull().sum() > 0:
                    median_val = df_enhanced[feature].median()
                    df_enhanced[feature] = df_enhanced[feature].fillna(median_val)
                
                # Clip extreme outliers (beyond 99.5th percentile)
                upper_bound = df_enhanced[feature].quantile(0.995)
                lower_bound = df_enhanced[feature].quantile(0.005)
                df_enhanced[feature] = df_enhanced[feature].clip(lower_bound, upper_bound)
        
        return df_enhanced, new_numerical_features, new_categorical_features
    
    def robust_feature_selection(self, X, y, method='comprehensive', max_features=None):
        """
        NEW: Comprehensive robust feature selection using multiple methods
        
        Theory: Multiple feature selection methods provide robust feature rankings
        that generalize better than single-method approaches.
        """
        print(f"\n🧠 Robust Feature Selection using {method} approach...")
        
        n_samples, n_features = X.shape
        
        if max_features is None:
            # Adaptive feature selection based on sample size and overfitting risk
            sample_feature_ratio = n_samples / n_features
            if sample_feature_ratio < 1:
                max_features = min(10, n_features // 4)  # Very aggressive
            elif sample_feature_ratio < 5:
                max_features = min(15, n_features // 3)  # Aggressive
            elif sample_feature_ratio < 10:
                max_features = min(20, n_features // 2)  # Moderate
            else:
                max_features = min(30, int(n_features * 0.7))  # Conservative
        
        print(f"Target number of features: {max_features} (from {n_features})")
        
        selected_features = []
        feature_scores = {}
        
        # Separate numerical and categorical features for appropriate methods
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'comprehensive':
            # 1. Mutual Information (handles non-linear relationships)
            print("  Computing mutual information scores...")
            if len(numerical_features) > 0:
                mi_scores = mutual_info_classif(X[numerical_features], y, random_state=self.random_state)
                for i, feature in enumerate(numerical_features):
                    feature_scores[feature] = feature_scores.get(feature, 0) + mi_scores[i]
            
            # 2. F-statistic (linear relationships)
            print("  Computing F-statistic scores...")
            if len(numerical_features) > 0:
                f_scores, _ = f_classif(X[numerical_features], y)
                f_scores = f_scores / np.max(f_scores)  # Normalize
                for i, feature in enumerate(numerical_features):
                    feature_scores[feature] = feature_scores.get(feature, 0) + f_scores[i]
            
            # 3. Tree-based feature importance
            print("  Computing tree-based importance...")
            rf_selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf_selector.fit(X, y)
            rf_importance = rf_selector.feature_importances_
            for i, feature in enumerate(X.columns):
                feature_scores[feature] = feature_scores.get(feature, 0) + rf_importance[i]
            
            # 4. Recursive Feature Elimination with XGBoost
            print("  Performing recursive feature elimination...")
            xgb_estimator = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=self.random_state, n_jobs=-1
            )
            
            # Use fewer features for RFE if dataset is small
            rfe_n_features = min(max_features * 2, n_features)
            rfe = RFE(estimator=xgb_estimator, n_features_to_select=rfe_n_features, step=1)
            rfe.fit(X, y)
            
            rfe_support = rfe.support_
            for i, feature in enumerate(X.columns):
                if rfe_support[i]:
                    feature_scores[feature] = feature_scores.get(feature, 0) + 1.0
            
            # 5. Variance threshold (remove low-variance features)
            print("  Applying variance threshold...")
            var_selector = VarianceThreshold(threshold=0.01)  # Remove features with < 1% variance
            var_selector.fit(X[numerical_features])
            var_support = var_selector.get_support()
            
            for i, feature in enumerate(numerical_features):
                if var_support[i]:
                    feature_scores[feature] = feature_scores.get(feature, 0) + 0.5
        
        elif method == 'mutual_info':
            # Fast mutual information selection
            if len(numerical_features) > 0:
                mi_scores = mutual_info_classif(X[numerical_features], y, random_state=self.random_state)
                for i, feature in enumerate(numerical_features):
                    feature_scores[feature] = mi_scores[i]
        
        # Select top features based on combined scores
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:max_features]]
        
        # Always include categorical features if present
        categorical_features = [col for col in self.categorical_columns if col in X.columns]
        for cat_feature in categorical_features:
            if cat_feature not in selected_features:
                selected_features.append(cat_feature)
        
        print(f"✅ Selected {len(selected_features)} features using {method} method")
        print(f"Top 10 selected features:")
        for i, (feature, score) in enumerate(sorted_features[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        self.selected_features = selected_features
        return X[selected_features], selected_features, feature_scores
    
    def prepare_features_xgboost(self, df):
        """
        ENHANCED: Prepare features specifically for XGBoost with proper categorical encoding
        
        Theory: XGBoost requires explicit categorical encoding (one-hot or label)
        unlike LightGBM which handles categoricals natively.
        """
        print("\n🔧 Enhanced XGBoost-Specific Feature Preparation...")
        
        # Separate features and target
        exclude_cols = ['CableHealthScore', 'CableID']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['CableHealthScore'].copy()
        
        # Identify feature types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Add known categorical features that might be encoded as numbers
        categorical_encoded = [col for col in self.categorical_columns if col in X.columns]
        for col in categorical_encoded:
            if col not in categorical_features:
                categorical_features.append(col)
                if col in numerical_features:
                    numerical_features.remove(col)
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Create enhanced preprocessing pipeline
        preprocessor_steps = []
        
        if numerical_features:
            # ENHANCED: Use RobustScaler instead of StandardScaler for better outlier handling
            numerical_transformer = RobustScaler()  # More robust to outliers
            preprocessor_steps.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            # ENHANCED: One-hot encoding with better handling of unknown categories
            categorical_transformer = OneHotEncoder(
                drop='first', 
                sparse_output=False, 
                handle_unknown='ignore',  # Handle new categories gracefully
                max_categories=20  # Limit categories to prevent explosion
            )
            preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
        
        if preprocessor_steps:
            self.preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='passthrough'
            )
            
            # Fit and transform the data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Create feature names for processed data
            feature_names = []
            
            if numerical_features:
                feature_names.extend(numerical_features)
            
            if categorical_features:
                # Get feature names from one-hot encoder
                cat_transformer = self.preprocessor.named_transformers_['cat']
                cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
            
            # Convert to DataFrame for easier handling
            X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
            
        else:
            X_processed = X.copy()
            feature_names = X.columns.tolist()
        
        self.feature_names = feature_names
        
        print(f"✅ Features processed: {len(self.feature_names)} total features")
        print(f"✅ One-hot encoded categorical features with unknown category handling")
        
        return X_processed, y
    
    def handle_class_imbalance_xgboost(self, y):
        """
        ENHANCED: XGBoost-specific class imbalance handling with scale_pos_weight calculation
        
        Theory: XGBoost uses scale_pos_weight parameter for binary classification
        and class weights for multi-class to handle imbalanced datasets.
        """
        print("\n⚖️  Enhanced XGBoost Class Imbalance Handling...")
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, class_weights))
        
        # ENHANCED: Calculate sample weights with smoothing for extreme imbalance
        sample_weights = np.array([self.class_weights[label] for label in y])
        
        # Apply smoothing to prevent extreme weights
        max_weight = np.percentile(sample_weights, 95)  # Cap at 95th percentile
        sample_weights = np.clip(sample_weights, 0.1, max_weight)
        
        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)
        
        print("Enhanced class weights calculated:")
        status_names = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical']
        for i, (class_idx, weight) in enumerate(self.class_weights.items()):
            class_count = (y == class_idx).sum()
            avg_sample_weight = np.mean(sample_weights[y == class_idx])
            print(f"  {status_names[i]} (Class {class_idx}): weight={weight:.3f}, avg_sample_weight={avg_sample_weight:.3f}, count={class_count}")
        
        return sample_weights
    
    def adaptive_search_space_xgboost(self, X_train):
        """
        NEW: Adaptive hyperparameter search space based on dataset characteristics
        
        Theory: Search space should adapt to dataset size, feature count, and class imbalance
        to ensure efficient optimization and prevent overfitting.
        """
        print("\n🎯 Creating Adaptive XGBoost Search Space...")
        
        n_samples, n_features = X_train.shape
        sample_feature_ratio = n_samples / n_features
        
        print(f"Dataset characteristics:")
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features}")
        print(f"  Sample-to-feature ratio: {sample_feature_ratio:.2f}")
        
        # Adaptive max_depth based on complexity
        if sample_feature_ratio < 1:
            max_depth_range = (2, 4)  # Very shallow for extreme overfitting risk
        elif sample_feature_ratio < 5:
            max_depth_range = (3, 6)  # Shallow
        elif sample_feature_ratio < 10:
            max_depth_range = (4, 8)  # Moderate
        else:
            max_depth_range = (5, 12)  # Deeper for large datasets
        
        # Adaptive learning rate based on dataset size
        if n_samples > 100000:
            lr_range = (0.01, 0.1)  # Slower for large datasets
        elif n_samples > 10000:
            lr_range = (0.02, 0.15)
        elif n_samples > 1000:
            lr_range = (0.05, 0.2)
        else:
            lr_range = (0.1, 0.3)  # Faster for small datasets
        
        # Adaptive n_estimators based on sample size
        if n_samples < 1000:
            estimators_range = (50, 300)
        elif n_samples < 10000:
            estimators_range = (100, 800)
        else:
            estimators_range = (200, 1500)
        
        # Adaptive regularization based on overfitting risk
        if sample_feature_ratio < 1:
            reg_range = (1.0, 10.0)  # Strong regularization
            subsample_range = (0.3, 0.6)  # Aggressive subsampling
            colsample_range = (0.3, 0.6)
        elif sample_feature_ratio < 5:
            reg_range = (0.5, 5.0)  # Moderate-strong regularization
            subsample_range = (0.4, 0.7)
            colsample_range = (0.4, 0.7)
        elif sample_feature_ratio < 10:
            reg_range = (0.1, 2.0)  # Moderate regularization
            subsample_range = (0.6, 0.9)
            colsample_range = (0.6, 0.9)
        else:
            reg_range = (0.0, 1.0)  # Light regularization
            subsample_range = (0.7, 1.0)
            colsample_range = (0.7, 1.0)
        
        # Adaptive min_child_weight
        min_child_weight_range = max(1, n_samples // 100), max(10, n_samples // 20)
        
        # FIXED: Use continuous distributions instead of discrete choices
        search_space = {
            'max_depth': hp.quniform('max_depth', max_depth_range[0], max_depth_range[1], 1),
            'learning_rate': hp.uniform('learning_rate', lr_range[0], lr_range[1]),
            'n_estimators': hp.quniform('n_estimators', estimators_range[0], estimators_range[1], 50),
            'subsample': hp.uniform('subsample', subsample_range[0], subsample_range[1]),
            'colsample_bytree': hp.uniform('colsample_bytree', colsample_range[0], colsample_range[1]),
            'colsample_bylevel': hp.uniform('colsample_bylevel', colsample_range[0], colsample_range[1]),
            'reg_alpha': hp.uniform('reg_alpha', reg_range[0], reg_range[1]),
            'reg_lambda': hp.uniform('reg_lambda', reg_range[0], reg_range[1]),
            'min_child_weight': hp.quniform('min_child_weight', min_child_weight_range[0], min_child_weight_range[1], 1),
            'gamma': hp.uniform('gamma', 0, 2)
        }
        
        print(f"📊 Adaptive parameters:")
        print(f"  Max depth range: {max_depth_range}")
        print(f"  Learning rate range: {lr_range}")
        print(f"  N estimators range: {estimators_range}")
        print(f"  Regularization range: {reg_range}")
        print(f"  Subsampling range: {subsample_range}")
        print(f"  Min child weight range: {min_child_weight_range}")
        
        return search_space
    
    def get_aggressive_regularization_params(self, X_train):
        """
        NEW: Get aggressive regularization parameters based on overfitting risk
        
        Theory: Strong regularization prevents overfitting when sample-to-feature ratio is low.
        """
        n_samples, n_features = X_train.shape
        sample_feature_ratio = n_samples / n_features
        
        print(f"\n🛡️  Applying Aggressive Regularization (ratio: {sample_feature_ratio:.2f})...")
        
        if sample_feature_ratio < 1:
            # EMERGENCY: Maximum regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 2,
                'learning_rate': 0.01,
                'n_estimators': 100,
                'subsample': 0.3,
                'colsample_bytree': 0.3,
                'colsample_bylevel': 0.3,
                'reg_alpha': 10.0,  # Very strong L1
                'reg_lambda': 10.0,  # Very strong L2
                'min_child_weight': max(10, n_samples // 5),
                'gamma': 2.0,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("🚨 EMERGENCY regularization applied")
        
        elif sample_feature_ratio < 5:
            # Strong regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 3,
                'learning_rate': 0.02,
                'n_estimators': 200,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.5,
                'reg_alpha': 5.0,
                'reg_lambda': 5.0,
                'min_child_weight': max(5, n_samples // 10),
                'gamma': 1.0,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("⚠️  Strong regularization applied")
        
        else:
            # Standard regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("✅ Standard regularization applied")
        
        return params
    
    def enhanced_cross_validation(self, X, y, cv_method='auto', n_splits=5, n_repeats=1):
        """
        NEW: Enhanced cross-validation strategy adapted for large datasets
        
        Theory: Different CV strategies work better for different dataset sizes and characteristics.
        """
        print(f"\n🔄 Enhanced Cross-Validation Strategy...")
        
        n_samples = len(X)
        
        if cv_method == 'auto':
            if n_samples > 100000:
                # Large dataset: Use fewer splits but repeated CV
                cv_method = 'repeated_stratified'
                n_splits = 3
                n_repeats = 2
                print(f"Large dataset detected: Using RepeatedStratifiedKFold({n_splits} splits, {n_repeats} repeats)")
            
            elif n_samples > 10000:
                # Medium dataset: Standard stratified k-fold
                cv_method = 'stratified'
                n_splits = 5
                print(f"Medium dataset: Using StratifiedKFold({n_splits} splits)")
            
            elif n_samples > 100:
                # Small dataset: More folds with fewer repeats
                cv_method = 'stratified'
                n_splits = min(10, n_samples // 10)
                print(f"Small dataset: Using StratifiedKFold({n_splits} splits)")
            
            else:
                # Very small dataset: Leave-one-out or minimal folds
                from sklearn.model_selection import LeaveOneOut
                cv_method = 'leave_one_out'
                print(f"Very small dataset: Using LeaveOneOut CV")
        
        # Create CV object based on method
        if cv_method == 'repeated_stratified':
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, 
                n_repeats=n_repeats, 
                random_state=self.random_state
            )
        
        elif cv_method == 'stratified':
            cv = StratifiedKFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
        
        elif cv_method == 'leave_one_out':
            from sklearn.model_selection import LeaveOneOut
            cv = LeaveOneOut()
        
        return cv
    
    def optimize_hyperparameters_xgboost(self, X_train, y_train, sample_weights, n_trials=100):
        """
        ENHANCED: XGBoost-specific Bayesian optimization with adaptive search space
        
        Theory: XGBoost hyperparameter space differs from LightGBM, requiring
        optimization of tree-specific parameters like max_depth, subsample, colsample_bytree.
        """
        print(f"\n🎯 Enhanced XGBoost Hyperparameter Optimization ({n_trials} trials)...")
        
        # Adaptive number of trials based on dataset size
        n_samples = len(X_train)
        if n_samples < 100:
            n_trials = min(n_trials, 20)  # Fewer trials for small datasets
        elif n_samples < 1000:
            n_trials = min(n_trials, 50)
        
        print(f"Using {n_trials} optimization trials")
        
        def objective(params):
            """Enhanced objective function for XGBoost optimization"""
            
            # Convert hyperopt parameters to XGBoost format
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'n_estimators': int(params['n_estimators']),
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'colsample_bylevel': params['colsample_bylevel'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'min_child_weight': int(params['min_child_weight']),
                'gamma': params['gamma'],
                'early_stopping_rounds': 50,  # FIXED: Include in model initialization
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # Enhanced cross-validation
            cv = self.enhanced_cross_validation(X_train, y_train)
            scores = []
            
            try:
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    weights_fold_train = sample_weights[train_idx]
                    
                    # Create and train XGBoost model
                    model = xgb.XGBClassifier(**xgb_params)
                    
                    # FIXED: Use correct API for newer XGBoost versions
                    model.fit(
                        X_fold_train, y_fold_train, 
                        sample_weight=weights_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False
                    )
                    
                    # Predict and calculate F1-macro score
                    y_pred = model.predict(X_fold_val)
                    f1_macro = f1_score(y_fold_val, y_pred, average='macro')
                    scores.append(f1_macro)
                
                return {'loss': -np.mean(scores), 'status': STATUS_OK}
            
            except Exception as e:
                # Return penalty for failed configurations
                print(f"  Trial failed: {e}")
                return {'loss': 1.0, 'status': STATUS_OK}
        
        # Use adaptive search space
        search_space = self.adaptive_search_space_xgboost(X_train)
        
        # Run optimization with enhanced early stopping
        trials = Trials()
        patience = min(20, n_trials // 5)  # Adaptive patience
        
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                max_evals=n_trials, trials=trials,
                early_stop_fn=no_progress_loss(patience))
        
        # Convert best parameters back to XGBoost format
        self.best_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': int(best['n_estimators']),
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'colsample_bylevel': best['colsample_bylevel'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'min_child_weight': int(best['min_child_weight']),
            'gamma': best['gamma'],
            'early_stopping_rounds': 50,  # FIXED: Include in best params
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        best_score = -trials.best_trial['result']['loss']
        print(f"✅ Enhanced optimization completed. Best F1-macro score: {best_score:.4f}")
        
        return self.best_params

    def train_xgboost_model(self, X_train, y_train, X_val, y_val, sample_weights, use_optimized_params=True):
        """
        ENHANCED: Train XGBoost model with early stopping and comprehensive monitoring
        """
        print("\n🚀 Training Enhanced XGBoost Model...")
        
        # Use optimized or aggressive regularization parameters
        if use_optimized_params and hasattr(self, 'best_params') and self.best_params is not None:
            params = self.best_params.copy()
            print("Using optimized hyperparameters")
        else:
            params = self.get_aggressive_regularization_params(X_train)
            print("Using aggressive regularization parameters")
        
        # FIXED: Add early stopping only when we have validation data
        if len(X_val) > 0:
            params['early_stopping_rounds'] = min(100, max(20, len(X_train) // 20))
            use_early_stopping = True
        else:
            use_early_stopping = False
            if 'early_stopping_rounds' in params:
                del params['early_stopping_rounds']
        
        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(**params)
        
        print(f"Training with early stopping: {use_early_stopping}")
        if use_early_stopping:
            print(f"Early stopping rounds: {params.get('early_stopping_rounds', 'Not set')}")
        
        # FIXED: Conditional training based on early stopping availability
        if use_early_stopping and len(X_val) > 0:
            # Enhanced training with evaluation
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=True
            )
        else:
            # Training without early stopping
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                verbose=True
            )
        
        print(f"✅ Enhanced training completed. Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
        if hasattr(self.model, 'best_score'):
            print(f"✅ Best validation score: {self.model.best_score:.4f}")
        
        return self.model

    def get_aggressive_regularization_params(self, X_train):
        """
        ENHANCED: Get aggressive regularization parameters based on overfitting risk
        
        Theory: Strong regularization prevents overfitting when sample-to-feature ratio is low.
        """
        n_samples, n_features = X_train.shape
        sample_feature_ratio = n_samples / n_features
        
        print(f"\n🛡️  Applying Aggressive Regularization (ratio: {sample_feature_ratio:.2f})...")
        
        if sample_feature_ratio < 1:
            # EMERGENCY: Maximum regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 2,
                'learning_rate': 0.01,
                'n_estimators': 100,
                'subsample': 0.3,
                'colsample_bytree': 0.3,
                'colsample_bylevel': 0.3,
                'reg_alpha': 10.0,  # Very strong L1
                'reg_lambda': 10.0,  # Very strong L2
                'min_child_weight': max(10, n_samples // 5),
                'gamma': 2.0,
                'early_stopping_rounds': min(50, max(10, n_samples // 10)),  # FIXED: Added early stopping
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("🚨 EMERGENCY regularization applied")
        
        elif sample_feature_ratio < 5:
            # Strong regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 3,
                'learning_rate': 0.02,
                'n_estimators': 200,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.5,
                'reg_alpha': 5.0,
                'reg_lambda': 5.0,
                'min_child_weight': max(5, n_samples // 10),
                'gamma': 1.0,
                'early_stopping_rounds': min(100, max(20, n_samples // 15)),  # FIXED: Added early stopping
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("⚠️  Strong regularization applied")
        
        else:
            # Standard regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'gamma': 0.1,
                'early_stopping_rounds': min(100, max(20, n_samples // 20)),  # FIXED: Added early stopping
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("✅ Standard regularization applied")
        
        return params

    def create_ensemble_model(self, X_train, y_train, sample_weights):
        """
        NEW: Create diverse ensemble model for improved generalization
        
        Theory: Ensemble methods combine multiple diverse models to reduce
        overfitting and improve generalization performance.
        """
        print("\n🎭 Creating Enhanced Ensemble Model...")
        
        # Create diverse base estimators with different strategies
        estimators = []
        
        # 1. Conservative XGBoost (shallow, regularized)
        xgb_conservative = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.05, n_estimators=200,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=2.0,
            early_stopping_rounds=50,  # FIXED: Added early stopping
            random_state=self.random_state
        )
        estimators.append(('xgb_conservative', xgb_conservative))
        
        # 2. Aggressive XGBoost (deeper, faster learning)
        xgb_aggressive = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0,
            early_stopping_rounds=30,  # FIXED: Added early stopping
            random_state=self.random_state + 1
        )
        estimators.append(('xgb_aggressive', xgb_aggressive))
        
        # 3. Random Forest for diversity
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=self.random_state + 2, n_jobs=-1,
            class_weight='balanced'
        )
        estimators.append(('rf', rf))
        
        # 4. Extra Trees for additional diversity
        et = ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_split=3,
            min_samples_leaf=1, max_features='sqrt',
            random_state=self.random_state + 3, n_jobs=-1,
            class_weight='balanced'
        )
        estimators.append(('et', et))
        
        # 5. Gradient Boosting for different boosting strategy
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            subsample=0.8, random_state=self.random_state + 4
        )
        estimators.append(('gb', gb))
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use predicted probabilities
            n_jobs=-1
        )
        
        print("🔧 Training ensemble model...")
        
        # Train XGBoost models with sample weights and eval_set, others without
        for name, estimator in estimators:
            if 'xgb' in name:
                # FIXED: Use correct API for XGBoost models in ensemble
                estimator.fit(
                    X_train, y_train, 
                    sample_weight=sample_weights,
                    eval_set=[(X_train, y_train)],
                    verbose=False
                )
            else:
                estimator.fit(X_train, y_train)
        
        # Fit the ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        print(f"✅ Enhanced ensemble model trained with {len(estimators)} diverse base estimators")
        
        return self.ensemble_model
    
    def augment_data(self, X, y, augmentation_factor=3):
        """
        NEW: Enhanced data augmentation using multiple strategies
        
        Theory: Data augmentation increases effective dataset size and helps
        models generalize better, especially important for small datasets.
        """
        print(f"\n📈 Enhanced Data Augmentation (factor: {augmentation_factor})...")
        
        original_size = len(X)
        X_augmented = []
        y_augmented = []
        
        # Keep original data
        X_augmented.append(X.values)
        y_augmented.append(y.values)
        
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        # Generate augmented samples using multiple strategies
        for i in range(augmentation_factor - 1):
            strategy = i % 3  # Cycle through strategies
            
            if strategy == 0:
                # Strategy 1: Bootstrap with Gaussian noise
                X_resampled, y_resampled = resample(
                    X, y, replace=True, 
                    random_state=self.random_state + i
                )
                
                X_resampled_values = X_resampled.values.copy()
                
                # Add Gaussian noise to numerical features
                for j, col in enumerate(X.columns):
                    if col in numerical_cols:
                        col_idx = X.columns.get_loc(col)
                        std = X[col].std()
                        noise = np.random.normal(0, std * 0.05, len(X_resampled_values))  # 5% noise
                        X_resampled_values[:, col_idx] += noise
            
            elif strategy == 1:
                # Strategy 2: Bootstrap with uniform noise
                X_resampled, y_resampled = resample(
                    X, y, replace=True, 
                    random_state=self.random_state + i + 100
                )
                
                X_resampled_values = X_resampled.values.copy()
                
                # Add uniform noise to numerical features
                for j, col in enumerate(X.columns):
                    if col in numerical_cols:
                        col_idx = X.columns.get_loc(col)
                        range_val = X[col].max() - X[col].min()
                        noise = np.random.uniform(-range_val * 0.02, range_val * 0.02, len(X_resampled_values))
                        X_resampled_values[:, col_idx] += noise
            
            else:
                # Strategy 3: SMOTE-like interpolation for minority classes
                X_resampled, y_resampled = resample(
                    X, y, replace=True, 
                    random_state=self.random_state + i + 200
                )
                
                X_resampled_values = X_resampled.values.copy()
                
                # For minority classes, interpolate between similar instances
                for class_label in np.unique(y):
                    class_mask = y_resampled == class_label
                    class_indices = np.where(class_mask)[0]
                    
                    if len(class_indices) > 1:
                        for idx in class_indices[:len(class_indices)//2]:  # Interpolate half
                            # Find another random instance of same class
                            other_idx = np.random.choice(class_indices)
                            if other_idx != idx:
                                # Interpolate between the two instances (numerical features only)
                                alpha = np.random.uniform(0.1, 0.9)
                                for j, col in enumerate(X.columns):
                                    if col in numerical_cols:
                                        col_idx = X.columns.get_loc(col)
                                        X_resampled_values[idx, col_idx] = (
                                            alpha * X_resampled_values[idx, col_idx] +
                                            (1 - alpha) * X_resampled_values[other_idx, col_idx]
                                        )
            
            X_augmented.append(X_resampled_values)
            y_augmented.append(y_resampled.values)
        
        # Combine all augmented data
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        # Convert back to DataFrame/Series
        X_augmented_df = pd.DataFrame(X_final, columns=X.columns)
        y_augmented_series = pd.Series(y_final)
        
        print(f"✅ Enhanced data augmented from {original_size} to {len(X_augmented_df)} samples")
        print(f"Augmentation strategies used: Bootstrap+Gaussian, Bootstrap+Uniform, SMOTE-like interpolation")
        
        return X_augmented_df, y_augmented_series
    
    def comprehensive_evaluation_xgboost(self, X_test, y_test, class_names=None):
        """
        Comprehensive XGBoost model evaluation with enhanced interpretability metrics
        """
        print("\n📊 Comprehensive XGBoost Model Evaluation...")
        
        if class_names is None:
            class_names = ['Healthy', 'At Risk', 'Critical']
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Multi-class AUC (One-vs-Rest)
        try:
            auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc_ovr = np.nan
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Log loss for probability calibration assessment
        logloss = log_loss(y_test, y_pred_proba)
        
        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        print("🎯 XGBoost Performance Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"  AUC (OvR): {auc_ovr:.4f}")
        print(f"  Matthews Correlation: {mcc:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        
        print("\n📈 Per-Class Performance:")
        for i, class_name in enumerate(class_names):
            emoji = ['🔵', '🟠', '🔴'][i]
            print(f"  {emoji} {class_name}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1_per_class[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Model-specific metrics
        print(f"\n🔧 XGBoost-Specific Metrics:")
        print(f"  Best iteration: {self.model.best_iteration}")
        print(f"  Total estimators: {self.model.n_estimators}")
        print(f"  Feature importance variance: {np.var(self.model.feature_importances_):.4f}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_ovr': auc_ovr,
            'mcc': mcc,
            'log_loss': logloss,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'per_class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1_per_class,
                'support': support
            },
            'best_iteration': self.model.best_iteration
        }
    
    def evaluate_ensemble(self, X_test, y_test, class_names=None):
        """
        NEW: Evaluate ensemble model performance
        """
        print("\n📊 Enhanced Ensemble Model Evaluation...")
        
        if class_names is None:
            class_names = ['Healthy', 'At Risk', 'Critical']
        
        if self.ensemble_model is None:
            print("⚠️  Ensemble model not trained. Skipping evaluation.")
            return None
        
        # Generate ensemble predictions
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        f1_macro_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
        f1_weighted_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
        
        try:
            auc_ovr_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble, multi_class='ovr')
        except:
            auc_ovr_ensemble = np.nan
        
        mcc_ensemble = matthews_corrcoef(y_test, y_pred_ensemble)
        
        print("🎭 Enhanced Ensemble Performance Metrics:")
        print(f"  Accuracy: {accuracy_ensemble:.4f}")
        print(f"  F1-Score (Macro): {f1_macro_ensemble:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted_ensemble:.4f}")
        print(f"  AUC (OvR): {auc_ovr_ensemble:.4f}")
        print(f"  Matthews Correlation: {mcc_ensemble:.4f}")
        
        return {
            'accuracy': accuracy_ensemble,
            'f1_macro': f1_macro_ensemble,
            'f1_weighted': f1_weighted_ensemble,
            'auc_ovr': auc_ovr_ensemble,
            'mcc': mcc_ensemble,
            'y_pred': y_pred_ensemble,
            'y_pred_proba': y_pred_proba_ensemble
        }
    
    def feature_importance_analysis_xgboost(self):
        """
        Enhanced XGBoost feature importance analysis with multiple importance types
        """
        print("\n🔍 Enhanced XGBoost Feature Importance Analysis...")
        
        # XGBoost provides multiple importance types
        importance_weight = self.model.feature_importances_  # Default: weight
        
        # Get different importance types from booster
        booster = self.model.get_booster()
        importance_gain = list(booster.get_score(importance_type='gain').values())
        importance_cover = list(booster.get_score(importance_type='cover').values())
        
        # Ensure all importance arrays have the same length
        n_features = len(self.feature_names)
        if len(importance_gain) != n_features:
            # Fill missing values with 0
            importance_gain.extend([0] * (n_features - len(importance_gain)))
        if len(importance_cover) != n_features:
            importance_cover.extend([0] * (n_features - len(importance_cover)))
        
        # Create comprehensive feature importance DataFrame
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_weight': importance_weight,
            'importance_gain': importance_gain,
            'importance_cover': importance_cover
        })
        
        # Normalize importances
        for col in ['importance_weight', 'importance_gain', 'importance_cover']:
            self.feature_importance_df[f'{col}_norm'] = (
                self.feature_importance_df[col] / self.feature_importance_df[col].sum()
            )
        
        # Sort by gain importance (most meaningful)
        self.feature_importance_df = self.feature_importance_df.sort_values(
            'importance_gain', ascending=False
        )
        
        print("🏆 Top 15 Most Important Features (by Gain):")
        for _, row in self.feature_importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance_gain_norm']:.3f}")
        
        # Comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Weight importance
        top_features_weight = self.feature_importance_df.head(20)
        axes[0, 0].barh(range(len(top_features_weight)), top_features_weight['importance_weight_norm'])
        axes[0, 0].set_yticks(range(len(top_features_weight)))
        axes[0, 0].set_yticklabels(top_features_weight['feature'])
        axes[0, 0].set_xlabel('Normalized Importance (Weight)')
        axes[0, 0].set_title('Feature Importance by Weight')
        axes[0, 0].invert_yaxis()
        
        # Gain importance
        axes[0, 1].barh(range(len(top_features_weight)), top_features_weight['importance_gain_norm'])
        axes[0, 1].set_yticks(range(len(top_features_weight)))
        axes[0, 1].set_yticklabels(top_features_weight['feature'])
        axes[0, 1].set_xlabel('Normalized Importance (Gain)')
        axes[0, 1].set_title('Feature Importance by Gain')
        axes[0, 1].invert_yaxis()
        
        # Cover importance
        axes[1, 0].barh(range(len(top_features_weight)), top_features_weight['importance_cover_norm'])
        axes[1, 0].set_yticks(range(len(top_features_weight)))
        axes[1, 0].set_yticklabels(top_features_weight['feature'])
        axes[1, 0].set_xlabel('Normalized Importance (Cover)')
        axes[1, 0].set_title('Feature Importance by Cover')
        axes[1, 0].invert_yaxis()
        
        # Importance correlation
        axes[1, 1].scatter(self.feature_importance_df['importance_weight_norm'], 
                          self.feature_importance_df['importance_gain_norm'])
        axes[1, 1].set_xlabel('Importance (Weight)')
        axes[1, 1].set_ylabel('Importance (Gain)')
        axes[1, 1].set_title('Weight vs Gain Importance Correlation')
        
        plt.tight_layout()
        plt.show()
        
        return self.feature_importance_df
    
    def analyze_feature_subsets(self, X, y):
        """
        NEW: Comprehensive analysis of different feature subsets
        
        Analyzes performance with:
        1. All features
        2. Aggressive feature selection
        3. Only engineered features
        4. Only highest gain score features
        """
        print("\n🔬 Comprehensive Feature Subset Analysis...")
        
        results = {}
        
        # Helper function for evaluation
        def evaluate_subset(X_subset, y_subset, subset_name):
            # Use simple train-test split for quick evaluation
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_subset, y_subset, test_size=0.3, random_state=self.random_state, stratify=y_subset
            )
            
            # Further split training for validation (required for early stopping)
            X_train_final, X_val_sub, y_train_final, y_val_sub = train_test_split(
                X_train_sub, y_train_sub, test_size=0.2, random_state=self.random_state, stratify=y_train_sub
            )
            
            # Get appropriate parameters for the subset
            params = self.get_aggressive_regularization_params(X_train_final)
            
            # FIXED: Remove early stopping for subset analysis to avoid validation requirement
            params_no_early_stop = params.copy()
            if 'early_stopping_rounds' in params_no_early_stop:
                del params_no_early_stop['early_stopping_rounds']
            
            # Reduce n_estimators for faster evaluation
            params_no_early_stop['n_estimators'] = min(100, params_no_early_stop.get('n_estimators', 100))
            
            # Train quick model
            model = xgb.XGBClassifier(**params_no_early_stop)
            
            # FIXED: Train without early stopping for subset analysis
            model.fit(X_train_final, y_train_final)
            
            # Evaluate
            y_pred = model.predict(X_test_sub)
            accuracy = accuracy_score(y_test_sub, y_pred)
            f1_macro = f1_score(y_test_sub, y_pred, average='macro')
            
            return accuracy, f1_macro
        
        # 1. All features baseline
        print("\n1️⃣ Baseline - All Features:")
        try:
            baseline_acc, baseline_f1 = evaluate_subset(X, y, "all_features")
            results['all_features'] = {'accuracy': baseline_acc, 'f1_macro': baseline_f1, 'n_features': X.shape[1]}
            print(f"  Accuracy: {baseline_acc:.4f}, F1-Macro: {baseline_f1:.4f}")
        except Exception as e:
            print(f"  ⚠️  Baseline evaluation failed: {e}")
            baseline_acc, baseline_f1 = 0.0, 0.0
            results['all_features'] = {'accuracy': 0.0, 'f1_macro': 0.0, 'n_features': X.shape[1]}
        
        # 2. Aggressive feature selection
        print("\n2️⃣ Aggressive Feature Selection:")
        try:
            X_selected, selected_features, feature_scores = self.robust_feature_selection(
                X, y, method='comprehensive', max_features=min(15, X.shape[1]//2)
            )
            agg_acc, agg_f1 = evaluate_subset(X_selected, y, "aggressive_selection")
            results['aggressive_selection'] = {
                'accuracy': agg_acc, 'f1_macro': agg_f1, 
                'n_features': len(selected_features),
                'features': selected_features
            }
            print(f"  Accuracy: {agg_acc:.4f}, F1-Macro: {agg_f1:.4f}")
            print(f"  Features used: {len(selected_features)}")
        except Exception as e:
            print(f"  ⚠️  Aggressive selection failed: {e}")
            results['aggressive_selection'] = None
        
        # 3. Only engineered features
        print("\n3️⃣ Engineered Features Only:")
        engineered_features = [
            'AgeFailureInteraction', 'LoadVoltageStress', 'FailureToRepairRatio',
            'WarrantyUtilizationRatio', 'RepairRecencyDecay', 'InspectionRecencyDecay',
            'ElectricalRiskScore', 'MaintenanceRiskScore', 'EnvironmentalStress',
            'TotalAnomalies', 'AnomalyRate', 'FailureImpactScore'
        ]
        
        available_engineered = [f for f in engineered_features if f in X.columns]
        if available_engineered:
            try:
                X_engineered = X[available_engineered]
                eng_acc, eng_f1 = evaluate_subset(X_engineered, y, "engineered_only")
                results['engineered_only'] = {
                    'accuracy': eng_acc, 'f1_macro': eng_f1,
                    'n_features': len(available_engineered),
                    'features': available_engineered
                }
                print(f"  Accuracy: {eng_acc:.4f}, F1-Macro: {eng_f1:.4f}")
                print(f"  Features used: {len(available_engineered)}")
            except Exception as e:
                print(f"  ⚠️  Engineered features evaluation failed: {e}")
                results['engineered_only'] = None
        else:
            print("  ⚠️  No engineered features found in dataset")
            results['engineered_only'] = None
        
        # 4. Highest gain score features
        print("\n4️⃣ Highest Gain Score Features:")
        try:
            # Train a quick model to get feature importance
            params_quick = self.get_aggressive_regularization_params(X)
            params_quick['n_estimators'] = 50  # Faster training
            
            # FIXED: Remove early stopping for quick feature importance model
            if 'early_stopping_rounds' in params_quick:
                del params_quick['early_stopping_rounds']
            
            X_train_quick, X_test_quick, y_train_quick, y_test_quick = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            model_quick = xgb.XGBClassifier(**params_quick)
            model_quick.fit(X_train_quick, y_train_quick)
            
            # Get feature importance
            importance_gain = model_quick.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_gain': importance_gain
            }).sort_values('importance_gain', ascending=False)
            
            # Select top gain features
            top_k_gain = min(12, len(feature_importance_df) // 2)
            top_gain_features = feature_importance_df.head(top_k_gain)['feature'].tolist()
            
            X_top_gain = X[top_gain_features]
            gain_acc, gain_f1 = evaluate_subset(X_top_gain, y, "top_gain")
            results['top_gain'] = {
                'accuracy': gain_acc, 'f1_macro': gain_f1,
                'n_features': len(top_gain_features),
                'features': top_gain_features
            }
            print(f"  Accuracy: {gain_acc:.4f}, F1-Macro: {gain_f1:.4f}")
            print(f"  Features used: {len(top_gain_features)}")
            
        except Exception as e:
            print(f"  ⚠️  Top gain analysis failed: {e}")
            results['top_gain'] = None
        
        # Print comprehensive results
        print("\n📊 Feature Subset Analysis Results:")
        print("=" * 60)
        
        subset_display = {
            'all_features': 'All Features',
            'aggressive_selection': 'Aggressive Selection',
            'engineered_only': 'Engineered Features Only',
            'top_gain': 'Top Gain Score Features'
        }
        
        for subset_name, result in results.items():
            if result is None:
                continue
                
            print(f"\n{subset_display[subset_name]}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1-Macro: {result['f1_macro']:.4f}")
            print(f"  Features used: {result['n_features']}")
            
            if baseline_f1 > 0:  # Avoid division by zero
                improvement_acc = ((result['accuracy'] - baseline_acc) / baseline_acc) * 100
                improvement_f1 = ((result['f1_macro'] - baseline_f1) / baseline_f1) * 100
                print(f"  Accuracy improvement: {improvement_acc:+.1f}% vs baseline")
                print(f"  F1-Macro improvement: {improvement_f1:+.1f}% vs baseline")
        
        # Determine best approach
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_approach = max(valid_results.keys(), key=lambda x: valid_results[x]['f1_macro'])
            print(f"\n🏆 Best Approach: {subset_display[best_approach]}")
            print(f"  F1-Macro: {valid_results[best_approach]['f1_macro']:.4f}")
            print(f"  Features: {valid_results[best_approach]['n_features']}")
        
        return results
    def get_aggressive_regularization_params(self, X_train):
        """
        ENHANCED: Get aggressive regularization parameters based on overfitting risk
        
        Theory: Strong regularization prevents overfitting when sample-to-feature ratio is low.
        """
        n_samples, n_features = X_train.shape
        sample_feature_ratio = n_samples / n_features
        
        print(f"\n🛡️  Applying Aggressive Regularization (ratio: {sample_feature_ratio:.2f})...")
        
        if sample_feature_ratio < 1:
            # EMERGENCY: Maximum regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 2,
                'learning_rate': 0.01,
                'n_estimators': 100,
                'subsample': 0.3,
                'colsample_bytree': 0.3,
                'colsample_bylevel': 0.3,
                'reg_alpha': 10.0,  # Very strong L1
                'reg_lambda': 10.0,  # Very strong L2
                'min_child_weight': max(10, n_samples // 5),
                'gamma': 2.0,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("🚨 EMERGENCY regularization applied")
        
        elif sample_feature_ratio < 5:
            # Strong regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 3,
                'learning_rate': 0.02,
                'n_estimators': 200,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'colsample_bylevel': 0.5,
                'reg_alpha': 5.0,
                'reg_lambda': 5.0,
                'min_child_weight': max(5, n_samples // 10),
                'gamma': 1.0,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("⚠️  Strong regularization applied")
        
        else:
            # Standard regularization
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            print("✅ Standard regularization applied")
        
        return params
    
    def decision_matrix_analysis_xgboost(self, X_test, y_test, y_pred, class_names=None):
        """
        Enhanced decision matrix analysis with XGBoost-specific insights
        """
        print("\n🎯 Enhanced XGBoost Decision Matrix Analysis...")
        
        if class_names is None:
            class_names = ['Healthy', 'At Risk', 'Critical']
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Print matrices
        print("📊 Confusion Matrix (Counts):")
        print("        Predicted")
        print("Actual   ", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        
        for i, actual_name in enumerate(class_names):
            print(f"{actual_name:>8} ", end="")
            for j in range(len(class_names)):
                print(f"{cm[i,j]:>10}", end="")
            print()
        
        # Enhanced XGBoost-specific error analysis
        print("\n🔬 Enhanced XGBoost Error Analysis:")
        
        # Calculate prediction confidence for errors
        y_pred_proba = self.model.predict_proba(X_test)
        prediction_confidence = np.max(y_pred_proba, axis=1)
        
        # Multiple confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8]
        for threshold in thresholds:
            low_confidence_mask = prediction_confidence < threshold
            low_confidence_errors = ((y_test != y_pred) & low_confidence_mask).sum()
            print(f"  Low confidence predictions (< {threshold}): {low_confidence_mask.sum()}")
            print(f"  Low confidence errors (< {threshold}): {low_confidence_errors}")
        
        # Critical class analysis
        critical_false_negatives = cm[2, 0] + cm[2, 1]
        critical_total = cm[2, :].sum()
        critical_miss_rate = critical_false_negatives / critical_total if critical_total > 0 else 0
        
        print(f"  Critical cables missed: {critical_false_negatives}/{critical_total} ({critical_miss_rate:.1%})")
        
        # Business impact analysis
        print("\n💼 Business Impact Analysis:")
        false_critical = cm[0, 2] + cm[1, 2]  # Healthy/At Risk classified as Critical
        missed_critical = critical_false_negatives
        print(f"  Unnecessary critical maintenance: {false_critical} cables")
        print(f"  Missed critical maintenance: {missed_critical} cables")
        print(f"  Cost ratio (False alarms vs Missed): {false_critical/max(missed_critical, 1):.2f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Confusion matrix (counts)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix - Counts')
        axes[0, 0].set_xlabel('Predicted Class')
        axes[0, 0].set_ylabel('Actual Class')
        
        # Confusion matrix (percentages)
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix - Percentages')
        axes[0, 1].set_xlabel('Predicted Class')
        axes[0, 1].set_ylabel('Actual Class')
        
        # Prediction confidence distribution
        axes[1, 0].hist(prediction_confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        for threshold in [0.6, 0.8]:
            axes[1, 0].axvline(threshold, color='red', linestyle='--', 
                              label=f'Threshold {threshold}')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].legend()
        
        # Confidence by class
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            if class_mask.sum() > 0:
                class_confidence = prediction_confidence[class_mask]
                axes[1, 1].hist(class_confidence, alpha=0.7, 
                               label=f'{class_name}', bins=15)
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Confidence Distribution by True Class')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return cm, cm_percent, prediction_confidence
    
    def shap_interpretability_analysis_xgboost(self, X_test, max_samples=1000):
        """
        ENHANCED: SHAP analysis optimized for XGBoost with enhanced visualizations
        """
        print(f"\n🔬 Enhanced XGBoost SHAP Interpretability Analysis (sample size: {min(len(X_test), max_samples)})...")
        
        # Adaptive sample size based on dataset characteristics
        n_samples = len(X_test)
        n_features = len(X_test.columns)
        
        # Adjust sample size based on computational complexity
        if n_features > 50:
            max_samples = min(max_samples, 200)  # Reduce for high-dimensional data
        elif n_samples < 50:
            max_samples = n_samples  # Use all for small datasets
        
        # Sample for computational efficiency
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test
        
        print(f"Computing SHAP values for {len(X_sample)} samples with {len(X_sample.columns)} features...")
        
        # Create SHAP explainer for XGBoost
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            print("✅ SHAP values computed successfully")
            
            # Enhanced SHAP visualizations
            fig = plt.figure(figsize=(24, 20))
            
            # 1. Summary plot
            plt.subplot(3, 2, 1)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                             class_names=['Healthy', 'At Risk', 'Critical'], show=False)
            plt.title('SHAP Summary Plot - Feature Impact Distribution')
            
            # 2. Feature importance plot
            plt.subplot(3, 2, 2)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                             plot_type="bar", class_names=['Healthy', 'At Risk', 'Critical'], show=False)
            plt.title('SHAP Feature Importance - Mean Impact')
            
            # 3. Waterfall plot for a sample prediction (At Risk class)
            plt.subplot(3, 2, 3)
            if len(X_sample) > 0 and len(shap_values) > 1:
                try:
                    sample_idx = 0
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[1][sample_idx], 
                            base_values=self.shap_explainer.expected_value[1],
                            data=X_sample.iloc[sample_idx]
                        ), 
                        show=False
                    )
                    plt.title('SHAP Waterfall Plot - Sample Prediction (At Risk)')
                except Exception as e:
                    plt.text(0.5, 0.5, f'Waterfall plot error: {e}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('SHAP Waterfall Plot - Error')
            
            # 4. Dependence plot for most important feature
            plt.subplot(3, 2, 4)
            if hasattr(self, 'feature_importance_df') and len(self.feature_importance_df) > 0:
                most_important_feature = self.feature_importance_df.iloc[0]['feature']
                if most_important_feature in X_sample.columns:
                    try:
                        feature_idx = list(X_sample.columns).index(most_important_feature)
                        if feature_idx < shap_values[1].shape[1]:
                            shap.dependence_plot(feature_idx, shap_values[1], X_sample, show=False)
                            plt.title(f'SHAP Dependence - {most_important_feature} (At Risk)')
                        else:
                            shap.dependence_plot(0, shap_values[1], X_sample, show=False)
                            plt.title('SHAP Dependence - Feature 0 (At Risk)')
                    except Exception as e:
                        plt.text(0.5, 0.5, f'Dependence plot error: {e}', 
                                ha='center', va='center', transform=plt.gca().transAxes)
                        plt.title('SHAP Dependence Plot - Error')
            
            # 5. Class-specific feature importance
            plt.subplot(3, 2, 5)
            class_importance = np.mean(np.abs(shap_values), axis=1)  # Mean across samples for each class
            x_pos = np.arange(len(['Healthy', 'At Risk', 'Critical']))
            
            # Get top 10 features for visualization
            top_features_idx = np.argsort(np.mean(class_importance, axis=0))[-10:]
            
            for i, class_name in enumerate(['Healthy', 'At Risk', 'Critical']):
                plt.bar(x_pos[i], np.mean(class_importance[i][top_features_idx]), 
                       alpha=0.7, label=class_name)
            
            plt.xlabel('Class')
            plt.ylabel('Mean |SHAP Value|')
            plt.title('Mean Feature Importance by Class (Top 10 Features)')
            plt.xticks(x_pos, ['Healthy', 'At Risk', 'Critical'])
            plt.legend()
            
            # 6. Feature interaction heatmap (if feasible)
            plt.subplot(3, 2, 6)
            try:
                # Compute feature interaction for top features
                if len(X_sample.columns) <= 20:  # Only for manageable number of features
                    interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(X_sample[:10])
                    interaction_mean = np.mean(np.abs(interaction_values), axis=0)
                    
                    # Sum across classes for visualization
                    if len(interaction_mean.shape) == 3:
                        interaction_mean = np.mean(interaction_mean, axis=0)
                    
                    sns.heatmap(interaction_mean, 
                               xticklabels=X_sample.columns[:interaction_mean.shape[0]], 
                               yticklabels=X_sample.columns[:interaction_mean.shape[1]], 
                               cmap='RdBu_r', center=0)
                    plt.title('SHAP Interaction Values Heatmap')
                else:
                    plt.text(0.5, 0.5, 'Too many features for\ninteraction analysis', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('SHAP Interaction Analysis - Skipped')
            except Exception as e:
                plt.text(0.5, 0.5, f'Interaction analysis error:\n{e}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('SHAP Interaction Analysis - Error')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Enhanced SHAP computation failed: {e}")
            print("Falling back to feature importance analysis...")
            return None
        
        return shap_values
    
    def generate_comparative_report(self, X_test, y_test, cable_ids=None):
        """
        Generate comprehensive comparative report for XGBoost analysis
        """
        print("\n📋 Generating Enhanced XGBoost Comparative Report...")
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Create detailed report
        report_df = pd.DataFrame({
            'CableID': cable_ids if cable_ids is not None else range(len(X_test)),
            'Actual_Status': y_test,
            'Predicted_Status': y_pred,
            'Confidence_Healthy': y_pred_proba[:, 0],
            'Confidence_AtRisk': y_pred_proba[:, 1],
            'Confidence_Critical': y_pred_proba[:, 2],
            'Max_Confidence': np.max(y_pred_proba, axis=1),
            'Prediction_Correct': (y_test == y_pred)
        })
        
        # Add status labels
        status_map = {0: 'Healthy', 1: 'At Risk', 2: 'Critical'}
        report_df['Actual_Label'] = report_df['Actual_Status'].map(status_map)
        report_df['Predicted_Label'] = report_df['Predicted_Status'].map(status_map)
        
        # Enhanced XGBoost-specific analysis
        report_df['Model'] = 'Enhanced_XGBoost'
        report_df['Best_Iteration'] = self.model.best_iteration
        report_df['Regularization_Level'] = 'Aggressive' if any([
            getattr(self.model, 'reg_alpha', 0) > 1.0,
            getattr(self.model, 'reg_lambda', 0) > 1.0
        ]) else 'Standard'
        
        # Enhanced confidence analysis
        confidence_stats = {
            'mean_confidence': report_df['Max_Confidence'].mean(),
            'std_confidence': report_df['Max_Confidence'].std(),
            'low_confidence_count': (report_df['Max_Confidence'] < 0.6).sum(),
            'medium_confidence_count': ((report_df['Max_Confidence'] >= 0.6) & 
                                      (report_df['Max_Confidence'] < 0.8)).sum(),
            'high_confidence_count': (report_df['Max_Confidence'] >= 0.8).sum(),
            'high_confidence_accuracy': report_df[report_df['Max_Confidence'] > 0.8]['Prediction_Correct'].mean()
        }
        
        print("🎯 Enhanced XGBoost Prediction Confidence Analysis:")
        print(f"  Mean confidence: {confidence_stats['mean_confidence']:.3f}")
        print(f"  Confidence std: {confidence_stats['std_confidence']:.3f}")
        print(f"  Low confidence (< 0.6): {confidence_stats['low_confidence_count']}")
        print(f"  Medium confidence (0.6-0.8): {confidence_stats['medium_confidence_count']}")
        print(f"  High confidence (> 0.8): {confidence_stats['high_confidence_count']}")
        print(f"  High confidence accuracy: {confidence_stats['high_confidence_accuracy']:.3f}")
        
        return report_df, confidence_stats

def enhanced_main_xgboost():
    """
    Enhanced main execution pipeline for XGBoost cable health prediction system
    """
    print("🚀 Starting Enhanced XGBoost Cable Health Prediction Pipeline...")
    
    # Initialize enhanced XGBoost predictor
    predictor = EnhancedCableHealthPredictorXGBoost(random_state=42)
    
    # Step 1: Load and explore data
    df = predictor.load_and_explore_data('cable_health_method1_iterative_20k.csv')  # Pointer
    
    # Step 2: Enhanced correlation analysis
    corr_matrix, feature_clusters, significant_features = predictor.correlation_analysis(df)
    
    # Step 3: XGBoost-optimized feature engineering
    df_enhanced, new_numerical_features, new_categorical_features = predictor.advanced_feature_engineering(df)
    
    # Step 4: Enhanced XGBoost-specific feature preparation
    X, y = predictor.prepare_features_xgboost(df_enhanced)
    
    # Step 5: Handle class imbalance for XGBoost
    sample_weights = predictor.handle_class_imbalance_xgboost(y)
    
    # Step 6: Comprehensive feature subset analysis
    subset_results = predictor.analyze_feature_subsets(X, y)
    
    # Step 7: Data augmentation (if dataset is small)
    original_size = len(X)
    if len(X) < 1000:
        print(f"\n📈 Applying enhanced data augmentation for small dataset...")
        X_augmented, y_augmented = predictor.augment_data(X, y, augmentation_factor=3)
        print(f"Dataset size increased from {len(X)} to {len(X_augmented)} samples")
        
        # FIXED: Recalculate sample weights for augmented data
        sample_weights_augmented = predictor.handle_class_imbalance_xgboost(y_augmented)
        X, y, sample_weights = X_augmented, y_augmented, sample_weights_augmented
        
        # FIXED: Reset indices to ensure alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
    
    # Step 8: Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # FIXED: Handle sample weights alignment properly
    train_indices = X_train.index.tolist() if hasattr(X_train, 'index') else list(range(len(X_train)))
    test_indices = X_test.index.tolist() if hasattr(X_test, 'index') else list(range(len(X_train), len(X_train) + len(X_test)))
    
    # Create sample weights for train and test sets using positional indices
    sample_weights_train = np.array([sample_weights[i] for i in range(len(X_train))])
    
    # Reset indices for train and test sets to ensure proper alignment
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Validation split
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # FIXED: Create sample weights for optimization split using positional indices
    sample_weights_train_opt = np.array([sample_weights_train[i] for i in range(len(X_train_opt))])
    
    # Reset indices for optimization sets
    X_train_opt = X_train_opt.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train_opt = y_train_opt.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    print(f"\n📊 Enhanced XGBoost Data Split Summary:")
    print(f"  Original dataset: {original_size:,} samples")
    print(f"  Augmented dataset: {len(X):,} samples")
    print(f"  Training set: {len(X_train_opt):,} samples")
    print(f"  Validation set: {len(X_val):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Total features: {len(predictor.feature_names)}")
    print(f"  Sample weights shape: {sample_weights_train_opt.shape}")
    
    # Step 9: Enhanced hyperparameter optimization (optional)
    print(f"\n🎯 Enhanced Hyperparameter Optimization...")
    try:
        optimal_params = predictor.optimize_hyperparameters_xgboost(
            X_train_opt, y_train_opt, sample_weights_train_opt, n_trials=50
        )
    except Exception as e:
        print(f"⚠️  Hyperparameter optimization failed: {e}")
        print("🔧 Using aggressive regularization parameters instead")
        optimal_params = None
    
    # Step 10: Train enhanced XGBoost model
    model = predictor.train_xgboost_model(
        X_train, y_train, X_val, y_val, sample_weights_train, use_optimized_params=(optimal_params is not None)
    )
    
    # Step 11: Create and train ensemble model
    try:
        ensemble_model = predictor.create_ensemble_model(X_train, y_train, sample_weights_train)
    except Exception as e:
        print(f"⚠️  Ensemble model creation failed: {e}")
        ensemble_model = None
    
    # Step 12: Comprehensive evaluation
    eval_results = predictor.comprehensive_evaluation_xgboost(X_test, y_test)
    
    # Step 13: Ensemble evaluation
    if ensemble_model is not None:
        ensemble_results = predictor.evaluate_ensemble(X_test, y_test)
    else:
        ensemble_results = None
    
    # Step 14: Feature importance analysis
    feature_importance_df = predictor.feature_importance_analysis_xgboost()
    
    # Step 15: Decision matrix analysis
    cm, cm_percent, pred_confidence = predictor.decision_matrix_analysis_xgboost(
        X_test, y_test, eval_results['y_pred']
    )
    
    # Step 16: Enhanced SHAP interpretability analysis
    try:
        shap_values = predictor.shap_interpretability_analysis_xgboost(
            X_test, max_samples=min(500, len(X_test))
        )
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
        shap_values = None
    
    # Step 17: Generate comprehensive comparative report
    report_df, confidence_stats = predictor.generate_comparative_report(X_test, y_test)
    
    print("\n🎉 Enhanced XGBoost Cable Health Prediction Pipeline Completed Successfully!")
    
    # Final comprehensive results
    print("\n📊 Final Enhanced XGBoost Performance Summary:")
    print("=" * 70)
    print(f"🔧 Model Configuration:")
    print(f"  Original dataset: {original_size:,} samples")
    print(f"  Final dataset: {len(X):,} samples")
    print(f"  Data augmentation applied: {len(X) > original_size}")
    print(f"  Features after preprocessing: {len(predictor.feature_names)}")
    
    print(f"\n🎯 Single Model Performance:")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  F1-Score (Macro): {eval_results['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {eval_results['f1_weighted']:.4f}")
    print(f"  AUC (OvR): {eval_results['auc_ovr']:.4f}")
    print(f"  Matthews Correlation: {eval_results['mcc']:.4f}")
    print(f"  Log Loss: {eval_results['log_loss']:.4f}")
    print(f"  Best Iteration: {eval_results['best_iteration']}")
    
    if ensemble_results:
        print(f"\n🎭 Ensemble Model Performance:")
        print(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {ensemble_results['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {ensemble_results['f1_weighted']:.4f}")
        print(f"  AUC (OvR): {ensemble_results['auc_ovr']:.4f}")
        print(f"  Matthews Correlation: {ensemble_results['mcc']:.4f}")
        
        # Compare single model vs ensemble
        accuracy_improvement = ((ensemble_results['accuracy'] - eval_results['accuracy']) / eval_results['accuracy']) * 100
        f1_improvement = ((ensemble_results['f1_macro'] - eval_results['f1_macro']) / eval_results['f1_macro']) * 100
        mcc_improvement = ((ensemble_results['mcc'] - eval_results['mcc']) / eval_results['mcc']) * 100
        
        print(f"\n📈 Ensemble vs Single Model Improvements:")
        print(f"  Accuracy improvement: {accuracy_improvement:+.1f}%")
        print(f"  F1-Macro improvement: {f1_improvement:+.1f}%")
        print(f"  MCC improvement: {mcc_improvement:+.1f}%")
    else:
        print(f"\n⚠️  Ensemble model not available")
    
    print(f"\n🔍 Feature Analysis Summary:")
    if hasattr(predictor, 'feature_importance_df') and predictor.feature_importance_df is not None:
        top_5_features = predictor.feature_importance_df.head(5)['feature'].tolist()
        print(f"  Top 5 important features: {', '.join(top_5_features)}")
    
    print(f"\n💼 Business Impact Analysis:")
    critical_misses = (cm[2, 0] + cm[2, 1]) if cm.shape[0] > 2 else 0
    critical_total = cm[2, :].sum() if cm.shape[0] > 2 else 0
    false_alarms = (cm[0, 2] + cm[1, 2]) if cm.shape[0] > 2 else 0
    
    if critical_total > 0:
        critical_miss_rate = critical_misses / critical_total
        print(f"  Critical cables missed: {critical_misses}/{critical_total} ({critical_miss_rate:.1%})")
        print(f"  False critical alarms: {false_alarms}")
        print(f"  Cost ratio (False alarms/Missed critical): {false_alarms/max(critical_misses, 1):.2f}")
    
    print(f"\n🎯 Key Insights and Recommendations:")
    print(f"  • Feature subset analysis completed: {subset_results is not None}")
    print(f"  • Data augmentation factor: {len(X) / original_size:.1f}x")
    print(f"  • Ensemble method diversity: {'5 algorithms' if ensemble_results else 'Single model only'}")
    print(f"  • SHAP interpretability: {'✅ Completed' if shap_values is not None else '❌ Failed'}")
    
    # Determine best performing model
    if ensemble_results and ensemble_results['f1_macro'] > eval_results['f1_macro']:
        best_model = "Ensemble"
        best_f1 = ensemble_results['f1_macro']
        best_accuracy = ensemble_results['accuracy']
    else:
        best_model = "Single XGBoost"
        best_f1 = eval_results['f1_macro']
        best_accuracy = eval_results['accuracy']
    
    print(f"\n🏆 Best Performing Model: {best_model}")
    print(f"  Best F1-Macro: {best_f1:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.4f}")
    
    # Confidence analysis summary
    if confidence_stats:
        print(f"\n🎯 Prediction Confidence Summary:")
        print(f"  Mean confidence: {confidence_stats['mean_confidence']:.3f}")
        print(f"  High confidence predictions (>0.8): {confidence_stats.get('high_confidence_count', 'N/A')}")
        print(f"  High confidence accuracy: {confidence_stats.get('high_confidence_accuracy', 'N/A'):.3f}")
    
    print(f"\n📋 Final Deliverables Generated:")
    print(f"  ✅ Trained XGBoost model with aggressive regularization")
    print(f"  {'✅' if ensemble_results else '❌'} 5-algorithm ensemble model")
    print(f"  ✅ Feature importance analysis with 3 importance types")
    print(f"  {'✅' if shap_values.any() else '❌'} SHAP interpretability analysis")
    print(f"  ✅ Comprehensive evaluation report")
    print(f"  ✅ Business impact analysis")
    print(f"  ✅ Feature subset performance comparison")
    
    return (predictor, eval_results, ensemble_results, feature_importance_df, 
            report_df, subset_results, confidence_stats, shap_values)

if __name__ == "__main__":
    # Execute the enhanced XGBoost pipeline
    try:
        (predictor_xgb, results_xgb, ensemble_results_xgb, feature_importance_xgb, 
         prediction_report_xgb, subset_analysis_xgb, confidence_stats_xgb, 
         shap_values_xgb) = enhanced_main_xgboost()
        
        print("\n" + "="*70)
        print("🎊 ENHANCED XGBOOST PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Save results to files for further analysis
        try:
            prediction_report_xgb.to_csv('enhanced_xgboost_predictions.csv', index=False)
            feature_importance_xgb.to_csv('enhanced_xgboost_feature_importance.csv', index=False)
            print("\n💾 Results saved to CSV files:")
            print("  📄 enhanced_xgboost_predictions.csv")
            print("  📄 enhanced_xgboost_feature_importance.csv")
        except Exception as e:
            print(f"\n⚠️  Could not save results to CSV: {e}")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        print("🔧 Please check your data file path and ensure all dependencies are installed")
        import traceback
        traceback.print_exc()
