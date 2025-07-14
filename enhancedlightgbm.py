"""
Enhanced Cable Health Prediction System using LightGBM
=====================================================
Advanced predictive analytics for cable health assessment with 3-class classification:
- 0: Healthy (Good condition, normal risk)
- 1: At Risk (Deterioration signs, preventive maintenance needed)  
- 2: Critical (Poor condition, immediate action required)

Algorithm Choice: LightGBM selected for optimal performance on large tabular datasets
with mixed feature types and class imbalance handling capabilities.

ENHANCEMENTS:
- Adaptive hyperparameter search space
- Nested cross-validation for reliable performance estimation
- Mutual information feature selection
- Enhanced regularization for large datasets
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
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_recall_fscore_support,
                           roc_auc_score, matthews_corrcoef)

# Advanced Analytics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils import resample
import shap
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🔧 Enhanced Cable Health Prediction System Initialized")
print("📊 Algorithm: LightGBM (Gradient Boosting Machine)")
print("🎯 Task: Multi-class Classification (Healthy/At Risk/Critical)")
print("🚀 NEW: Adaptive optimization, Nested CV, Ensemble methods, Data augmentation")

class EnhancedCableHealthPredictor:
    """
    Enhanced Cable Health Prediction System using LightGBM
    
    This class implements a comprehensive machine learning pipeline for predicting
    cable health status with emphasis on handling class imbalance and providing
    interpretable results for maintenance decision-making.
    
    ENHANCEMENTS:
    - Adaptive hyperparameter optimization
    - Nested cross-validation
    - Mutual information feature selection
    - Enhanced regularization
    - Ensemble methods
    - Data augmentation capabilities
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.ensemble_model = None
        self.feature_names = None
        self.categorical_features = None
        self.class_weights = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        self.feature_selector = None
        self.selected_features = None
        self.nested_cv_results = {}
        
        # Define categorical features based on domain knowledge
        self.categorical_columns = [
            'MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory',
            'FloodZoneRisk'
        ]
        
        print(f"✅ Enhanced CableHealthPredictor initialized with random_state={random_state}")
    
    def load_and_explore_data(self, filepath):
        """
        Load cable health dataset and perform comprehensive exploratory analysis
        
        Theory: Understanding data distribution is crucial for model selection and
        hyperparameter tuning, especially for imbalanced classification tasks.
        """
        print("\n📂 Loading and Exploring Dataset...")
        
        # Load data with proper handling of mixed data types
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Enhanced sample size analysis
        n_samples, n_features = df.shape[0], df.shape[1] - 1  # Exclude target
        sample_to_feature_ratio = n_samples / n_features
        print(f"Sample-to-feature ratio: {sample_to_feature_ratio:.2f}")
        
        if sample_to_feature_ratio < 10:
            print("⚠️  WARNING: Sample-to-feature ratio < 10:1. Risk of overfitting detected!")
            print("🔧 RECOMMENDATION: Use aggressive feature selection and enhanced regularization")
        elif sample_to_feature_ratio < 5:
            print("🚨 CRITICAL: Sample-to-feature ratio < 5:1. High overfitting risk!")
            print("🔧 MANDATORY: Feature selection, data augmentation, and ensemble methods required")
        
        # Class distribution analysis - Critical for imbalanced learning
        print("\n🎯 Target Variable Analysis:")
        class_counts = df['CableHealthScore'].value_counts().sort_index()
        class_props = df['CableHealthScore'].value_counts(normalize=True).sort_index()
        
        for i, (count, prop) in enumerate(zip(class_counts, class_props)):
            status = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical'][i]
            print(f"Class {i} ({status}): {count:,} samples ({prop:.1%})")
        
        # Calculate imbalance ratio for optimization strategy
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        
        # Enhanced imbalance assessment
        if imbalance_ratio > 10:
            print("🚨 SEVERE imbalance detected - Advanced resampling required")
        elif imbalance_ratio > 5:
            print("⚠️  MODERATE imbalance detected - Class weighting recommended")
        else:
            print("✅ Balanced dataset detected")
        
        # Feature analysis
        print(f"\n📊 Feature Analysis:")
        print(f"Total features: {len(df.columns) - 1}")
        print(f"Numerical features: {len(df.select_dtypes(include=[np.number]).columns) - 1}")
        print(f"Categorical features: {len(self.categorical_columns)}")
        
        # Missing value analysis
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\n⚠️  Missing Values Detected:")
            print(missing_values[missing_values > 0])
        else:
            print("✅ No missing values detected")
        
        return df
    
    def correlation_analysis(self, df):
        """
        Perform comprehensive correlation analysis to identify feature relationships
        and potential multicollinearity issues affecting model interpretability.
        """
        print("\n🔍 Enhanced Correlation Analysis...")
        
        # Select numerical features for correlation analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove('CableHealthScore')  # Remove target
        
        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Identify highly correlated feature pairs (|correlation| > 0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"⚠️  High Correlation Pairs Detected (|r| > 0.8):")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("✅ No concerning multicollinearity detected")
        
        # Target correlation analysis
        target_corr = df[numerical_cols + ['CableHealthScore']].corr()['CableHealthScore'].drop('CableHealthScore')
        top_correlated = target_corr.abs().sort_values(ascending=False).head(10)
        
        print(f"\n🎯 Top 10 Features Correlated with Cable Health:")
        for feature, corr_val in top_correlated.items():
            direction = "↗️" if target_corr[feature] > 0 else "↘️"
            print(f"  {direction} {feature}: {corr_val:.3f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix, high_corr_pairs
    
    def advanced_feature_engineering(self, df):
        """
        Perform domain-specific feature engineering based on cable health expertise
        
        Theory: Feature engineering leverages domain knowledge to create meaningful
        predictors that capture complex relationships in cable degradation patterns.
        """
        print("\n⚙️  Advanced Feature Engineering...")
        
        df_enhanced = df.copy()
        
        # 1. Age-related risk indicators
        df_enhanced['RemainingLifeRatio'] = df_enhanced['YearsRemainingWarranty'] / df_enhanced['RatedLifespanYears']
        df_enhanced['AgeUtilizationRatio'] = df_enhanced['AssetAgeYears'] / df_enhanced['RatedLifespanYears']
        
        # 2. Failure rate trends (combining multiple failure indicators)
        df_enhanced['CompoundFailureRate'] = (
            df_enhanced['FailureRatePerYear'] * df_enhanced['NumberOfFailuresPast3Years'] * 
            (df_enhanced['NumberOfRepairs'] + 1)
        )
        
        # 3. Electrical stress indicators
        df_enhanced['ElectricalStressScore'] = (
            df_enhanced['AvgVoltageDeviationPercent'] * df_enhanced['PeakLoadKW'] * 
            df_enhanced['OverloadEventCountPast3Years']
        )
        
        # 4. Maintenance responsiveness
        df_enhanced['MaintenanceResponsiveness'] = np.where(
            df_enhanced['NumberOfRepairs'] > 0,
            df_enhanced['TimeSinceLastInspectionDays'] / df_enhanced['NumberOfRepairs'],
            df_enhanced['TimeSinceLastInspectionDays']
        )
        
        # 5. Environmental risk composite
        df_enhanced['EnvironmentalRiskScore'] = (
            df_enhanced['SoilCorrosivityIndex'] * df_enhanced['FloodZoneRisk'] * 
            (df_enhanced['AvgGroundTemperatureCelsius'] / 30.0)  # Normalized
        )
        
        # 6. Critical load indicator
        df_enhanced['CriticalLoadIndicator'] = (
            df_enhanced['SensitiveCustomerCount'] * df_enhanced['ConnectedLoadKW'] * 
            df_enhanced['CriticalityScore']
        )
        
        # 7. Recent anomaly clustering
        df_enhanced['RecentAnomalyCluster'] = (
            df_enhanced['TemperatureAnomaliesPast1Year'] + 
            df_enhanced['VibrationAnomaliesPast1Year'] + 
            df_enhanced['PDThresholdBreachesPast1Year']
        )
        
        new_features = [
            'RemainingLifeRatio', 'AgeUtilizationRatio', 'CompoundFailureRate',
            'ElectricalStressScore', 'MaintenanceResponsiveness', 
            'EnvironmentalRiskScore', 'CriticalLoadIndicator', 'RecentAnomalyCluster'
        ]
        
        print(f"✅ Created {len(new_features)} engineered features:")
        for feature in new_features:
            print(f"  • {feature}")
        
        return df_enhanced, new_features
    
    def mutual_info_feature_selection(self, X, y, k='auto'):
        """
        NEW: Mutual information-based feature selection for non-linear relationships
        
        Theory: Mutual information captures non-linear dependencies between features
        and target, making it superior to correlation-based methods for tree models.
        """
        print(f"\n🧠 Mutual Information Feature Selection...")
        
        if k == 'auto':
            # Adaptive feature selection based on sample size
            n_samples = len(X)
            if n_samples < 100:
                k = min(10, X.shape[1] // 2)
            elif n_samples < 1000:
                k = min(15, X.shape[1] // 2)
            else:
                k = min(20, X.shape[1] // 2)
        
        # Calculate mutual information scores
        numerical_features = X.select_dtypes(include=[np.number]).columns
        X_numerical = X[numerical_features]
        
        mi_scores = mutual_info_classif(X_numerical, y, random_state=self.random_state)
        
        # Create feature importance DataFrame
        mi_df = pd.DataFrame({
            'feature': numerical_features,
            'mutual_info_score': mi_scores
        }).sort_values('mutual_info_score', ascending=False)
        
        # Select top k features
        selected_features = mi_df.head(k)['feature'].tolist()
        
        # Add categorical features if they exist
        categorical_features = [col for col in self.categorical_columns if col in X.columns]
        selected_features.extend(categorical_features)
        
        print(f"📊 Selected {len(selected_features)} features based on mutual information:")
        print(f"Top 10 by MI score:")
        for _, row in mi_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['mutual_info_score']:.4f}")
        
        self.selected_features = selected_features
        return X[selected_features], selected_features, mi_df
    
    def prepare_features(self, df):
        """
        Prepare features for LightGBM training with optimal preprocessing
        
        Theory: LightGBM handles categorical features natively, requiring minimal
        preprocessing while maintaining interpretability and performance.
        """
        print("\n🔧 Feature Preparation...")
        
        # Separate features and target
        X = df.drop(['CableHealthScore', 'CableID'], axis=1)
        y = df['CableHealthScore']
        
        # Store feature names for interpretability
        self.feature_names = X.columns.tolist()
        
        # Identify categorical features present in data
        self.categorical_features = [col for col in self.categorical_columns if col in X.columns]
        
        print(f"Features prepared: {len(self.feature_names)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        return X, y
    
    def handle_class_imbalance(self, y):
        """
        Calculate optimal class weights for imbalanced cable health classification
        
        Theory: Class weighting inversely proportional to class frequencies helps
        LightGBM focus on minority classes (At Risk, Critical) which are often
        more important for maintenance decisions.
        """
        print("\n⚖️  Handling Class Imbalance...")
        
        # Compute class weights using sklearn's balanced approach
        classes = np.unique(y)
        self.class_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Create weight dictionary for LightGBM
        weight_dict = dict(zip(classes, self.class_weights))
        
        print("Class weights calculated:")
        status_names = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical']
        for i, (class_idx, weight) in enumerate(weight_dict.items()):
            print(f"  {status_names[i]} (Class {class_idx}): {weight:.3f}")
        
        return weight_dict
    
    def adaptive_search_space(self, X_train):
        """
        FIXED: Adaptive hyperparameter search space with proper range validation
        
        Theory: Search space should adapt to dataset size, feature count, and complexity
        to ensure efficient optimization and prevent overfitting.
        """
        print("\n🎯 Creating Adaptive Search Space...")
        
        n_samples, n_features = X_train.shape
        
        # Adjust max_depth based on feature count and sample size
        if n_samples < 100:
            max_depth_max = min(6, max(3, n_features // 5))
        elif n_samples < 1000:
            max_depth_max = min(10, max(5, n_features // 3))
        else:
            max_depth_max = min(15, max(7, n_features // 2))
        
        # Adjust num_leaves based on feature count
        if n_samples < 100:
            max_num_leaves = min(63, max(15, n_features))
        else:
            max_num_leaves = min(255, max(31, n_features * 2))
        
        # Adjust learning rate range based on dataset size
        if n_samples > 100000:
            lr_min, lr_max = 0.01, 0.05
        elif n_samples > 10000:
            lr_min, lr_max = 0.01, 0.1
        elif n_samples > 1000:
            lr_min, lr_max = 0.05, 0.15
        else:
            lr_min, lr_max = 0.1, 0.3  # Higher learning rate for small datasets
        
        # FIXED: Adjust regularization based on sample-to-feature ratio with proper range validation
        sample_feature_ratio = n_samples / n_features
        if sample_feature_ratio < 5:
            reg_min, reg_max = 0.5, 2.0  # Strong regularization
            min_data_leaf_min = max(1, n_samples // 20)  # More conservative minimum
        elif sample_feature_ratio < 10:
            reg_min, reg_max = 0.1, 1.0  # Moderate regularization
            min_data_leaf_min = max(1, n_samples // 30)  # More conservative minimum
        else:
            reg_min, reg_max = 0.0, 0.5  # Light regularization
            min_data_leaf_min = max(1, n_samples // 50)
        
        # FIXED: Ensure min_data_in_leaf range is valid
        min_data_leaf_max = max(min_data_leaf_min + 1, min(200, n_samples // 3))  # Ensure max > min
        
        # Additional validation to prevent invalid ranges
        if min_data_leaf_min >= min_data_leaf_max:
            min_data_leaf_min = 1
            min_data_leaf_max = max(2, min(10, n_samples // 2))
        
        search_space = {
            'num_leaves': hp.quniform('num_leaves', 15, max_num_leaves, 2),
            'max_depth': hp.quniform('max_depth', 3, max_depth_max, 1),
            'learning_rate': hp.uniform('learning_rate', lr_min, lr_max),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.9),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.9),
            'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', min_data_leaf_min, min_data_leaf_max, 1),
            'lambda_l1': hp.uniform('lambda_l1', reg_min, reg_max),
            'lambda_l2': hp.uniform('lambda_l2', reg_min, reg_max)
        }
        
        print(f"📊 Adaptive parameters:")
        print(f"  Max depth range: 3-{max_depth_max}")
        print(f"  Num leaves range: 15-{max_num_leaves}")
        print(f"  Learning rate range: {lr_min}-{lr_max}")
        print(f"  Regularization range: {reg_min}-{reg_max}")
        print(f"  Min data in leaf: {min_data_leaf_min}-{min_data_leaf_max}")
        
        return search_space
    
    def nested_cross_validation(self, X, y, outer_splits=5, inner_splits=3, use_optimized_params=True):
        """
        FIXED: Nested cross-validation with proper validation dataset handling
        
        Theory: Nested CV provides an unbiased estimate of model performance by
        separating hyperparameter optimization (inner loop) from performance
        estimation (outer loop).
        """
        print(f"\n🔄 Nested Cross-Validation ({outer_splits} outer, {inner_splits} inner splits)...")
        
        # Adjust splits for small datasets
        n_samples = len(X)
        if n_samples < 50:
            outer_splits = min(3, n_samples // 5)
            inner_splits = min(2, n_samples // 10)
            print(f"⚠️  Small dataset detected. Adjusted to {outer_splits} outer, {inner_splits} inner splits")
        
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=self.random_state)
        outer_scores = []
        fold_predictions = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"  Processing outer fold {fold + 1}/{outer_splits}...")
            
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
            
            # FIXED: Create validation split for early stopping
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
                X_train_outer, y_train_outer, test_size=0.2, random_state=self.random_state, stratify=y_train_outer
            )
            
            # Inner loop for hyperparameter optimization
            if use_optimized_params and len(X_train_inner) > inner_splits * 5:  # Minimum samples per fold
                try:
                    best_params = self.optimize_hyperparameters(
                        X_train_inner, y_train_inner, 
                        n_trials=min(50, len(X_train_inner) // 2)  # Adaptive trials
                    )
                except Exception as e:
                    print(f"    Optimization failed, using default params: {e}")
                    best_params = self.get_enhanced_default_params(X_train_inner)
            else:
                print(f"    Using default params due to small sample size")
                best_params = self.get_enhanced_default_params(X_train_inner)
            
            # FIXED: Train model on inner training set with proper validation
            train_data = lgb.Dataset(X_train_inner, label=y_train_inner,
                                categorical_feature=self.categorical_features)
            val_data = lgb.Dataset(X_val_inner, label=y_val_inner,
                                categorical_feature=self.categorical_features,
                                reference=train_data)
            
            # FIXED: Train with validation dataset for early stopping
            model = lgb.train(
                best_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate on outer test set
            y_pred_proba = model.predict(X_test_outer, num_iteration=model.best_iteration)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            fold_score = f1_score(y_test_outer, y_pred, average='macro')
            outer_scores.append(fold_score)
            
            fold_predictions.append({
                'fold': fold,
                'y_true': y_test_outer.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
        
        nested_mean = np.mean(outer_scores)
        nested_std = np.std(outer_scores)
        
        print(f"🎯 Nested CV Results:")
        print(f"  Mean F1-macro: {nested_mean:.4f} ± {nested_std:.4f}")
        print(f"  Individual fold scores: {[f'{score:.3f}' for score in outer_scores]}")
        
        self.nested_cv_results = {
            'mean_score': nested_mean,
            'std_score': nested_std,
            'fold_scores': outer_scores,
            'fold_predictions': fold_predictions
        }
        
        return nested_mean, nested_std, fold_predictions
    
    def get_enhanced_default_params(self, X_train):
        """
        FIXED: Enhanced default parameters with proper validation and range checking
        """
        n_samples, n_features = X_train.shape
        sample_feature_ratio = n_samples / n_features
        
        # FIXED: Ensure min_data_in_leaf is reasonable for dataset size
        min_data_leaf = max(1, min(20, n_samples // 20))
        
        if sample_feature_ratio < 5:
            # Strong regularization for small datasets
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': min(31, max(15, n_features)),
                'max_depth': min(6, max(3, n_features // 5)),
                'learning_rate': 0.15,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 3,
                'min_data_in_leaf': min_data_leaf,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'is_unbalance': True,
                'verbosity': -1,
                'random_state': self.random_state
            }
        else:
            # Standard parameters
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'max_depth': 7,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': min_data_leaf,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'is_unbalance': True,
                'verbosity': -1,
                'random_state': self.random_state
            }
        
        return params
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=100):
        """
        FIXED: Bayesian optimization with proper validation dataset handling
        """
        print(f"\n🎯 Enhanced Hyperparameter Optimization ({n_trials} trials)...")
        
        def objective(params):
            """Enhanced objective function with proper validation"""
            # Convert hyperopt parameters to LightGBM format
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': int(params['num_leaves']),
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'feature_fraction': params['feature_fraction'],
                'bagging_fraction': params['bagging_fraction'],
                'bagging_freq': int(params['bagging_freq']),
                'min_data_in_leaf': int(params['min_data_in_leaf']),
                'lambda_l1': params['lambda_l1'],
                'lambda_l2': params['lambda_l2'],
                'is_unbalance': True,
                'verbosity': -1,
                'random_state': self.random_state
            }
            
            # Enhanced stratified K-Fold cross-validation
            n_samples = len(X_train)
            n_splits = min(5, max(2, n_samples // 10))  # Adaptive splits
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            
            try:
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # FIXED: Create LightGBM datasets with proper validation
                    train_data = lgb.Dataset(X_fold_train, label=y_fold_train, 
                                        categorical_feature=self.categorical_features)
                    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, 
                                        categorical_feature=self.categorical_features,
                                        reference=train_data)
                    
                    # Train model with adaptive early stopping
                    early_stopping_rounds = min(50, max(10, n_samples // 20))
                    model = lgb.train(
                        lgb_params, 
                        train_data, 
                        valid_sets=[train_data, val_data],
                        valid_names=['train', 'valid'],
                        num_boost_round=500,  # Reduced for faster optimization
                        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
                    )
                    
                    # Predict and calculate F1-macro score
                    y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
                    y_pred_class = np.argmax(y_pred, axis=1)
                    f1_macro = f1_score(y_fold_val, y_pred_class, average='macro')
                    scores.append(f1_macro)
                
                return {'loss': -np.mean(scores), 'status': STATUS_OK}
            
            except Exception as e:
                # Return penalty for failed configurations
                print(f"  Trial failed: {str(e)[:50]}...")
                return {'loss': 1.0, 'status': STATUS_OK}
        
        # Use adaptive search space with validation
        try:
            search_space = self.adaptive_search_space(X_train)
        except Exception as e:
            print(f"⚠️  Search space creation failed: {e}")
            # Fallback to simple search space
            search_space = {
                'num_leaves': hp.quniform('num_leaves', 15, 63, 2),
                'max_depth': hp.quniform('max_depth', 3, 8, 1),
                'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
                'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.9),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.9),
                'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 20, 1),
                'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
                'lambda_l2': hp.uniform('lambda_l2', 0.0, 1.0)
            }
        
        # Run optimization with enhanced early stopping
        trials = Trials()
        patience = min(20, n_trials // 5)  # Adaptive patience
        
        try:
            best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                    max_evals=n_trials, trials=trials,
                    early_stop_fn=no_progress_loss(patience))
            
            # Convert best parameters back to LightGBM format
            self.best_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': int(best['num_leaves']),
                'max_depth': int(best['max_depth']),
                'learning_rate': best['learning_rate'],
                'feature_fraction': best['feature_fraction'],
                'bagging_fraction': best['bagging_fraction'],
                'bagging_freq': int(best['bagging_freq']),
                'min_data_in_leaf': int(best['min_data_in_leaf']),
                'lambda_l1': best['lambda_l1'],
                'lambda_l2': best['lambda_l2'],
                'is_unbalance': True,
                'verbosity': -1,
                'random_state': self.random_state
            }
            
            best_score = -trials.best_trial['result']['loss']
            print(f"✅ Enhanced optimization completed. Best F1-macro score: {best_score:.4f}")
            
            return self.best_params
            
        except Exception as e:
            print(f"⚠️  Optimization failed completely: {e}")
            return self.get_enhanced_default_params(X_train)
    
    def train_model(self, X_train, y_train, X_val, y_val, use_optimized_params=True):
        """
        Train LightGBM model with optimized parameters and early stopping
        
        Theory: Early stopping prevents overfitting by monitoring validation
        performance and stopping when no improvement is observed.
        """
        print("\n🚀 Training LightGBM Model...")
        
        # Use optimized parameters if available, otherwise use enhanced defaults
        if use_optimized_params and hasattr(self, 'best_params'):
            params = self.best_params.copy()
            print("Using optimized hyperparameters")
        else:
            params = self.get_enhanced_default_params(X_train)
            print("Using enhanced default hyperparameters")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, 
                               categorical_feature=self.categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, 
                             categorical_feature=self.categorical_features,
                             reference=train_data)
        
        # Adaptive early stopping based on dataset size
        n_samples = len(X_train)
        early_stopping_rounds = min(100, max(20, n_samples // 10))
        num_boost_round = min(5000, max(500, n_samples * 2))
        
        # Train model with early stopping
        self.model = lgb.train(
            params, 
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"✅ Training completed. Best iteration: {self.model.best_iteration}")
        
        return self.model
    
    def create_ensemble_model(self, X_train, y_train):
        """
        NEW: Create ensemble model using voting classifier for diversity
        
        Theory: Ensemble methods reduce overfitting through model diversity
        and typically provide more robust predictions than single models.
        """
        print("\n🎭 Creating Ensemble Model...")
        
        # Create diverse base estimators
        estimators = []
        
        # LightGBM with different configurations
        lgb_conservative = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
            random_state=self.random_state, class_weight='balanced'
        )
        estimators.append(('lgb_conservative', lgb_conservative))
        
        lgb_aggressive = lgb.LGBMClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            num_leaves=127, feature_fraction=0.7, bagging_fraction=0.7,
            random_state=self.random_state + 1, class_weight='balanced'
        )
        estimators.append(('lgb_aggressive', lgb_aggressive))
        
        # Random Forest for diversity
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=self.random_state + 2,
            class_weight='balanced'
        )
        estimators.append(('rf', rf))
        
        # Extra Trees for additional diversity
        et = ExtraTreesClassifier(
            n_estimators=100, max_depth=12, min_samples_split=3,
            min_samples_leaf=1, random_state=self.random_state + 3,
            class_weight='balanced'
        )
        estimators.append(('et', et))
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use predicted probabilities
            n_jobs=-1
        )
        
        print("🔧 Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        
        print(f"✅ Ensemble model trained with {len(estimators)} base estimators")
        
        return self.ensemble_model
    
    def augment_data(self, X, y, augmentation_factor=3):
        """
        NEW: Data augmentation using bootstrap resampling with noise
        
        Theory: Data augmentation increases effective dataset size and helps
        models generalize better, especially important for small datasets.
        """
        print(f"\n📈 Data Augmentation (factor: {augmentation_factor})...")
        
        original_size = len(X)
        X_augmented = []
        y_augmented = []
        
        # Keep original data
        X_augmented.append(X.values)
        y_augmented.append(y.values)
        
        # Generate augmented samples
        for i in range(augmentation_factor - 1):
            # Bootstrap resampling
            X_resampled, y_resampled = resample(
                X, y, replace=True, 
                random_state=self.random_state + i
            )
            
            # Add small amount of noise to numerical features
            X_resampled_values = X_resampled.values.copy()
            numerical_mask = X.dtypes != 'object'
            
            for j, is_numerical in enumerate(numerical_mask):
                if is_numerical:
                    # Add Gaussian noise (1% of std)
                    std = X.iloc[:, j].std()
                    noise = np.random.normal(0, std * 0.01, len(X_resampled_values))
                    X_resampled_values[:, j] += noise
            
            X_augmented.append(X_resampled_values)
            y_augmented.append(y_resampled.values)
        
        # Combine all augmented data
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        # Convert back to DataFrame/Series
        X_augmented_df = pd.DataFrame(X_final, columns=X.columns)
        y_augmented_series = pd.Series(y_final)
        
        print(f"✅ Data augmented from {original_size} to {len(X_augmented_df)} samples")
        
        return X_augmented_df, y_augmented_series
    
    def comprehensive_evaluation(self, X_test, y_test, class_names=None):
        """
        Comprehensive model evaluation with multiple metrics optimized for
        imbalanced multi-class classification in cable health assessment.
        """
        print("\n📊 Comprehensive Model Evaluation...")
        
        if class_names is None:
            class_names = ['Healthy', 'At Risk', 'Critical']
        
        # Generate predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Multi-class AUC (One-vs-Rest)
        try:
            auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc_ovr = np.nan
        
        # Matthews Correlation Coefficient (balanced measure for multi-class)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        print("🎯 Overall Performance Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Macro): {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"  AUC (OvR): {auc_ovr:.4f}")
        print(f"  Matthews Correlation: {mcc:.4f}")
        
        print("\n📈 Per-Class Performance:")
        for i, class_name in enumerate(class_names):
            emoji = ['🔵', '🟠', '🔴'][i]
            print(f"  {emoji} {class_name}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1_per_class[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_ovr': auc_ovr,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'per_class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1_per_class,
                'support': support
            }
        }
    
    def evaluate_ensemble(self, X_test, y_test, class_names=None):
        """
        NEW: Evaluate ensemble model performance
        """
        print("\n📊 Ensemble Model Evaluation...")
        
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
        
        print("🎭 Ensemble Performance Metrics:")
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
    
    def feature_importance_analysis(self):
        """
        Comprehensive feature importance analysis using multiple methods
        
        Theory: Feature importance helps identify key cable health indicators
        for maintenance decision-making and model interpretability.
        """
        print("\n🔍 Feature Importance Analysis...")
        
        # LightGBM built-in feature importance
        importance_split = self.model.feature_importance(importance_type='split')
        importance_gain = self.model.feature_importance(importance_type='gain')
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_split': importance_split,
            'importance_gain': importance_gain
        })
        
        # Normalize importances for better comparison
        feature_importance_df['importance_split_norm'] = (
            feature_importance_df['importance_split'] / 
            feature_importance_df['importance_split'].sum()
        )
        feature_importance_df['importance_gain_norm'] = (
            feature_importance_df['importance_gain'] / 
            feature_importance_df['importance_gain'].sum()
        )
        
        # Sort by gain importance (more meaningful for prediction)
        feature_importance_df = feature_importance_df.sort_values(
            'importance_gain', ascending=False
        )
        
        print("🏆 Top 15 Most Important Features (by Gain):")
        for i, row in feature_importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance_gain_norm']:.3f}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(top_features)), top_features['importance_gain_norm'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Normalized Importance (Gain)')
        plt.title('Feature Importance by Gain')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        plt.barh(range(len(top_features)), top_features['importance_split_norm'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Normalized Importance (Split)')
        plt.title('Feature Importance by Split Count')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def decision_matrix_analysis(self, y_test, y_pred, class_names=None):
        """
        Create comprehensive decision matrix for cable health assessment
        
        Theory: Decision matrix (confusion matrix) provides detailed insights
        into model performance across different classes, crucial for understanding
        classification errors in maintenance planning.
        """
        print("\n🎯 Decision Matrix Analysis...")
        
        if class_names is None:
            class_names = ['Healthy', 'At Risk', 'Critical']
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Print detailed analysis
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
        
        print("\n📈 Confusion Matrix (Percentages):")
        print("        Predicted")
        print("Actual   ", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        
        for i, actual_name in enumerate(class_names):
            print(f"{actual_name:>8} ", end="")
            for j in range(len(class_names)):
                print(f"{cm_percent[i,j]:>9.1f}%", end="")
            print()
        
        # Calculate critical metrics for cable health assessment
        print("\n🚨 Critical Error Analysis:")
        
        # False negatives for Critical class (most dangerous)
        critical_false_negatives = cm[2, 0] + cm[2, 1]  # Critical classified as Healthy/At Risk
        critical_total = cm[2, :].sum()
        critical_miss_rate = critical_false_negatives / critical_total if critical_total > 0 else 0
        
        print(f"  Critical cables missed: {critical_false_negatives}/{critical_total} ({critical_miss_rate:.1%})")
        
        # False positives for Critical class (unnecessary maintenance)
        critical_false_positives = cm[0, 2] + cm[1, 2]  # Healthy/At Risk classified as Critical
        
        print(f"  Unnecessary critical maintenance: {critical_false_positives}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Cable Health Prediction')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.show()
        
        # Create percentage heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Percentages')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.show()
        
        return cm, cm_percent
    
    def enhanced_shap_analysis(self, X_test, max_samples=500):
        """
        ENHANCED: SHAP analysis with statistical significance testing
        
        Theory: SHAP provides game-theoretic explanations for individual predictions
        by computing the contribution of each feature to the final prediction.
        Enhanced with bootstrap confidence intervals for reliability.
        """
        print(f"\n🔬 Enhanced SHAP Interpretability Analysis...")
        
        # Adaptive sample size based on dataset size
        n_samples = len(X_test)
        if n_samples < 50:
            sample_size = n_samples
            print("⚠️  Small dataset: using all samples for SHAP analysis")
        else:
            sample_size = min(max_samples, n_samples)
        
        # Sample data for analysis
        if sample_size < len(X_test):
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test
        
        print(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values with error handling
        try:
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Bootstrap confidence intervals for SHAP values
            if len(X_sample) >= 20:  # Only if we have enough samples
                print("Computing bootstrap confidence intervals...")
                bootstrap_shap = []
                n_bootstrap = min(10, len(X_sample) // 5)
                
                for i in range(n_bootstrap):
                    bootstrap_indices = np.random.choice(len(X_sample), 
                                                       min(len(X_sample), 50), 
                                                       replace=True)
                    X_bootstrap = X_sample.iloc[bootstrap_indices]
                    shap_bootstrap = self.shap_explainer.shap_values(X_bootstrap)
                    bootstrap_shap.append(shap_bootstrap)
                
                # Calculate confidence intervals
                bootstrap_mean = np.mean(bootstrap_shap, axis=0)
                bootstrap_std = np.std(bootstrap_shap, axis=0)
                
                print("✅ SHAP values computed with confidence intervals")
            else:
                print("✅ SHAP values computed (insufficient samples for confidence intervals)")
            
        except Exception as e:
            print(f"⚠️  SHAP computation failed: {e}")
            print("Falling back to feature importance analysis...")
            return None
        
        # Enhanced visualizations
        try:
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                             class_names=['Healthy', 'At Risk', 'Critical'], show=False)
            plt.title('SHAP Summary Plot - Feature Impact on Cable Health Prediction')
            plt.tight_layout()
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                             plot_type="bar", class_names=['Healthy', 'At Risk', 'Critical'], show=False)
            plt.title('SHAP Feature Importance - Cable Health Prediction')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️  SHAP visualization failed: {e}")
        
        return shap_values
    
    def analyze_feature_subsets(self, X, y):
        """
        NEW: Comprehensive analysis of different feature subsets with proper unpacking
        
        Analyzes performance with:
        1. All features
        2. Aggressive feature selection (top k)
        3. Only engineered features
        4. Only highest gain score features
        """
        print("\n🔬 Comprehensive Feature Subset Analysis...")
        
        results = {}
        
        # Helper function to safely handle nested CV results
        def safe_nested_cv(X_subset, y_subset, subset_name):
            """Safely handle nested CV with proper unpacking and fallbacks"""
            try:
                # FIXED: Properly unpack all 3 return values
                cv_result = self.nested_cross_validation(X_subset, y_subset, outer_splits=3, inner_splits=2)
                
                # Handle different return formats
                if isinstance(cv_result, tuple):
                    if len(cv_result) == 3:
                        mean_score, std_score, fold_predictions = cv_result
                    elif len(cv_result) == 2:
                        mean_score, std_score = cv_result
                        fold_predictions = None
                    else:
                        print(f"⚠️  Unexpected return format for {subset_name}: {len(cv_result)} values")
                        mean_score, std_score = cv_result[0], cv_result[1] if len(cv_result) > 1 else 0.0
                        fold_predictions = None
                else:
                    # Single value returned
                    mean_score = cv_result
                    std_score = 0.0
                    fold_predictions = None
                
                return mean_score, std_score, fold_predictions
                
            except Exception as e:
                print(f"⚠️  Nested CV failed for {subset_name}: {e}")
                print(f"🔧 Falling back to simple train-test split...")
                
                # Fallback: Simple train-test split evaluation
                try:
                    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                        X_subset, y_subset, test_size=0.3, random_state=self.random_state, stratify=y_subset
                    )
                    
                    # Use default parameters for quick evaluation
                    params = self.get_enhanced_default_params(X_train_sub)
                    
                    # Create datasets
                    train_data = lgb.Dataset(X_train_sub, label=y_train_sub,
                                        categorical_feature=self.categorical_features)
                    
                    # Train model
                    model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=0)
                    
                    # Evaluate
                    y_pred_proba = model.predict(X_test_sub)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    fallback_score = f1_score(y_test_sub, y_pred, average='macro')
                    
                    print(f"✅ Fallback evaluation completed for {subset_name}: {fallback_score:.4f}")
                    return fallback_score, 0.0, None
                    
                except Exception as e2:
                    print(f"❌ Fallback evaluation also failed for {subset_name}: {e2}")
                    return 0.0, 0.0, None
        
        # 1. All features baseline
        print("\n1️⃣ Baseline - All Features:")
        baseline_score, baseline_std, baseline_predictions = safe_nested_cv(X, y, "all_features")
        results['all_features'] = {'mean': baseline_score, 'std': baseline_std}
        print(f"  F1-Macro: {baseline_score:.4f} ± {baseline_std:.4f}")
        
        # 2. Aggressive feature selection using mutual information
        print("\n2️⃣ Aggressive Feature Selection (Mutual Information):")
        try:
            n_samples = len(X)
            if n_samples < 100:
                k_aggressive = min(8, X.shape[1] // 3)
            else:
                k_aggressive = min(15, X.shape[1] // 2)
            
            X_selected, selected_features, mi_df = self.mutual_info_feature_selection(X, y, k=k_aggressive)
            selected_score, selected_std, selected_predictions = safe_nested_cv(X_selected, y, "aggressive_selection")
            results['aggressive_selection'] = {
                'mean': selected_score, 
                'std': selected_std,
                'features': selected_features,
                'n_features': len(selected_features)
            }
            print(f"  F1-Macro: {selected_score:.4f} ± {selected_std:.4f}")
            print(f"  Features used: {len(selected_features)}")
            
        except Exception as e:
            print(f"⚠️  Aggressive selection failed: {e}")
            results['aggressive_selection'] = None
        
        # 3. Only engineered features
        print("\n3️⃣ Engineered Features Only:")
        engineered_features = [
            'RemainingLifeRatio', 'AgeUtilizationRatio', 'CompoundFailureRate',
            'ElectricalStressScore', 'MaintenanceResponsiveness', 
            'EnvironmentalRiskScore', 'CriticalLoadIndicator', 'RecentAnomalyCluster'
        ]
        
        available_engineered = [f for f in engineered_features if f in X.columns]
        if available_engineered:
            try:
                X_engineered = X[available_engineered]
                engineered_score, engineered_std, engineered_predictions = safe_nested_cv(X_engineered, y, "engineered_only")
                results['engineered_only'] = {
                    'mean': engineered_score, 
                    'std': engineered_std,
                    'features': available_engineered,
                    'n_features': len(available_engineered)
                }
                print(f"  F1-Macro: {engineered_score:.4f} ± {engineered_std:.4f}")
                print(f"  Features used: {len(available_engineered)}")
                
            except Exception as e:
                print(f"⚠️  Engineered features evaluation failed: {e}")
                results['engineered_only'] = None
        else:
            print("⚠️  No engineered features found in dataset")
            results['engineered_only'] = None
        
        # 4. Highest gain score features
        print("\n4️⃣ Highest Gain Score Features:")
        try:
            # Train a quick model to get feature importance
            X_train_quick, X_test_quick, y_train_quick, y_test_quick = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            params_quick = self.get_enhanced_default_params(X_train_quick)
            train_data = lgb.Dataset(X_train_quick, label=y_train_quick,
                                categorical_feature=self.categorical_features)
            
            model_quick = lgb.train(params_quick, train_data, num_boost_round=100, verbose_eval=0)
            
            # Get feature importance
            importance_gain = model_quick.feature_importance(importance_type='gain')
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_gain': importance_gain
            }).sort_values('importance_gain', ascending=False)
            
            # Select top gain features
            top_k_gain = min(10, len(feature_importance_df) // 2)
            top_gain_features = feature_importance_df.head(top_k_gain)['feature'].tolist()
            
            X_top_gain = X[top_gain_features]
            gain_score, gain_std, gain_predictions = safe_nested_cv(X_top_gain, y, "top_gain")
            results['top_gain'] = {
                'mean': gain_score, 
                'std': gain_std,
                'features': top_gain_features,
                'n_features': len(top_gain_features)
            }
            print(f"  F1-Macro: {gain_score:.4f} ± {gain_std:.4f}")
            print(f"  Features used: {len(top_gain_features)}")
            
        except Exception as e:
            print(f"⚠️  Top gain analysis failed: {e}")
            results['top_gain'] = None
        
        # Print comprehensive results
        print("\n📊 Feature Subset Analysis Results:")
        print("=" * 60)
        
        subset_display = {
            'all_features': 'All Features',
            'aggressive_selection': 'Aggressive Selection (MI)',
            'engineered_only': 'Engineered Features Only',
            'top_gain': 'Top Gain Score Features'
        }
        
        for subset_name, result in results.items():
            if result is None:
                continue
                
            print(f"\n{subset_display[subset_name]}:")
            print(f"  F1-Macro: {result['mean']:.4f} ± {result['std']:.4f}")
            
            if 'n_features' in result:
                print(f"  Features used: {result['n_features']}")
                if baseline_score > 0:  # Avoid division by zero
                    improvement = ((result['mean'] - baseline_score) / baseline_score) * 100
                    print(f"  Improvement: {improvement:+.1f}% vs baseline")
        
        # Determine best approach
        valid_results = {k: v for k, v in results.items() if v is not None and v['mean'] > 0}
        if valid_results:
            best_approach = max(valid_results.keys(), key=lambda x: valid_results[x]['mean'])
            print(f"\n🏆 Best Approach: {subset_display[best_approach]}")
            print(f"  Score: {valid_results[best_approach]['mean']:.4f} ± {valid_results[best_approach]['std']:.4f}")
        
        return results
    
    def generate_prediction_report(self, X_test, y_test, cable_ids=None):
        """
        Generate comprehensive prediction report for cable health assessment
        """
        print("\n📋 Generating Prediction Report...")
        
        # Generate predictions with probabilities
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Create report DataFrame
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
        
        # Identify high-risk misclassifications
        critical_misses = report_df[
            (report_df['Actual_Status'] == 2) & (report_df['Predicted_Status'] != 2)
        ]
        
        if len(critical_misses) > 0:
            print(f"⚠️  {len(critical_misses)} critical cables misclassified:")
            print(critical_misses[['CableID', 'Predicted_Label', 'Max_Confidence']].to_string(index=False))
        else:
            print("✅ All critical cables correctly identified")
        
        return report_df

# Enhanced main execution pipeline
def enhanced_main():
    """
    Enhanced main execution pipeline for cable health prediction system
    """
    print("🚀 Starting Enhanced Cable Health Prediction Pipeline...")
    
    # Initialize enhanced predictor
    predictor = EnhancedCableHealthPredictor(random_state=42)
    
    # Step 1: Load and explore data
    df = predictor.load_and_explore_data('cable_health_method2_clustering_20k.csv')    #Pointer 
    
    # Step 2: Enhanced correlation analysis
    corr_matrix, high_corr_pairs = predictor.correlation_analysis(df)
    
    # Step 3: Advanced feature engineering
    df_enhanced, new_features = predictor.advanced_feature_engineering(df)
    
    # Step 4: Prepare features
    X, y = predictor.prepare_features(df_enhanced)
    
    # Step 5: Handle class imbalance
    class_weights = predictor.handle_class_imbalance(y)
    
    # Step 6: Comprehensive feature subset analysis
    subset_results = predictor.analyze_feature_subsets(X, y)
    
    # Step 7: Data augmentation (if dataset is small)
    if len(X) < 1000:
        print(f"\n📈 Applying data augmentation for small dataset...")
        X_augmented, y_augmented = predictor.augment_data(X, y, augmentation_factor=3)
        print(f"Dataset size increased from {len(X)} to {len(X_augmented)} samples")
        X, y = X_augmented, y_augmented
    
    # Step 8: Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training set for validation
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Enhanced Data Split Summary:")
    print(f"  Training set: {len(X_train_opt):,} samples")
    print(f"  Validation set: {len(X_val):,} samples") 
    print(f"  Test set: {len(X_test):,} samples")
    
    # Step 9: Nested Cross-Validation for reliable performance estimation
    print(f"\n🔄 Performing Nested Cross-Validation...")
    # FIXED: Properly handle all return values
    try:
        cv_result = predictor.nested_cross_validation(X_train, y_train, outer_splits=5, inner_splits=3)
        if isinstance(cv_result, tuple) and len(cv_result) >= 2:
            nested_mean, nested_std = cv_result[0], cv_result[1]
            fold_predictions = cv_result[2] if len(cv_result) > 2 else None
        else:
            nested_mean, nested_std = cv_result, 0.0
            fold_predictions = None
    except Exception as e:
        print(f"⚠️  Nested CV failed: {e}")
        nested_mean, nested_std = 0.0, 0.0
        fold_predictions = None
    
    # Step 10: Enhanced hyperparameter optimization
    try:
        optimal_params = predictor.optimize_hyperparameters(X_train_opt, y_train_opt, n_trials=50)
    except Exception as e:
        print(f"⚠️  Hyperparameter optimization failed: {e}")
        optimal_params = None
    
    # Step 11: Train main model
    model = predictor.train_model(X_train, y_train, X_val, y_val, use_optimized_params=(optimal_params is not None))
    
    # Step 12: Create and train ensemble model
    try:
        ensemble_model = predictor.create_ensemble_model(X_train, y_train)
    except Exception as e:
        print(f"⚠️  Ensemble model creation failed: {e}")
        ensemble_model = None
    
    # Step 13: Comprehensive evaluation
    eval_results = predictor.comprehensive_evaluation(X_test, y_test)
    
    # Step 14: Ensemble evaluation
    if ensemble_model is not None:
        ensemble_results = predictor.evaluate_ensemble(X_test, y_test)
    else:
        ensemble_results = None
    
    # Step 15: Feature importance analysis
    feature_importance_df = predictor.feature_importance_analysis()
    
    # Step 16: Decision matrix analysis
    cm, cm_percent = predictor.decision_matrix_analysis(y_test, eval_results['y_pred'])
    
    # Step 17: Enhanced SHAP interpretability analysis
    try:
        shap_values = predictor.enhanced_shap_analysis(X_test, max_samples=min(500, len(X_test)))
    except Exception as e:
        print(f"⚠️  SHAP analysis failed: {e}")
        shap_values = None
    
    # Step 18: Generate prediction report
    report_df = predictor.generate_prediction_report(X_test, y_test)
    
    print("\n🎉 Enhanced Cable Health Prediction Pipeline Completed Successfully!")
    
    # Final comprehensive results
    print("\n📊 Final Enhanced Performance Summary:")
    print("=" * 60)
    print(f"🔄 Nested CV Performance: {nested_mean:.4f} ± {nested_std:.4f}")
    print(f"🎯 Single Model Accuracy: {eval_results['accuracy']:.4f}")
    print(f"🏆 Single Model F1-Macro: {eval_results['f1_macro']:.4f}")
    print(f"⚖️  Single Model MCC: {eval_results['mcc']:.4f}")
    
    if ensemble_results:
        print(f"🎭 Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"🎭 Ensemble F1-Macro: {ensemble_results['f1_macro']:.4f}")
        print(f"🎭 Ensemble MCC: {ensemble_results['mcc']:.4f}")
        
        # Compare single model vs ensemble
        improvement = ((ensemble_results['f1_macro'] - eval_results['f1_macro']) / eval_results['f1_macro']) * 100
        print(f"📈 Ensemble Improvement: {improvement:+.1f}%")
    
    print("\n🎯 Key Insights:")
    print(f"  • Most reliable estimate (Nested CV): {nested_mean:.4f}")
    print(f"  • Feature subset analysis completed")
    print(f"  • Data augmentation applied: {len(X) > len(df)}")
    print(f"  • Ensemble method trained: {ensemble_results is not None}")
    
    return predictor, eval_results, ensemble_results, feature_importance_df, report_df, subset_results

if __name__ == "__main__":
    # Execute the enhanced pipeline
    (predictor, results, ensemble_results, feature_importance, 
     prediction_report, subset_analysis) = enhanced_main()
