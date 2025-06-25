"""
Cable Health Prediction System using XGBoost
============================================
Advanced predictive analytics for cable health assessment with 3-class classification:
- 0: Healthy (Good condition, normal risk)
- 1: At Risk (Deterioration signs, preventive maintenance needed)  
- 2: Critical (Poor condition, immediate action required)

Algorithm Choice: XGBoost selected for comparative analysis against LightGBM
with focus on interpretability and robust performance on imbalanced datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_recall_fscore_support,
                           roc_auc_score, matthews_corrcoef, log_loss)

# Advanced Analytics
from sklearn.utils.class_weight import compute_class_weight
import shap
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🔧 Cable Health Prediction System - XGBoost Implementation")
print("📊 Algorithm: XGBoost (Extreme Gradient Boosting)")
print("🎯 Task: Multi-class Classification with Enhanced Interpretability")

class CableHealthPredictorXGBoost:
    """
    Advanced Cable Health Prediction System using XGBoost
    
    This implementation provides comprehensive machine learning pipeline optimized
    for XGBoost with enhanced interpretability features for cable maintenance
    decision-making and direct performance comparison with LightGBM.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.class_weights = None
        self.preprocessor = None
        self.shap_explainer = None
        self.feature_importance_df = None
        
        # Define categorical features for proper encoding
        self.categorical_columns = [
            'MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory',
            'FloodZoneRisk'
        ]
        
        print(f"✅ XGBoost CableHealthPredictor initialized with random_state={random_state}")
    
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
        
        if imbalance_ratio > 3:
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
        print("\n🔍 Correlation Analysis for XGBoost Optimization...")
        
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
        
        df_enhanced['WarrantyUtilizationRatio'] = (
            df_enhanced['AssetAgeYears'] / 
            (df_enhanced['AssetAgeYears'] + df_enhanced['YearsRemainingWarranty'])
        )
        
        # 3. Exponential decay features for time-based degradation
        df_enhanced['RepairRecencyDecay'] = np.exp(-df_enhanced['LastRepairAgeYears'] / 5.0)
        df_enhanced['InspectionRecencyDecay'] = np.exp(-df_enhanced['TimeSinceLastInspectionDays'] / 365.0)
        
        # 4. Risk aggregation features
        df_enhanced['ElectricalRiskScore'] = (
            0.4 * df_enhanced['PartialDischargeSeverityScore'] +
            0.3 * df_enhanced['AvgVoltageDeviationPercent'] +
            0.3 * df_enhanced['OverloadEventCountPast3Years']
        ).clip(0, 10)  # Normalize to 0-10 scale
        
        df_enhanced['MaintenanceRiskScore'] = (
            0.5 * df_enhanced['NumberOfRepairs'] +
            0.3 * df_enhanced['FailureRatePerYear'] +
            0.2 * df_enhanced['TimeSinceLastInspectionDays'] / 365
        )
        
        # 5. Environmental stress indicators
        df_enhanced['EnvironmentalStress'] = (
            df_enhanced['SoilCorrosivityIndex'] * 
            df_enhanced['AvgGroundTemperatureCelsius'] / 30.0 *
            (df_enhanced['FloodZoneRisk'] + 1)  # Add 1 to avoid zero multiplication
        )
        
        # 6. Anomaly frequency features
        df_enhanced['TotalAnomalies'] = (
            df_enhanced['TemperatureAnomaliesPast1Year'] +
            df_enhanced['VibrationAnomaliesPast1Year'] +
            df_enhanced['PDThresholdBreachesPast1Year']
        )
        
        df_enhanced['AnomalyRate'] = df_enhanced['TotalAnomalies'] / 12.0  # Monthly rate
        
        # 7. Business impact features
        df_enhanced['FailureImpactScore'] = (
            df_enhanced['EstimatedDowntimeCostPerFailure'] / 1000.0 *  # Scale down
            df_enhanced['SensitiveCustomerCount'] *
            df_enhanced['CriticalityScore']
        )
        
        # 8. Categorical interaction features (for tree-based learning)
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
        
        # Handle infinite values that might result from feature engineering
        for feature in new_numerical_features:
            if feature in df_enhanced.columns:
                df_enhanced[feature] = df_enhanced[feature].replace([np.inf, -np.inf], np.nan)
                if df_enhanced[feature].isnull().sum() > 0:
                    df_enhanced[feature] = df_enhanced[feature].fillna(df_enhanced[feature].median())
        
        return df_enhanced, new_numerical_features, new_categorical_features
    
    def prepare_features_xgboost(self, df):
        """
        Prepare features specifically for XGBoost with proper categorical encoding
        
        Theory: XGBoost requires explicit categorical encoding (one-hot or label)
        unlike LightGBM which handles categoricals natively.
        """
        print("\n🔧 XGBoost-Specific Feature Preparation...")
        
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
        
        # Create preprocessing pipeline
        preprocessor_steps = []
        
        if numerical_features:
            # StandardScaler for numerical features (optional for XGBoost but can help)
            numerical_transformer = StandardScaler()
            preprocessor_steps.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            # One-hot encoding for categorical features
            categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
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
        print(f"✅ One-hot encoded categorical features")
        
        return X_processed, y
    
    def handle_class_imbalance_xgboost(self, y):
        """
        XGBoost-specific class imbalance handling with scale_pos_weight calculation
        
        Theory: XGBoost uses scale_pos_weight parameter for binary classification
        and class weights for multi-class to handle imbalanced datasets.
        """
        print("\n⚖️  XGBoost Class Imbalance Handling...")
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, class_weights))
        
        # Calculate sample weights for XGBoost
        sample_weights = np.array([self.class_weights[label] for label in y])
        
        print("Class weights calculated:")
        status_names = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical']
        for i, (class_idx, weight) in enumerate(self.class_weights.items()):
            class_count = (y == class_idx).sum()
            print(f"  {status_names[i]} (Class {class_idx}): weight={weight:.3f}, count={class_count}")
        
        return sample_weights
    
    def optimize_hyperparameters_xgboost(self, X_train, y_train, sample_weights, n_trials=100):
        """
        XGBoost-specific Bayesian optimization with focus on tree parameters
        
        Theory: XGBoost hyperparameter space differs from LightGBM, requiring
        optimization of tree-specific parameters like max_depth, subsample, colsample_bytree.
        """
        print(f"\n🎯 XGBoost Hyperparameter Optimization ({n_trials} trials)...")
        
        def objective(params):
            """Objective function for XGBoost optimization"""
            
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
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                weights_fold_train = sample_weights[train_idx]
                
                # Create and train XGBoost model
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_fold_train, y_fold_train, 
                         sample_weight=weights_fold_train,
                         eval_set=[(X_fold_val, y_fold_val)],
                         early_stopping_rounds=50,
                         verbose=False)
                
                # Predict and calculate F1-macro score
                y_pred = model.predict(X_fold_val)
                f1_macro = f1_score(y_fold_val, y_pred, average='macro')
                scores.append(f1_macro)
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        # XGBoost-specific search space
        search_space = {
            'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 800, 1000]),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': hp.uniform('reg_alpha', 0, 2),
            'reg_lambda': hp.uniform('reg_lambda', 0, 2),
            'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
            'gamma': hp.uniform('gamma', 0, 1)
        }
        
        # Run optimization
        trials = Trials()
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                   max_evals=n_trials, trials=trials,
                   early_stop_fn=no_progress_loss(20))
        
        # Convert best parameters
        self.best_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': [3, 4, 5, 6, 7, 8][best['max_depth']],
            'learning_rate': best['learning_rate'],
            'n_estimators': [100, 200, 300, 500, 800, 1000][best['n_estimators']],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'colsample_bylevel': best['colsample_bylevel'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
            'min_child_weight': [1, 3, 5, 7][best['min_child_weight']],
            'gamma': best['gamma'],
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        best_score = -trials.best_trial['result']['loss']
        print(f"✅ Optimization completed. Best F1-macro score: {best_score:.4f}")
        
        return self.best_params
    
    def train_xgboost_model(self, X_train, y_train, X_val, y_val, sample_weights, use_optimized_params=True):
        """
        Train XGBoost model with early stopping and comprehensive monitoring
        """
        print("\n🚀 Training XGBoost Model...")
        
        # Use optimized or default parameters
        if use_optimized_params and hasattr(self, 'best_params'):
            params = self.best_params.copy()
            print("Using optimized hyperparameters")
        else:
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
                'n_jobs': -1,
                'early_stopping_rounds': 100
            }
            print("Using default hyperparameters")
        
        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping and evaluation
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ['train', 'validation']
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=True,
        )
        
        print(f"✅ Training completed. Best iteration: {self.model.best_iteration}")
        print(f"✅ Best validation score: {self.model.best_score:.4f}")
        
        return self.model
    
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
    
    def feature_importance_analysis_xgboost(self):
        """
        Enhanced XGBoost feature importance analysis with multiple importance types
        """
        print("\n🔍 XGBoost Feature Importance Analysis...")
        
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
    
    def decision_matrix_analysis_xgboost(self, X_test, y_test, y_pred, class_names=None):
        """
        Enhanced decision matrix analysis with XGBoost-specific insights
        """
        print("\n🎯 XGBoost Decision Matrix Analysis...")
        
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
        
        # XGBoost-specific error analysis
        print("\n🔬 XGBoost Error Analysis:")
        
        # Calculate prediction confidence for errors
        y_pred_proba = self.model.predict_proba(X_test)
        prediction_confidence = np.max(y_pred_proba, axis=1)
        
        # Identify low-confidence predictions
        low_confidence_threshold = 0.6
        low_confidence_mask = prediction_confidence < low_confidence_threshold
        low_confidence_errors = ((y_test != y_pred) & low_confidence_mask).sum()
        
        print(f"  Low confidence predictions (< {low_confidence_threshold}): {low_confidence_mask.sum()}")
        print(f"  Low confidence errors: {low_confidence_errors}")
        
        # Critical class analysis
        critical_false_negatives = cm[2, 0] + cm[2, 1]
        critical_total = cm[2, :].sum()
        critical_miss_rate = critical_false_negatives / critical_total if critical_total > 0 else 0
        
        print(f"  Critical cables missed: {critical_false_negatives}/{critical_total} ({critical_miss_rate:.1%})")
        
        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Confusion matrix (counts)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix - Counts')
        axes[0].set_xlabel('Predicted Class')
        axes[0].set_ylabel('Actual Class')
        
        # Confusion matrix (percentages)
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix - Percentages')
        axes[1].set_xlabel('Predicted Class')
        axes[1].set_ylabel('Actual Class')
        
        # Prediction confidence distribution
        axes[2].hist(prediction_confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2].axvline(low_confidence_threshold, color='red', linestyle='--', label=f'Low Confidence Threshold ({low_confidence_threshold})')
        axes[2].set_xlabel('Prediction Confidence')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Prediction Confidence Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return cm, cm_percent, prediction_confidence
    
    def shap_interpretability_analysis_xgboost(self, X_test, max_samples=1000):
        """
        SHAP analysis optimized for XGBoost with enhanced visualizations
        """
        print(f"\n🔬 XGBoost SHAP Interpretability Analysis (sample size: {min(len(X_test), max_samples)})...")
        
        # Sample for computational efficiency
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test
        
        # Create SHAP explainer for XGBoost
        self.shap_explainer = shap.TreeExplainer(self.model)
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        print("✅ SHAP values computed")
        
        # Enhanced SHAP visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                         class_names=['Healthy', 'At Risk', 'Critical'], show=False)
        plt.title('SHAP Summary Plot - Feature Impact Distribution')
        
        # Feature importance plot
        plt.subplot(2, 2, 2)
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                         plot_type="bar", class_names=['Healthy', 'At Risk', 'Critical'], show=False)
        plt.title('SHAP Feature Importance - Mean Impact')
        
        # Waterfall plot for a sample prediction
        plt.subplot(2, 2, 3)
        if len(X_sample) > 0:
            shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                               base_values=self.shap_explainer.expected_value[1],
                                               data=X_sample.iloc[0]), show=False)
            plt.title('SHAP Waterfall Plot - Sample Prediction')
        
        plt.subplot(2, 2, 4)
        # Dependence plot for most important feature
        if hasattr(self, 'feature_importance_df') and len(self.feature_importance_df) > 0:
            most_important_feature = self.feature_importance_df.iloc[0]['feature']
            if most_important_feature in X_sample.columns:
                try:
                    # Get the correct feature index
                    feature_idx = list(X_sample.columns).index(most_important_feature)
                    
                    # Ensure feature index is within bounds
                    if feature_idx < shap_values[1].shape[1]:
                        shap.dependence_plot(feature_idx, shap_values[1], X_sample, show=False)
                        plt.title(f'SHAP Dependence Plot - {most_important_feature} (At Risk Class)')
                    else:
                        # Fallback: use first feature if index is out of bounds
                        shap.dependence_plot(0, shap_values[1], X_sample, show=False)
                        plt.title(f'SHAP Dependence Plot - Feature 0 (At Risk Class)')
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Could not create dependence plot: {e}")
                    plt.text(0.5, 0.5, 'Dependence plot unavailable', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('SHAP Dependence Plot - Error')

        plt.tight_layout()
        plt.show()
        
        return shap_values
    
    def generate_comparative_report(self, X_test, y_test, cable_ids=None):
        """
        Generate comprehensive comparative report for XGBoost vs LightGBM analysis
        """
        print("\n📋 Generating XGBoost Comparative Report...")
        
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
        
        # XGBoost-specific analysis
        report_df['Model'] = 'XGBoost'
        report_df['Best_Iteration'] = self.model.best_iteration
        
        # Confidence analysis
        confidence_stats = {
            'mean_confidence': report_df['Max_Confidence'].mean(),
            'std_confidence': report_df['Max_Confidence'].std(),
            'low_confidence_count': (report_df['Max_Confidence'] < 0.6).sum(),
            'high_confidence_accuracy': report_df[report_df['Max_Confidence'] > 0.8]['Prediction_Correct'].mean()
        }
        
        print("🎯 XGBoost Prediction Confidence Analysis:")
        print(f"  Mean confidence: {confidence_stats['mean_confidence']:.3f}")
        print(f"  Confidence std: {confidence_stats['std_confidence']:.3f}")
        print(f"  Low confidence predictions: {confidence_stats['low_confidence_count']}")
        print(f"  High confidence accuracy: {confidence_stats['high_confidence_accuracy']:.3f}")
        
        return report_df, confidence_stats

# Main execution pipeline for XGBoost
def main_xgboost():
    """
    Main execution pipeline for XGBoost cable health prediction system
    """
    print("🚀 Starting XGBoost Cable Health Prediction Pipeline...")
    
    # Initialize XGBoost predictor
    predictor = CableHealthPredictorXGBoost(random_state=42)
    
    # Step 1: Load and explore data
    df = predictor.load_and_explore_data('cable_health_sample_ordinal_encoded_20000.csv')  # Using the provided dataset
    
    # Step 2: Enhanced correlation analysis
    corr_matrix, feature_clusters, significant_features = predictor.correlation_analysis(df)
    
    # Step 3: XGBoost-optimized feature engineering
    df_enhanced, new_numerical_features, new_categorical_features = predictor.advanced_feature_engineering(df)
    
    # Step 4: XGBoost-specific feature preparation
    X, y = predictor.prepare_features_xgboost(df_enhanced)
    
    # Step 5: Handle class imbalance for XGBoost
    sample_weights = predictor.handle_class_imbalance_xgboost(y)
    
    # Step 6: Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Adjust sample weights for split
    train_indices = X_train.index
    sample_weights_train = sample_weights[train_indices]
    
    # Validation split
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    train_opt_indices = X_train_opt.index
    sample_weights_train_opt = sample_weights[train_opt_indices]
    
    print(f"\n📊 XGBoost Data Split Summary:")
    print(f"  Training set: {len(X_train_opt):,} samples")
    print(f"  Validation set: {len(X_val):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Total features: {len(predictor.feature_names)}")
    
    # Step 7: Hyperparameter optimization (optional)
    # optimal_params = predictor.optimize_hyperparameters_xgboost(X_train_opt, y_train_opt, sample_weights_train_opt, n_trials=50)
    
    # Step 8: Train XGBoost model
    model = predictor.train_xgboost_model(X_train, y_train, X_val, y_val, sample_weights_train, use_optimized_params=False)
    
    # Step 9: Comprehensive evaluation
    eval_results = predictor.comprehensive_evaluation_xgboost(X_test, y_test)
    
    # Step 10: Feature importance analysis
    feature_importance_df = predictor.feature_importance_analysis_xgboost()
    
    # Step 11: Decision matrix analysis
    cm, cm_percent, pred_confidence = predictor.decision_matrix_analysis_xgboost(X_test, y_test, eval_results['y_pred'])
    
    # Step 12: SHAP interpretability analysis
    shap_values = predictor.shap_interpretability_analysis_xgboost(X_test, max_samples=500)
    
    # Step 13: Generate comparative report
    report_df, confidence_stats = predictor.generate_comparative_report(X_test, y_test)
    
    print("\n🎉 XGBoost Cable Health Prediction Pipeline Completed!")
    print("\n📊 Final XGBoost Performance Summary:")
    print(f"  🎯 Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  🏆 F1-Score (Macro): {eval_results['f1_macro']:.4f}")
    print(f"  ⚖️  Matthews Correlation: {eval_results['mcc']:.4f}")
    print(f"  📈 Log Loss: {eval_results['log_loss']:.4f}")
    print(f"  🔧 Best Iteration: {eval_results['best_iteration']}")
    
    return predictor, eval_results, feature_importance_df, report_df, confidence_stats

if __name__ == "__main__":
    # Execute the XGBoost pipeline
    predictor_xgb, results_xgb, feature_importance_xgb, prediction_report_xgb, confidence_stats_xgb = main_xgboost()
