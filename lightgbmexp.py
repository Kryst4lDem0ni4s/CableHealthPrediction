"""
Cable Health Prediction System using LightGBM
==============================================
Advanced predictive analytics for cable health assessment with 3-class classification:
- 0: Healthy (Good condition, normal risk)
- 1: At Risk (Deterioration signs, preventive maintenance needed)  
- 2: Critical (Poor condition, immediate action required)

Algorithm Choice: LightGBM selected for optimal performance on large tabular datasets
with mixed feature types and class imbalance handling capabilities.
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
import shap
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🔧 Cable Health Prediction System Initialized")
print("📊 Algorithm: LightGBM (Gradient Boosting Machine)")
print("🎯 Task: Multi-class Classification (Healthy/At Risk/Critical)")

class CableHealthPredictor:
    """
    Advanced Cable Health Prediction System using LightGBM
    
    This class implements a comprehensive machine learning pipeline for predicting
    cable health status with emphasis on handling class imbalance and providing
    interpretable results for maintenance decision-making.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.class_weights = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        
        # Define categorical features based on domain knowledge
        self.categorical_columns = [
            'MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory',
            'FloodZoneRisk'
        ]
        
        print(f"✅ CableHealthPredictor initialized with random_state={random_state}")
    
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
        
        # Class distribution analysis - Critical for imbalanced learning
        print("\n🎯 Target Variable Analysis:")
        class_counts = df['CableHealthScore'].value_counts().sort_index()
        class_props = df['CableHealthScore'].value_counts(normalize=True).sort_index()
        
        for i, (count, prop) in enumerate(zip(class_counts, class_props)):
            status = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical'][i]
            print(f"Class {i} ({status}): {count:,} samples ({prop:.1%})")
        
        # Calculate imbalance ratio for optimization strategy
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"Imbalance Ratio: {imbalance_ratio:.2f} (Moderate imbalance detected)")
        
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
        print("\n🔍 Correlation Analysis...")
        
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
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=100):
        """
        Bayesian optimization for LightGBM hyperparameters using Hyperopt
        
        Theory: Bayesian optimization with Tree Parzen Estimator efficiently
        explores hyperparameter space by building probabilistic models of
        the objective function.
        """
        print(f"\n🎯 Hyperparameter Optimization ({n_trials} trials)...")
        
        def objective(params):
            """Objective function for Bayesian optimization"""
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
                'is_unbalance': True,  # Critical for class imbalance[3][7]
                'verbosity': -1,
                'random_state': self.random_state
            }
            
            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train, 
                                       categorical_feature=self.categorical_features)
                val_data = lgb.Dataset(X_fold_val, label=y_fold_val, 
                                     categorical_feature=self.categorical_features,
                                     reference=train_data)
                
                # Train model
                model = lgb.train(lgb_params, train_data, valid_sets=[val_data],
                                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                # Predict and calculate F1-macro score (optimal for imbalanced multi-class)
                y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
                y_pred_class = np.argmax(y_pred, axis=1)
                f1_macro = f1_score(y_fold_val, y_pred_class, average='macro')
                scores.append(f1_macro)
            
            return {'loss': -np.mean(scores), 'status': STATUS_OK}
        
        # Define search space based on LightGBM best practices[3][6]
        search_space = {
            'num_leaves': hp.choice('num_leaves', [31, 63, 127, 255]),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9, 11]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
            'feature_fraction': hp.uniform('feature_fraction', 0.6, 0.9),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.9),
            'bagging_freq': hp.choice('bagging_freq', [1, 3, 5, 7]),
            'min_data_in_leaf': hp.choice('min_data_in_leaf', [20, 50, 100, 200]),
            'lambda_l1': hp.uniform('lambda_l1', 0, 1),
            'lambda_l2': hp.uniform('lambda_l2', 0, 1)
        }
        
        # Run optimization
        trials = Trials()
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                   max_evals=n_trials, trials=trials,
                   early_stop_fn=no_progress_loss(20))
        
        # Convert best parameters back to LightGBM format
        self.best_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': [31, 63, 127, 255][best['num_leaves']],
            'max_depth': [3, 5, 7, 9, 11][best['max_depth']],
            'learning_rate': best['learning_rate'],
            'feature_fraction': best['feature_fraction'],
            'bagging_fraction': best['bagging_fraction'],
            'bagging_freq': [1, 3, 5, 7][best['bagging_freq']],
            'min_data_in_leaf': [20, 50, 100, 200][best['min_data_in_leaf']],
            'lambda_l1': best['lambda_l1'],
            'lambda_l2': best['lambda_l2'],
            'is_unbalance': True,
            'verbosity': -1,
            'random_state': self.random_state
        }
        
        best_score = -trials.best_trial['result']['loss']
        print(f"✅ Optimization completed. Best F1-macro score: {best_score:.4f}")
        
        return self.best_params
    
    def train_model(self, X_train, y_train, X_val, y_val, use_optimized_params=True):
        """
        Train LightGBM model with optimized parameters and early stopping
        
        Theory: Early stopping prevents overfitting by monitoring validation
        performance and stopping when no improvement is observed.
        """
        print("\n🚀 Training LightGBM Model...")
        
        # Use optimized parameters if available, otherwise use defaults
        if use_optimized_params and hasattr(self, 'best_params'):
            params = self.best_params.copy()
            print("Using optimized hyperparameters")
        else:
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
                'min_data_in_leaf': 100,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'is_unbalance': True,
                'verbosity': -1,
                'random_state': self.random_state
            }
            print("Using default hyperparameters")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, 
                               categorical_feature=self.categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, 
                             categorical_feature=self.categorical_features,
                             reference=train_data)
        
        # Train model with early stopping
        self.model = lgb.train(
            params, 
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print(f"✅ Training completed. Best iteration: {self.model.best_iteration}")
        
        return self.model
    
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
    
    def feature_importance_analysis(self):
        """
        Comprehensive feature importance analysis using multiple methods
        
        Theory: Feature importance helps identify key cable health indicators
        for maintenance decision-making and model interpretability.
        """
        print("\n🔍 Feature Importance Analysis...")
        
        # LightGBM built-in feature importance (split-based)[4]
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
    
    def shap_interpretability_analysis(self, X_test, max_samples=1000):
        """
        SHAP (SHapley Additive exPlanations) analysis for model interpretability
        
        Theory: SHAP provides game-theoretic explanations for individual predictions
        by computing the contribution of each feature to the final prediction.
        """
        print(f"\n🔬 SHAP Interpretability Analysis (sample size: {min(len(X_test), max_samples)})...")
        
        # Limit sample size for computational efficiency
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        print("✅ SHAP values computed")
        
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
        
        return shap_values
    
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

# Main execution pipeline
def main():
    """
    Main execution pipeline for cable health prediction system
    """
    print("🚀 Starting Cable Health Prediction Pipeline...")
    
    # Initialize predictor
    predictor = CableHealthPredictor(random_state=42)
    
    # Step 1: Load and explore data
    # Note: Replace with your actual dataset path
    df = predictor.load_and_explore_data('cable_health_sample_ordinal_encoded_20000.csv')
    
    # Step 2: Correlation analysis
    corr_matrix, high_corr_pairs = predictor.correlation_analysis(df)
    
    # Step 3: Advanced feature engineering
    df_enhanced, new_features = predictor.advanced_feature_engineering(df)
    
    # Step 4: Prepare features
    X, y = predictor.prepare_features(df_enhanced)
    
    # Step 5: Handle class imbalance
    class_weights = predictor.handle_class_imbalance(y)
    
    # Step 6: Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training set for validation
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Data Split Summary:")
    print(f"  Training set: {len(X_train_opt):,} samples")
    print(f"  Validation set: {len(X_val):,} samples") 
    print(f"  Test set: {len(X_test):,} samples")
    
    # Step 7: Hyperparameter optimization (optional - comment out for faster execution)
    # optimal_params = predictor.optimize_hyperparameters(X_train_opt, y_train_opt, n_trials=50)
    
    # Step 8: Train model
    model = predictor.train_model(X_train, y_train, X_val, y_val, use_optimized_params=False)
    
    # Step 9: Comprehensive evaluation
    eval_results = predictor.comprehensive_evaluation(X_test, y_test)
    
    # Step 10: Feature importance analysis
    feature_importance_df = predictor.feature_importance_analysis()
    
    # Step 11: Decision matrix analysis
    cm, cm_percent = predictor.decision_matrix_analysis(y_test, eval_results['y_pred'])
    
    # Step 12: SHAP interpretability analysis
    shap_values = predictor.shap_interpretability_analysis(X_test, max_samples=500)
    
    # Step 13: Generate prediction report
    report_df = predictor.generate_prediction_report(X_test, y_test)
    
    print("\n🎉 Cable Health Prediction Pipeline Completed Successfully!")
    print("\n📊 Final Performance Summary:")
    print(f"  🎯 Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  🏆 F1-Score (Macro): {eval_results['f1_macro']:.4f}")
    print(f"  ⚖️  Matthews Correlation: {eval_results['mcc']:.4f}")
    
    return predictor, eval_results, feature_importance_df, report_df

if __name__ == "__main__":
    # Execute the pipeline
    predictor, results, feature_importance, prediction_report = main()
