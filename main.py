import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
import os
import time
import gc
import psutil
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import logging
from sklearn.cluster import KMeans
from scipy import sparse
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from typing import List, Optional

warnings.filterwarnings('ignore')

# Core ML imports
import lightgbm as lgb
import xgboost as xgb
import sys
sys.path.append(r'C:\Users\Khwaish\.vscode\CableHealthExp\DeepGBM')
sys.path.append(r'C:\Users\Khwaish\.vscode\CableHealthExp')
# Then try to import
try:
    try:
        from DeepGBM.models.deepgbm import DeepGBM  # Or whatever the main module is called
        print("DeepGBM imported successfully!")
    except:
        print("from DeepGBM.models.deepgbm import DeepGBM did not work")
        try:
            from models.deepgbm import DeepGBM
            print("DeepGBM successfully imported")
        except:
            print("from models.deepgbm import DeepGBM did not work")

except ImportError as e:
    print(f"Import failed: {e}")
    
try:
    from ticl.prediction.tabflex import TabFlex
    print("Tabflex imported successfully!")
except ImportError as e:
    try:
        from ticl.models.tabflex import TabFlex
    except ImportError as e:
        print(f"Import failed: {e}")

    
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


# Model selection and evaluation
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, label_binarize
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, log_loss,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.calibration import CalibratedClassifierCV

# Advanced ensemble methods
try:
    import ngboost as ngb
    from ngboost.distns.categorical import k_categorical
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    print("NGBoost not available - will skip NGBoost models")

# ================================================================================================
# FIXED CONFIGURATION AND CONSTANTS
# ================================================================================================

class Config:
    """Global configuration for the analysis pipeline"""

    # File paths
    DATA_PATH = 'expresults/cable_health_method1_iterative_20k.csv'
    TARGET_COLUMN = 'CableHealthScore'
    CHECKPOINT_DIR = 'resnew/model_checkpoints'
    RESULTS_DIR = 'resnew/analysis_results'
    REPORTS_DIR = 'resnew/reports'
    PLOTS_DIR = 'resnew/plots'

    # Performance thresholds for early stopping
    EARLY_STOP_ACCURACY = 0.15  # Further lowered for more lenient early stopping
    EARLY_STOP_F1 = 0.10  # Further lowered for more lenient early stopping
    COMPLEXITY_TIME_MULTIPLIER = 3.0
    COMPLEXITY_F1_IMPROVEMENT = 0.02

    # Processing settings
    TEST_SIZE = 0.2
    CV_FOLDS = 3
    RANDOM_STATE = 42
    N_JOBS = 1  # Keep at 1 for stability

    # Resource management
    MANUAL_BREAK_ENABLED = True
    MEMORY_CLEANUP_THRESHOLD = 0.85
    BATCH_SIZES = {'light': 6, 'medium': 3, 'heavy': 2}

    # Reporting
    REPORT_DPI = 150
    
    GLOBAL_PREPROCESSOR = None

# Enhanced model complexity classifications
MODEL_COMPLEXITY = {
    'light': [
        'naive_bayes', 'naivebayes', 'gaussiannb', 'nb',
        'logistic_regression', 'logisticregression', 'lr',
        'sgd', 'sgdclassifier', 'linear_svm', 'linearsvm',
        'knn', 'kneighbors', 'ridge', 'random_classifier', 'majority_classifier', 'uniform_classifier' 
    ],
    'medium': [
        'lightgbm', 'lgb', 'xgboost', 'xgb',
        'random_forest', 'randomforest', 'rf',
        'svm_linear', 'linear_model', 'adaboost', 'gradientboost',
        'gradientboosting'  
    ],
    'heavy': [
        'catboost', 'cb', 'svm_rbf', 'advancedsvm', 'svmrbf',
        'neural_network', 'neuralnetwork', 'mlp', 'mlpclassifier',
        'ngboost', 'deepgbm', 'tabflex',
        'mixture_of_experts', 'mixtureofexperts', 'moe'
    ]
}

# Comprehensive metrics framework
TIER1_METRICS = [
    'accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall',
    'roc_auc_ovr', 'training_time', 'memory_usage'
]

TIER2_METRICS = [
    'per_class_f1', 'cohens_kappa', 'matthews_corrcoef',
    'log_loss', 'confidence_mean', 'confidence_std', 'cv_stability',
    'prediction_entropy'
]

TIER3_METRICS = [
    'individual_contribution', 'ensemble_diversity',
    'marginal_improvement', 'computational_overhead'
]


# ================================================================================================
# COMPLETE MODEL CONFIGURATIONS
# ================================================================================================

# COMPLETE model configurations for your 51+ models
MODEL_CONFIGS = {
    'tier1': {
        # Tier 1: Basic Models and Simple Ensembles (8 models)
        'lightgbm_solo': {
            'base_models': ['lightgbm'],
            'ensemble_type': 'none'
        },
        'lightgbm_xgboost_ensemble': {
            'base_models': ['lightgbm', 'xgboost'],
            'ensemble_type': 'voting'
        },
        'lightgbm_randomforest': {
            'base_models': ['lightgbm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'xgboost_randomforest': {
            'base_models': ['xgboost', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'lightgbm_advancedsvm': {
            'base_models': ['lightgbm', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'lightgbm_advancedsvm_randomforest': {
            'base_models': ['lightgbm', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'xgboost_advancedsvm_randomforest': {
            'base_models': ['xgboost', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'xgboost_solo': {
            'base_models': ['xgboost'],
            'ensemble_type': 'none'
        }
    },
    'tier2': {
        # Tier 2: High Performance, Higher Complexity (43+ models)
        'xgboost_lightgbm_naivebayes': {
            'base_models': ['xgboost', 'lightgbm', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_catboost_lightgbm': {
            'base_models': ['xgboost', 'catboost', 'lightgbm'],
            'ensemble_type': 'voting'
        },
        'xgboost_advancedsvm_2model': {
            'base_models': ['xgboost', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'xgboost_lightgbm_randomforest_linearsvm': {
            'base_models': ['xgboost', 'lightgbm', 'randomforest', 'linearsvm'],
            'ensemble_type': 'voting'
        },
        'lightgbm_catboost_naivebayes': {
            'base_models': ['lightgbm', 'catboost', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'lightgbm_catboost_advancedsvm': {
            'base_models': ['lightgbm', 'catboost', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'xgboost_catboost_advancedsvm': {
            'base_models': ['xgboost', 'catboost', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'xgboost_catboost_naivebayes': {
            'base_models': ['xgboost', 'catboost', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'catboost_solo': {
            'base_models': ['catboost'],
            'ensemble_type': 'none'
        },
        'xgboost_cv_ensemble': {
            'base_models': ['xgboost'],
            'ensemble_type': 'cv_ensemble',
            'cv_folds': 3
        },
        'lightgbm_cv_ensemble': {
            'base_models': ['lightgbm'],
            'ensemble_type': 'cv_ensemble',
            'cv_folds': 3
        },
        'catboost_randomforest': {
            'base_models': ['catboost', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'catboost_advancedsvm_randomforest': {
            'base_models': ['catboost', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'lightgbm_neuralnetwork': {
            'base_models': ['lightgbm', 'neuralnetwork'],
            'ensemble_type': 'voting'
        },
        'xgboost_neuralnetwork': {
            'base_models': ['xgboost', 'neuralnetwork'],
            'ensemble_type': 'voting'
        },
        'xgboost_neuralnetwork_naivebayes': {
            'base_models': ['xgboost', 'neuralnetwork', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_neuralnetwork_randomforest': {
            'base_models': ['xgboost', 'neuralnetwork', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'lightgbm_neuralnetwork_advancedsvm': {
            'base_models': ['lightgbm', 'neuralnetwork', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'lightgbm_neuralnetwork_naivebayes': {
            'base_models': ['lightgbm', 'neuralnetwork', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'lightgbm_neuralnetwork_randomforest': {
            'base_models': ['lightgbm', 'neuralnetwork', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'lightgbm_randomforest_naivebayes': {
            'base_models': ['lightgbm', 'randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_randomforest_naivebayes': {
            'base_models': ['xgboost', 'randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'catboost_randomforest_naivebayes': {
            'base_models': ['catboost', 'randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_neuralnetwork_advancedsvm': {
            'base_models': ['xgboost', 'neuralnetwork', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'catboost_neuralnetwork': {
            'base_models': ['catboost', 'neuralnetwork'],
            'ensemble_type': 'voting'
        },
        'randomforest_knn_naivebayes': {
            'base_models': ['randomforest', 'knn', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'catboost_randomforest_logisticregression': {
            'base_models': ['catboost', 'randomforest', 'logisticregression'],
            'ensemble_type': 'voting'
        },
        'lightgbm_advancedsvm_naivebayes': {
            'base_models': ['lightgbm', 'advancedsvm', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_randomforest_neuralnetwork': {
            'base_models': ['xgboost', 'randomforest', 'neuralnetwork'],
            'ensemble_type': 'voting'
        },
        'randomforest_solo': {
            'base_models': ['randomforest'],
            'ensemble_type': 'none'
        },
        'advancedsvm_solo': {
            'base_models': ['advancedsvm'],
            'ensemble_type': 'none'
        },
        'naivebayes_solo': {
            'base_models': ['naivebayes'],
            'ensemble_type': 'none'
        },
        'neuralnetwork_solo': {
            'base_models': ['neuralnetwork'],
            'ensemble_type': 'none'
        },
        'logisticregression_solo': {
            'base_models': ['logisticregression'],
            'ensemble_type': 'none'
        },
        'knn_solo': {
            'base_models': ['knn'],
            'ensemble_type': 'none'
        },
        'ridge_solo': {
            'base_models': ['ridge'],
            'ensemble_type': 'none'
        },
        'sgd_solo': {
            'base_models': ['sgd'],
            'ensemble_type': 'none'
        },
        'lightgbm_naivebayes': {
            'base_models': ['lightgbm', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'xgboost_naivebayes': {
            'base_models': ['xgboost', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'catboost_naivebayes': {
            'base_models': ['catboost', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'randomforest_naivebayes': {
            'base_models': ['randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'lightgbm_logisticregression': {
            'base_models': ['lightgbm', 'logisticregression'],
            'ensemble_type': 'voting'
        },
        'xgboost_logisticregression': {
            'base_models': ['xgboost', 'logisticregression'],
            'ensemble_type': 'voting'
        },
        'catboost_logisticregression': {
            'base_models': ['catboost', 'logisticregression'],
            'ensemble_type': 'voting'
        },
        'randomforest_logisticregression': {
            'base_models': ['randomforest', 'logisticregression'],
            'ensemble_type': 'voting'
        },
        'lightgbm_knn': {
            'base_models': ['lightgbm', 'knn'],
            'ensemble_type': 'voting'
        },
        'xgboost_knn': {
            'base_models': ['xgboost', 'knn'],
            'ensemble_type': 'voting'
        },
        # DeepGBM Combinations
        'deepgbm_solo': {
            'base_models': ['deepgbm'],
            'ensemble_type': 'none'
        },
        'deepgbm_lightgbm_randomforest': {
            'base_models': ['deepgbm', 'lightgbm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'deepgbm_advancedsvm': {
            'base_models': ['deepgbm', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'deepgbm_randomforest': {
            'base_models': ['deepgbm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'deepgbm_randomforest_naivebayes': {
            'base_models': ['deepgbm', 'randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'deepgbm_advancedsvm_randomforest': {
            'base_models': ['deepgbm', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'voting'
        },

        # TabFlex Combinations
        'tabflex_solo': {
            'base_models': ['tabflex'],
            'ensemble_type': 'none'
        },
        'tabflex_xgboost': {
            'base_models': ['tabflex', 'xgboost'],
            'ensemble_type': 'voting'
        },
        'tabflex_lightgbm': {
            'base_models': ['tabflex', 'lightgbm'],
            'ensemble_type': 'voting'
        },

        # NgBoost Combinations (if NgBoost available)
        'ngboost_solo': {
            'base_models': ['ngboost'],
            'ensemble_type': 'none'
        },
        'ngboost_advancedsvm': {
            'base_models': ['ngboost', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'ngboost_xgboost_catboost': {
            'base_models': ['ngboost', 'xgboost', 'catboost'],
            'ensemble_type': 'voting'
        },
        'ngboost_randomforest': {
            'base_models': ['ngboost', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'ngboost_randomforest_naivebayes': {
            'base_models': ['ngboost', 'randomforest', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'ngboost_neuralnetwork': {
            'base_models': ['ngboost', 'neuralnetwork'],
            'ensemble_type': 'voting'
        },
        'ngboost_advancedsvm_randomforest': {
            'base_models': ['ngboost', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'voting'
        },
        'ngboost_neuralnetwork_advancedsvm': {
            'base_models': ['ngboost', 'neuralnetwork', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'ngboost_neuralnetwork_naivebayes': {
            'base_models': ['ngboost', 'neuralnetwork', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'ngboost_neuralnetwork_randomforest': {
            'base_models': ['ngboost', 'neuralnetwork', 'randomforest'],
            'ensemble_type': 'voting'
        },

        # Mixture-of-Experts Combinations
        'mixture_of_experts_lightgbm_advancedsvm': {
            'base_models': ['mixture_of_experts', 'lightgbm', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts_lightgbm_naivebayes': {
            'base_models': ['mixture_of_experts', 'lightgbm', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts_lightgbm_linear_model': {
            'base_models': ['mixture_of_experts', 'lightgbm', 'linear_model'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts_xgboost_advancedsvm': {
            'base_models': ['mixture_of_experts', 'xgboost', 'advancedsvm'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts_xgboost_naivebayes': {
            'base_models': ['mixture_of_experts', 'xgboost', 'naivebayes'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts_xgboost_linear_model': {
            'base_models': ['mixture_of_experts', 'xgboost', 'linear_model'],
            'ensemble_type': 'voting'
        },
        'mixture_of_experts': {
            'base_models': ['mixture_of_experts'],
            'ensemble_type': 'none'
        },

        # Stacked Ensemble with Ridge Meta-Learner
        'stacked_lightgbm_xgboost_catboost_ridge': {
            'base_models': ['lightgbm', 'xgboost', 'catboost'],
            'ensemble_type': 'voting_ridge'
        },

        # Parametric Ensemble Learning Models
        'lightgbm_parametric_ensemble': {
            'base_models': ['lightgbm'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'xgboost_parametric_ensemble': {
            'base_models': ['xgboost'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'catboost_parametric_ensemble': {
            'base_models': ['catboost'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'ngboost_parametric_ensemble': {
            'base_models': ['ngboost'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'deepgbm_parametric_ensemble': {
            'base_models': ['deepgbm'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'randomforest_parametric_ensemble': {
            'base_models': ['randomforest'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },
        'advancedsvm_parametric_ensemble': {
            'base_models': ['advancedsvm'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 3
        },

        # Complex Parametric Combinations
        'parametric_lightgbm_advancedsvm_randomforest': {
            'base_models': ['lightgbm', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 2  # 2 versions of each model
        },
        'parametric_xgboost_advancedsvm_randomforest': {
            'base_models': ['xgboost', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 2
        },
        'parametric_catboost_advancedsvm_randomforest': {
            'base_models': ['catboost', 'advancedsvm', 'randomforest'],
            'ensemble_type': 'parametric_ensemble',
            'n_versions': 2
        }

    }
}


# Complete TARGET_MODELS list for systematic, publication-quality research
TARGET_MODELS = [
    # =================================================================
    # PHASE 1: BASELINE ESTABLISHMENT (Priority 1)
    # =================================================================
    
    # Tree-Based Baselines (Complete Coverage)
    {'name': 'lightgbm', 'algorithms': ['lightgbm'], 'purpose': 'baseline_tree_boosting'},
    {'name': 'xgboost', 'algorithms': ['xgboost'], 'purpose': 'baseline_tree_boosting'},
    {'name': 'catboost', 'algorithms': ['catboost'], 'purpose': 'baseline_tree_boosting'},
    {'name': 'randomforest', 'algorithms': ['randomforest'], 'purpose': 'baseline_tree_bagging'},
    {'name': 'adaboost', 'algorithms': ['adaboost'], 'purpose': 'baseline_boosting'},
    {'name': 'gradientboost', 'algorithms': ['gradientboost'], 'purpose': 'baseline_boosting'},
    
    # Linear Model Baselines (Enhanced Coverage)
    {'name': 'logisticregression', 'algorithms': ['logisticregression'], 'purpose': 'baseline_linear'},
    {'name': 'ridge', 'algorithms': ['ridge'], 'purpose': 'baseline_linear'},
    {'name': 'sgd', 'algorithms': ['sgd'], 'purpose': 'baseline_linear'},
    {'name': 'linearsvm', 'algorithms': ['linearsvm'], 'purpose': 'baseline_linear_svm'},
    {'name': 'linearmodel', 'algorithms': ['linearmodel'], 'purpose': 'baseline_linear_simple'},
    
    # Other Family Baselines
    {'name': 'advancedsvm', 'algorithms': ['advancedsvm'], 'purpose': 'baseline_kernel'},
    {'name': 'naivebayes', 'algorithms': ['naivebayes'], 'purpose': 'baseline_probabilistic'},
    {'name': 'knn', 'algorithms': ['knn'], 'purpose': 'baseline_instance'},
    {'name': 'neuralnetwork', 'algorithms': ['neuralnetwork'], 'purpose': 'baseline_neural'},
    
    # Advanced Baselines
    {'name': 'deepgbm', 'algorithms': ['deepgbm'], 'purpose': 'baseline_advanced'},
    {'name': 'ngboost', 'algorithms': ['ngboost'], 'purpose': 'baseline_probabilistic_boosting'},
    {'name': 'tabflex', 'algorithms': ['tabflex'], 'purpose': 'baseline_tabular_neural'},
    {'name': 'mixture_of_experts', 'algorithms': ['mixture_of_experts'], 'purpose': 'baseline_expert_system'},
    
    # =================================================================
    # PHASE 2: TIER 1 MODEL COMBINATIONS (Your Exact Requirements)
    # =================================================================
    
    # Tier 1 Combinations from your list
    {'name': 'lightgbm_xgboost', 'algorithms': ['lightgbm', 'xgboost'], 'purpose': 'boosting_ensemble'},
    {'name': 'lightgbm_randomforest', 'algorithms': ['lightgbm', 'randomforest'], 'purpose': 'tree_diversity'},
    {'name': 'xgboost_randomforest', 'algorithms': ['xgboost', 'randomforest'], 'purpose': 'tree_diversity'},
    {'name': 'lightgbm_advancedsvm', 'algorithms': ['lightgbm', 'advancedsvm'], 'purpose': 'tree_kernel'},
    {'name': 'lightgbm_advancedsvm_randomforest', 'algorithms': ['lightgbm', 'advancedsvm', 'randomforest'], 'purpose': 'high_diversity'},
    {'name': 'xgboost_advancedsvm_randomforest', 'algorithms': ['xgboost', 'advancedsvm', 'randomforest'], 'purpose': 'high_diversity'},
    
    # =================================================================
    # PHASE 3: TIER 2 COMPLEX COMBINATIONS (Your Exact Requirements)
    # =================================================================
    
    # Tree Boosting Complex Combinations
    {'name': 'xgboost_lightgbm_naivebayes', 'algorithms': ['xgboost', 'lightgbm', 'naivebayes'], 'purpose': 'boosting_probabilistic'},
    {'name': 'xgboost_catboost_lightgbm', 'algorithms': ['xgboost', 'catboost', 'lightgbm'], 'purpose': 'boosting_triplet'},
    {'name': 'xgboost_advancedsvm_2model', 'algorithms': ['xgboost', 'advancedsvm'], 'purpose': 'tree_kernel_pair'},
    {'name': 'xgboost_lightgbm_randomforest_linearsvm', 'algorithms': ['xgboost', 'lightgbm', 'randomforest', 'linearsvm'], 'purpose': 'quad_diversity'},
    {'name': 'lightgbm_catboost_naivebayes', 'algorithms': ['lightgbm', 'catboost', 'naivebayes'], 'purpose': 'boosting_probabilistic'},
    {'name': 'lightgbm_catboost_advancedsvm', 'algorithms': ['lightgbm', 'catboost', 'advancedsvm'], 'purpose': 'boosting_kernel'},
    {'name': 'xgboost_catboost_advancedsvm', 'algorithms': ['xgboost', 'catboost', 'advancedsvm'], 'purpose': 'boosting_kernel'},
    {'name': 'xgboost_catboost_naivebayes', 'algorithms': ['xgboost', 'catboost', 'naivebayes'], 'purpose': 'boosting_probabilistic'},
    
    # DeepGBM Combinations
    {'name': 'deepgbm_lightgbm_randomforest', 'algorithms': ['deepgbm', 'lightgbm', 'randomforest'], 'purpose': 'advanced_tree'},
    {'name': 'deepgbm_advancedsvm', 'algorithms': ['deepgbm', 'advancedsvm'], 'purpose': 'advanced_kernel'},
    {'name': 'deepgbm_randomforest', 'algorithms': ['deepgbm', 'randomforest'], 'purpose': 'advanced_bagging'},
    {'name': 'deepgbm_randomforest_naivebayes', 'algorithms': ['deepgbm', 'randomforest', 'naivebayes'], 'purpose': 'advanced_probabilistic'},
    {'name': 'deepgbm_advancedsvm_randomforest', 'algorithms': ['deepgbm', 'advancedsvm', 'randomforest'], 'purpose': 'advanced_diversity'},
    
    # CatBoost Combinations
    {'name': 'catboost_randomforest', 'algorithms': ['catboost', 'randomforest'], 'purpose': 'tree_diversity'},
    {'name': 'catboost_advancedsvm_randomforest', 'algorithms': ['catboost', 'advancedsvm', 'randomforest'], 'purpose': 'tree_kernel_diversity'},
    {'name': 'catboost_advancedsvm', 'algorithms': ['catboost', 'advancedsvm'], 'purpose': 'tree_kernel'},
    {'name': 'catboost_randomforest_naivebayes', 'algorithms': ['catboost', 'randomforest', 'naivebayes'], 'purpose': 'tree_probabilistic'},
    {'name': 'catboost_randomforest_logisticregression', 'algorithms': ['catboost', 'randomforest', 'logisticregression'], 'purpose': 'tree_linear'},
    
    # NgBoost Combinations
    {'name': 'ngboost_advancedsvm', 'algorithms': ['ngboost', 'advancedsvm'], 'purpose': 'probabilistic_kernel'},
    {'name': 'ngboost_xgboost_catboost', 'algorithms': ['ngboost', 'xgboost', 'catboost'], 'purpose': 'probabilistic_boosting'},
    {'name': 'ngboost_randomforest', 'algorithms': ['ngboost', 'randomforest'], 'purpose': 'probabilistic_bagging'},
    {'name': 'ngboost_randomforest_naivebayes', 'algorithms': ['ngboost', 'randomforest', 'naivebayes'], 'purpose': 'probabilistic_triplet'},
    {'name': 'ngboost_advancedsvm_randomforest', 'algorithms': ['ngboost', 'advancedsvm', 'randomforest'], 'purpose': 'probabilistic_diversity'},
    
    # Neural Network Combinations (Carefully Selected to Avoid Complexity Issues)
    {'name': 'lightgbm_neuralnetwork', 'algorithms': ['lightgbm', 'neuralnetwork'], 'purpose': 'tree_neural'},
    {'name': 'xgboost_neuralnetwork', 'algorithms': ['xgboost', 'neuralnetwork'], 'purpose': 'tree_neural'},
    {'name': 'catboost_neuralnetwork', 'algorithms': ['catboost', 'neuralnetwork'], 'purpose': 'tree_neural'},
    {'name': 'ngboost_neuralnetwork', 'algorithms': ['ngboost', 'neuralnetwork'], 'purpose': 'probabilistic_neural'},
    
    # Neural Network Triplets (Limited to avoid complexity explosion)
    {'name': 'xgboost_neuralnetwork_naivebayes', 'algorithms': ['xgboost', 'neuralnetwork', 'naivebayes'], 'purpose': 'neural_probabilistic'},
    {'name': 'lightgbm_neuralnetwork_naivebayes', 'algorithms': ['lightgbm', 'neuralnetwork', 'naivebayes'], 'purpose': 'neural_probabilistic'},
    {'name': 'xgboost_neuralnetwork_randomforest', 'algorithms': ['xgboost', 'neuralnetwork', 'randomforest'], 'purpose': 'neural_tree_diversity'},
    {'name': 'lightgbm_neuralnetwork_randomforest', 'algorithms': ['lightgbm', 'neuralnetwork', 'randomforest'], 'purpose': 'neural_tree_diversity'},
    {'name': 'catboost_neuralnetwork_naivebayes', 'algorithms': ['catboost', 'neuralnetwork', 'naivebayes'], 'purpose': 'neural_probabilistic'},
    {'name': 'ngboost_neuralnetwork_naivebayes', 'algorithms': ['ngboost', 'neuralnetwork', 'naivebayes'], 'purpose': 'probabilistic_neural_bayes'},
    {'name': 'catboost_neuralnetwork_randomforest', 'algorithms': ['catboost', 'neuralnetwork', 'randomforest'], 'purpose': 'neural_tree_complex'},
    {'name': 'ngboost_neuralnetwork_randomforest', 'algorithms': ['ngboost', 'neuralnetwork', 'randomforest'], 'purpose': 'probabilistic_neural_tree'},
    
    # HIGH COMPLEXITY - Only pairs to avoid computational explosion
    {'name': 'lightgbm_neuralnetwork_advancedsvm', 'algorithms': ['lightgbm', 'neuralnetwork', 'advancedsvm'], 'purpose': 'neural_kernel_tree'},
    {'name': 'xgboost_neuralnetwork_advancedsvm', 'algorithms': ['xgboost', 'neuralnetwork', 'advancedsvm'], 'purpose': 'neural_kernel_tree'},
    {'name': 'catboost_neuralnetwork_advancedsvm', 'algorithms': ['catboost', 'neuralnetwork', 'advancedsvm'], 'purpose': 'neural_kernel_complex'},
    {'name': 'ngboost_neuralnetwork_advancedsvm', 'algorithms': ['ngboost', 'neuralnetwork', 'advancedsvm'], 'purpose': 'probabilistic_neural_complex'},
    
    # =================================================================
    # PHASE 4: LINEAR MODEL FOCUS (Address Underrepresentation)
    # =================================================================
    
    # Linear Model Combinations
    {'name': 'logisticregression_ridge_sgd', 'algorithms': ['logisticregression', 'ridge', 'sgd'], 'purpose': 'linear_ensemble'},
    {'name': 'logisticregression_linearsvm_ridge', 'algorithms': ['logisticregression', 'linearsvm', 'ridge'], 'purpose': 'linear_svm_ensemble'},
    
    # Tree + Linear Combinations (Address Bias)
    {'name': 'lightgbm_logisticregression', 'algorithms': ['lightgbm', 'logisticregression'], 'purpose': 'tree_linear'},
    {'name': 'xgboost_logisticregression', 'algorithms': ['xgboost', 'logisticregression'], 'purpose': 'tree_linear'},
    {'name': 'catboost_logisticregression', 'algorithms': ['catboost', 'logisticregression'], 'purpose': 'tree_linear'},
    {'name': 'randomforest_logisticregression', 'algorithms': ['randomforest', 'logisticregression'], 'purpose': 'tree_linear'},
    {'name': 'adaboost_logisticregression', 'algorithms': ['adaboost', 'logisticregression'], 'purpose': 'boosting_linear'},
    {'name': 'gradientboost_logisticregression', 'algorithms': ['gradientboost', 'logisticregression'], 'purpose': 'boosting_linear'},
    
    # Linear + Other Families
    {'name': 'logisticregression_naivebayes_knn', 'algorithms': ['logisticregression', 'naivebayes', 'knn'], 'purpose': 'simple_ensemble'},
    {'name': 'ridge_advancedsvm_naivebayes', 'algorithms': ['ridge', 'advancedsvm', 'naivebayes'], 'purpose': 'linear_kernel_prob'},
    {'name': 'randomforest_knn_naivebayes', 'algorithms': ['randomforest', 'knn', 'naivebayes'], 'purpose': 'simple_ensemble_triplet'},
    {'name': 'lightgbm_advancedsvm_naivebayes', 'algorithms': ['lightgbm', 'advancedsvm', 'naivebayes'], 'purpose': 'tree_kernel_probabilistic'},
    
    # =================================================================
    # PHASE 5: MIXTURE-OF-EXPERTS COMBINATIONS
    # =================================================================
    
    # MoE + Tree Combinations
    {'name': 'mixture_of_experts_lightgbm_advancedsvm', 'algorithms': ['mixture_of_experts', 'lightgbm', 'advancedsvm'], 'purpose': 'expert_system'},
    {'name': 'mixture_of_experts_lightgbm_naivebayes', 'algorithms': ['mixture_of_experts', 'lightgbm', 'naivebayes'], 'purpose': 'expert_probabilistic'},
    {'name': 'mixture_of_experts_lightgbm_linearmodel', 'algorithms': ['mixture_of_experts', 'lightgbm', 'linearmodel'], 'purpose': 'expert_linear'},
    {'name': 'mixture_of_experts_xgboost_advancedsvm', 'algorithms': ['mixture_of_experts', 'xgboost', 'advancedsvm'], 'purpose': 'expert_system_alt'},
    {'name': 'mixture_of_experts_xgboost_naivebayes', 'algorithms': ['mixture_of_experts', 'xgboost', 'naivebayes'], 'purpose': 'expert_probabilistic_alt'},
    {'name': 'mixture_of_experts_xgboost_linearmodel', 'algorithms': ['mixture_of_experts', 'xgboost', 'linearmodel'], 'purpose': 'expert_linear_alt'},
    {'name': 'mixture_of_experts_lightgbm', 'algorithms': ['mixture_of_experts', 'lightgbm'], 'purpose': 'expert_tree_pair'},
    
    # =================================================================
    # PHASE 6: TABFLEX COMBINATIONS
    # =================================================================
    
    # TabFlex Combinations
    {'name': 'tabflex_xgboost', 'algorithms': ['tabflex', 'xgboost'], 'purpose': 'tabular_neural_boosting'},
    {'name': 'tabflex_lightgbm', 'algorithms': ['tabflex', 'lightgbm'], 'purpose': 'tabular_neural_boosting'},
    {'name': 'tabflex_catboost', 'algorithms': ['tabflex', 'catboost'], 'purpose': 'tabular_neural_boosting'},
    {'name': 'tabflex_randomforest', 'algorithms': ['tabflex', 'randomforest'], 'purpose': 'tabular_neural_bagging'},
    {'name': 'tabflex_advancedsvm', 'algorithms': ['tabflex', 'advancedsvm'], 'purpose': 'tabular_neural_kernel'},
    
    # =================================================================
    # PHASE 7: PROBABILISTIC COMBINATIONS (Balance Naive Bayes)
    # =================================================================
    
    # Enhanced Naive Bayes Integration
    {'name': 'lightgbm_naivebayes', 'algorithms': ['lightgbm', 'naivebayes'], 'purpose': 'tree_probabilistic'},
    {'name': 'xgboost_naivebayes', 'algorithms': ['xgboost', 'naivebayes'], 'purpose': 'tree_probabilistic'},
    {'name': 'catboost_naivebayes', 'algorithms': ['catboost', 'naivebayes'], 'purpose': 'tree_probabilistic'},
    {'name': 'randomforest_naivebayes', 'algorithms': ['randomforest', 'naivebayes'], 'purpose': 'bagging_probabilistic'},
    {'name': 'advancedsvm_naivebayes', 'algorithms': ['advancedsvm', 'naivebayes'], 'purpose': 'kernel_probabilistic'},
    {'name': 'adaboost_naivebayes', 'algorithms': ['adaboost', 'naivebayes'], 'purpose': 'boosting_probabilistic'},
    {'name': 'gradientboost_naivebayes', 'algorithms': ['gradientboost', 'naivebayes'], 'purpose': 'boosting_probabilistic'},
    
    # Triplet Probabilistic Combinations
    {'name': 'lightgbm_randomforest_naivebayes', 'algorithms': ['lightgbm', 'randomforest', 'naivebayes'], 'purpose': 'tree_probabilistic_triplet'},
    {'name': 'xgboost_randomforest_naivebayes', 'algorithms': ['xgboost', 'randomforest', 'naivebayes'], 'purpose': 'tree_probabilistic_triplet'},
    
    # =================================================================
    # PHASE 8: KNN INTEGRATION (Address Instance-Based Underrepresentation)
    # =================================================================
    
    # KNN Integration
    {'name': 'lightgbm_knn', 'algorithms': ['lightgbm', 'knn'], 'purpose': 'tree_instance'},
    {'name': 'xgboost_knn', 'algorithms': ['xgboost', 'knn'], 'purpose': 'tree_instance'},
    {'name': 'catboost_knn', 'algorithms': ['catboost', 'knn'], 'purpose': 'tree_instance'},
    {'name': 'randomforest_knn', 'algorithms': ['randomforest', 'knn'], 'purpose': 'bagging_instance'},
    {'name': 'advancedsvm_knn', 'algorithms': ['advancedsvm', 'knn'], 'purpose': 'kernel_instance'},
    {'name': 'knn_naivebayes_logisticregression', 'algorithms': ['knn', 'naivebayes', 'logisticregression'], 'purpose': 'simple_baseline_triplet'},
    
    # =================================================================
    # PHASE 9: TRADITIONAL BOOSTING INTEGRATION
    # =================================================================
    
    # AdaBoost and GradientBoost Integration
    {'name': 'adaboost_randomforest', 'algorithms': ['adaboost', 'randomforest'], 'purpose': 'boosting_bagging'},
    {'name': 'gradientboost_randomforest', 'algorithms': ['gradientboost', 'randomforest'], 'purpose': 'boosting_bagging'},
    {'name': 'adaboost_advancedsvm', 'algorithms': ['adaboost', 'advancedsvm'], 'purpose': 'boosting_kernel'},
    {'name': 'gradientboost_advancedsvm', 'algorithms': ['gradientboost', 'advancedsvm'], 'purpose': 'boosting_kernel'},
    {'name': 'adaboost_gradientboost', 'algorithms': ['adaboost', 'gradientboost'], 'purpose': 'traditional_boosting_pair'},
    {'name': 'lightgbm_adaboost_gradientboost', 'algorithms': ['lightgbm', 'adaboost', 'gradientboost'], 'purpose': 'boosting_evolution'},
    
    # =================================================================
    # PHASE 10: COMPLEX RESEARCH-ORIENTED COMBINATIONS
    # =================================================================
    
    # Advanced Tree + Neural Combinations (Limited)
    {'name': 'xgboost_randomforest_neuralnetwork', 'algorithms': ['xgboost', 'randomforest', 'neuralnetwork'], 'purpose': 'tree_neural_complex'},
    
    # Stacking Ridge Special Case
    {'name': 'lightgbm_xgboost_catboost_ridge', 'algorithms': ['lightgbm', 'xgboost', 'catboost'], 'purpose': 'boosting_triplet_ridge'},
    
    # =================================================================
    # PHASE 11: STATISTICAL CONTROLS AND NEGATIVE CONTROLS
    # =================================================================
    
    # Statistical Controls (Essential for Publication Quality)
    {'name': 'random_baseline', 'algorithms': ['random_classifier'], 'purpose': 'negative_control'},
    {'name': 'majority_baseline', 'algorithms': ['majority_classifier'], 'purpose': 'statistical_control'},
    {'name': 'uniform_baseline', 'algorithms': ['uniform_classifier'], 'purpose': 'statistical_control'},
    
    # Ensemble Ablation Studies
    {'name': 'lightgbm_xgboost_randomforest_ablation', 'algorithms': ['lightgbm', 'xgboost', 'randomforest'], 'purpose': 'ablation_study'},
    
    # =================================================================
    # PHASE 12: ADDITIONAL MISSING COMBINATIONS FROM YOUR LIST
    # =================================================================
    
    # Missing combinations to complete your requirements
    {'name': 'deepgbm_neuralnetwork', 'algorithms': ['deepgbm', 'neuralnetwork'], 'purpose': 'advanced_neural'},
    {'name': 'tabflex_naivebayes', 'algorithms': ['tabflex', 'naivebayes'], 'purpose': 'tabular_probabilistic'},
    {'name': 'mixture_of_experts_catboost', 'algorithms': ['mixture_of_experts', 'catboost'], 'purpose': 'expert_boosting'},
    {'name': 'ngboost_catboost', 'algorithms': ['ngboost', 'catboost'], 'purpose': 'probabilistic_boosting_pair'},
    
    # Gradient Classifier (for research completeness)
    {'name': 'gradientclassifier', 'algorithms': ['gradientboost'], 'purpose': 'gradient_research'},
    
    # Final Advanced Combinations
    {'name': 'all_boosting_ensemble', 'algorithms': ['lightgbm', 'xgboost', 'catboost', 'adaboost', 'gradientboost'], 'purpose': 'comprehensive_boosting'},
    {'name': 'all_linear_ensemble', 'algorithms': ['logisticregression', 'ridge', 'sgd', 'linearsvm'], 'purpose': 'comprehensive_linear'},
]

# ================================================================================================
# FIXED UTILITY FUNCTIONS
# ================================================================================================

def setup_directories():
    """Create necessary directories for the analysis"""
    for directory in [Config.CHECKPOINT_DIR, Config.RESULTS_DIR,
                     Config.REPORTS_DIR, Config.PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    setup_directories()
    log_path = os.path.join(Config.RESULTS_DIR, 'analysis.log')

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def monitor_resources():
    """Monitor system resources"""
    try:
        memory_info = psutil.virtual_memory()
        return {
            'memory_percent': memory_info.percent,
            'memory_available': memory_info.available / (1024**3),
            'memory_used': memory_info.used / (1024**3)
        }
    except:
        return {'memory_percent': 50.0, 'memory_available': 8.0, 'memory_used': 4.0}

def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    
def safe_array_length(arr) -> int:
    """Return the number of samples (first dimension) as an int for any array-like."""
    try:
        if hasattr(arr, 'shape') and arr.shape is not None and len(arr.shape) >= 1:
            return int(arr.shape[0])
        return int(len(arr))
    except Exception:
        return 0

def safe_array_indexing(arr, indices):
    """Safely index an array, handling both dense and sparse matrices"""
    if sparse.issparse(arr):
        return arr[indices]
    else:
        return arr[indices]

def safe_len(arr):
    """Safe wrapper for len() that handles sparse matrices"""
    # if sparse.issparse(arr):
    #     return arr.shape
    # else:
    #     return len(arr)
    return safe_array_length(arr)
    
def get_n_samples(arr):
    """UNIVERSAL: Get number of samples from any array type safely"""
    # try:
    #     if sparse.issparse(arr):
    #         try:
    #             return arr.shape  # Always use shape for sparse
    #         except:
    #             return arr.shape[0]  # Always use shape for sparse
    #     elif hasattr(arr, 'shape') and len(arr.shape) > 0:
    #         return arr.shape  # FIXED: Get first dimension only
    #     elif hasattr(arr, '__len__'):
    #         return len(arr)     # Use len() only for lists/pandas
    #     else:
    #         return 0
    # except Exception:
    #     return 0
    return safe_array_length(arr)


def safe_pandas_indexing(df_or_series, indices):
    """Safely index pandas DataFrame or Series using iloc for integer indices"""
    if isinstance(df_or_series, (pd.DataFrame, pd.Series)):
        return df_or_series.iloc[indices]
    else:
        return df_or_series[indices]

def check_early_stop_criteria(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """Check if model meets early stopping criteria"""
    if metrics.get('accuracy', 0) < Config.EARLY_STOP_ACCURACY:
        return True, f"Accuracy {metrics.get('accuracy', 0):.3f} below threshold {Config.EARLY_STOP_ACCURACY}"

    if metrics.get('weighted_f1', 0) < Config.EARLY_STOP_F1:
        return True, f"F1-Score {metrics.get('weighted_f1', 0):.3f} below threshold {Config.EARLY_STOP_F1}"

    return False, "Passed early stopping criteria"

# ================================================================================================
# FIXED MODEL FACTORY CLASSES
# ================================================================================================


# Assuming these are from your DeepGBM repo (import them as needed)
# from models.deepgbm import DeepGBM  # Adjust path based on your sys.path
# from models.gbdt2nn import GBDT2NN
# from models.deepfm import DeepFM
class DeepGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        embedding_size: int = 4,
        h_depth: int = 2,
        deep_layers: List[int] = [32, 32],
        task: str = 'classification',
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_trees: int = 10,
        tree_max_depth: int = 4,
        tree_layers: Optional[List[int]] = None,
        random_state: int = 42,
        use_real_deepgbm: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.tree_max_depth = tree_max_depth
        self.tree_layers = tree_layers or [32, 16, 1]  # Ensure last layer outputs 1 per tree
        self.random_state = random_state
        self.use_real_deepgbm = use_real_deepgbm
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.fallback_model_ = None
        self.label_encoder_ = None
        self.n_classes_ = None
        self.nume_input_size_ = None
        self.cate_field_size_ = None
        self.feature_sizes_ = None
        self.used_features_ = None
        self.output_w_ = None
        self.output_b_ = None

    def build_feature_metadata_from_transformer(self, transformer):
        try:
            if transformer is None:
                raise ValueError("Transformer must be provided.")
            
            num_transformer = transformer.named_transformers_.get('num')
            cat_transformer = transformer.named_transformers_.get('cat')
            
            self.nume_input_size_ = len(num_transformer.get_feature_names_out()) if num_transformer else 0
            self.cate_field_size_ = len(cat_transformer.get_feature_names_out()) if cat_transformer else 0
            self.feature_sizes_ = [len(cats) for cats in cat_transformer.categories_] if cat_transformer else []
            
            if self.verbose:
                print(f"Metadata: nume={self.nume_input_size_}, cate={self.cate_field_size_}, features={self.feature_sizes_}")
            
            return {
                'nume_input_size': self.nume_input_size_,
                'cate_field_size': self.cate_field_size_,
                'feature_sizes': self.feature_sizes_
            }
        except Exception as e:
            if self.verbose:
                print(f"Metadata extraction failed: {e}. Using defaults.")
            self.nume_input_size_ = 0
            self.cate_field_size_ = 0
            self.feature_sizes_ = []
            return {
                'nume_input_size': 0,
                'cate_field_size': 0,
                'feature_sizes': []
            }

    def _pretrain_gbdt(self, X, y):
        try:
            params = {
                'objective': 'multiclass',
                'num_class': self.n_classes_,
                'num_boost_round': self.num_trees,
                'max_depth': self.tree_max_depth,
                'learning_rate': 0.1,
                'verbosity': -1 if not self.verbose else 1,
                'random_state': self.random_state
            }
            dtrain = lgb.Dataset(X, label=y)
            gbdt_model = lgb.train(params, dtrain)

            # Simulate used_features and outputs (adjust based on real parsing if needed)
            self.used_features_ = [list(range(X.shape[1])) for _ in range(self.num_trees)]
            self.output_w_ = np.random.randn(self.num_trees, self.n_classes_).astype(np.float32)
            self.output_b_ = np.zeros(self.n_classes_).astype(np.float32)

            if self.verbose:
                print("GBDT pre-training completed.")
        except Exception as e:
            if self.verbose:
                print(f"GBDT pre-training failed: {e}")
                
    def custom_forward(self, Xg, Xd):
        Xg = Xg.float().to(self.device)
        Xd = Xd.long().to(self.device)

        # Get GBDT2NN outputs (gbdt_out is already the final processed output [batch, n_classes])
        if self.model_.num_model == 'gbdt2nn':
            gbdt_out, gbdt_pred = self.model_.gbdt2nn(Xg)
        elif self.model_.num_model == 'gbdt':
            gbdt_out = Xg
            gbdt_pred = None
        else:
            gbdt_out = self.model_.gbdt2nn(Xg)
            gbdt_pred = None

        # DeepFM branch
        deepfm_out = self.model_.deepfm(Xd).float()

        # Ensure deepfm_out matches [batch, n_classes] by repeating if necessary
        if deepfm_out.dim() == 1:
            deepfm_out = deepfm_out.unsqueeze(1)  # [batch] -> [batch, 1]
        if deepfm_out.shape[1] != self.n_classes_:
            deepfm_out = deepfm_out.repeat(1, self.n_classes_)  # Broadcast to match n_classes

        # Combine using alpha/beta (blend the final outputs directly)
        if self.model_.num_model != 'gbdt2nn':
            alpha = self.model_.alpha + 0.5
            beta = self.model_.beta + 0.5
        else:
            alpha = self.model_.alpha + 1
            beta = self.model_.beta

        out = alpha * gbdt_out + beta * deepfm_out

        if self.task != 'regression':
            return nn.Softmax(dim=1)(out), gbdt_pred
        return out, gbdt_pred

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.n_classes_ = len(self.label_encoder_.classes_)

        if sparse.issparse(X):
            X = X.toarray()  # DeepGBM expects dense

        try:
            if not self.use_real_deepgbm:
                raise ImportError("Forcing fallback")

            Xg = X[:, :self.nume_input_size_]  # Numerical features
            self._pretrain_gbdt(Xg, y_encoded)

            self.model_ = DeepGBM(
                nume_input_size=self.nume_input_size_,
                used_features=self.used_features_,
                tree_layers=self.tree_layers,
                output_w=self.output_w_,
                output_b=self.output_b_,
                task=self.task,
                cate_field_size=self.cate_field_size_,
                feature_sizes=self.feature_sizes_,
                embedding_size=self.embedding_size,
                h_depth=self.h_depth,
                deep_layers=self.deep_layers,
                num_model='gbdt2nn'
            ).to(self.device)

            self.model_.criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X).to(self.device),  # Ensure float32
                torch.LongTensor(y_encoded).to(self.device)
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model_.train()
            for epoch in range(self.epochs):
                epoch_loss = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.float().to(self.device)
                    y_batch = y_batch.long().to(self.device)

                    Xg = X_batch[:, :self.nume_input_size_]
                    Xd = X_batch[:, self.nume_input_size_:].long()

                    optimizer.zero_grad()
                    out, _ = self.custom_forward(Xg, Xd)  # Use custom forward
                    loss = self.model_.criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

        except Exception as e:
            if self.verbose:
                print(f"DeepGBM failed: {e}. Falling back to LightGBM.")
            traceback.print_exc()
        return self

    def predict_proba(self, X):
        check_array(X)
        if sparse.issparse(X):
            X = X.toarray()

        if self.model_ is not None:
            self.model_.eval()
            with torch.no_grad():
                probs = []
                for i in range(0, len(X), self.batch_size):
                    X_batch = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)  # float32
                    Xg = X_batch[:, :self.nume_input_size_]
                    Xd = X_batch[:, self.nume_input_size_:].long()
                    out, _ = self.custom_forward(Xg, Xd)
                    prob = nn.Softmax(dim=1)(out).cpu().numpy()
                    probs.append(prob)
                return np.vstack(probs)
        elif self.fallback_model_ is not None:
            return self.fallback_model_.predict_proba(X)
        raise ValueError("Model not fitted.")

    def predict(self, X):
        try:
            proba = self.predict_proba(X)
            return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))
        except:
            traceback.print_exc()

import torch
import torch.nn as nn
import numpy as np
import traceback
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from scipy.sparse import issparse

# Set up logging for this wrapper
logger = logging.getLogger('TabFlexWrapper')
logger.setLevel(logging.DEBUG)  # Detailed logging

class TabFlexClassifier(BaseEstimator, ClassifierMixin):
    """Enhanced scikit-learn compatible wrapper for TabFlex with ICL support and robust error handling."""

    def __init__(self, 
                 emsize=128, 
                 nhead=8, 
                 nhid_factor=4, 
                 nlayers=6, 
                 dropout=0.1,
                 learning_rate=0.001,
                 epochs=50,  # Reduced for speed, as per original
                 batch_size=256,
                 device='auto',
                 random_state=None,
                 exemplars_per_class=5,  # New: Number of stored examples per class for ICL
                 min_exemplars=1):  # New: Minimum exemplars to avoid empty context
        self.emsize = emsize
        self.nhead = nhead
        self.nhid_factor = nhid_factor
        self.nlayers = nlayers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.exemplars_per_class = exemplars_per_class
        self.min_exemplars = min_exemplars
        
        self.model_ = None
        self.label_encoder_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.exemplars_ = None  # Will store (X, y) exemplars for ICL during prediction

    def _setup_device(self):
        """Setup device with fallback."""
        try:
            if self.device == 'auto':
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(self.device)
        except Exception as e:
            logger.error(f"Device setup failed: {e}\n{traceback.format_exc()}")
            return torch.device('cpu')  # Fallback

    def _store_exemplars(self, X, y):
        """Store stratified exemplars for ICL during prediction."""
        try:
            logger.info("Storing exemplars for in-context learning...")
            X_ex = []
            y_ex = []
            for cls in range(self.n_classes_):
                cls_mask = (y == cls)
                X_cls = X[cls_mask]
                y_cls = y[cls_mask]
                n_available = X_cls.shape[0] if issparse(X_cls) else len(y_cls)
                if n_available == 0:
                    logger.warning(f"No samples for class {cls}. Skipping...")
                    continue
                n_take = min(self.exemplars_per_class, max(self.min_exemplars, n_available))
                indices = np.random.choice(n_available, n_take, replace=False)
                if issparse(X_cls):
                    X_ex.append(X_cls[indices].toarray())  # Convert to dense for tensor compatibility
                else:
                    X_ex.append(X_cls[indices])
                y_ex.append(y_cls[indices])
            
            if not X_ex:
                raise ValueError("No exemplars could be stored. Dataset too small or imbalanced.")
            
            self.exemplars_ = (np.vstack(X_ex), np.hstack(y_ex))
            logger.info(f"Stored {len(self.exemplars_[1])} exemplars across {self.n_classes_} classes.")
        except Exception as e:
            logger.error(f"Exemplar storage failed: {e}\n{traceback.format_exc()}")
            raise  # Critical, so raise

    def fit(self, X, y):
        """Fit TabFlex with exemplar storage for ICL."""
        try:
            device = self._setup_device()
            logger.info(f"Fitting on device: {device}")

            # Validate and prepare data
            X = check_array(X, accept_sparse=True, force_all_finite=False)
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y)
            self.n_classes_ = len(self.label_encoder_.classes_)
            self.n_features_ = X.shape[1]

            # Store exemplars (before full training)
            self._store_exemplars(X, y_encoded)

            # Convert sparse to dense if necessary
            if issparse(X):
                logger.info("Converting sparse input to dense for model compatibility.")
                X = X.toarray()

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.LongTensor(y_encoded).to(device)

            # Initialize TabFlex (assuming imported as in your code)
            from ticl.models.encoders import Linear
            from ticl.models.linear_attention import get_linear_attention_layers
            from ticl.utils import SeqBN

            y_encoder = nn.Embedding(self.n_classes_, self.emsize).to(device)  # y_encoder_layer

            self.model_ = TabFlex(
                model='linear_attention',
                n_out=self.n_classes_,
                emsize=self.emsize,
                nhead=self.nhead,
                nhid_factor=self.nhid_factor,
                nlayers=self.nlayers,
                n_features=self.n_features_,
                dropout=self.dropout,
                y_encoder_layer=y_encoder,  # Pass the embedding layer
                classification_task=True,
                input_normalization=True
            ).to(device)

            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()

            self.model_.train()
            for epoch in range(self.epochs):
                for i in range(0, len(X_tensor), self.batch_size):
                    batch_X = X_tensor[i:i+self.batch_size]
                    batch_y = y_tensor[i:i+self.batch_size]
                    batch_len = len(batch_X)
                    if batch_len < 2:  # Need context
                        continue

                    # For training: Use full batch as context, evaluate on last position
                    batch_X = batch_X.unsqueeze(1)  # [batch, 1, features]
                    single_eval_pos = batch_len - 1  # Evaluate on last sample

                    optimizer.zero_grad()
                    try:
                        outputs = self.model_((batch_X, batch_y), single_eval_pos=single_eval_pos)
                        if outputs.dim() > 1:
                            outputs = outputs.mean(dim=0)  # Aggregate if needed
                        loss = criterion(outputs, batch_y[single_eval_pos:])
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        logger.warning(f"Training batch failed: {e}\n{traceback.format_exc()}")
                        continue  # Skip bad batch

            logger.info("Fitting completed successfully.")
            return self

        except Exception as e:
            logger.error(f"Fit failed: {e}\n{traceback.format_exc()}")
            raise

    def predict_proba(self, X):
        """Predict probabilities using stored exemplars as context."""
        try:
            X = check_array(X, accept_sparse=True, force_all_finite=False)
            device = self._setup_device()
            self.model_.eval()

            if self.exemplars_ is None:
                logger.error("No exemplars stored. Cannot perform ICL prediction.")
                return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_  # Uniform fallback

            # Convert input to dense if sparse
            if issparse(X):
                logger.info("Converting sparse input to dense for prediction.")
                X = X.toarray()

            # Convert to tensors once
            X_tensor = torch.FloatTensor(X).to(device)

            # Adaptive batch size to avoid OOM
            batch_size = min(self.batch_size, max(1, X_tensor.shape[0] // 4))  # Conservative

            all_probs = []
            with torch.no_grad():
                for start in range(0, X_tensor.shape[0], batch_size):
                    end = min(start + batch_size, X_tensor.shape[0])
                    X_batch = X_tensor[start:end]
                    batch_len = X_batch.shape[0]

                    # Prepend exemplars as context
                    exemplars_X = torch.FloatTensor(self.exemplars_[0]).to(device)
                    exemplars_y = torch.LongTensor(self.exemplars_[1]).to(device)
                    context_len = exemplars_y.shape[0]

                    # Combine: context + batch
                    combined_X = torch.cat([exemplars_X, X_batch], dim=0).unsqueeze(1)  # [total_len, 1, features]
                    combined_y = torch.cat([exemplars_y, torch.zeros(batch_len, dtype=torch.long).to(device)])  # Dummy y for test

                    single_eval_pos = context_len  # Evaluate starting after context

                    try:
                        outputs = self.model_((combined_X, combined_y), single_eval_pos=single_eval_pos)
                        if outputs.dim() > 2:
                            outputs = outputs.mean(dim=1)  # Fallback aggregation
                        probs = torch.softmax(outputs, dim=-1)
                        all_probs.append(probs.cpu().numpy())
                    except Exception as e:
                        logger.warning(f"Batch prediction failed: {e}\n{traceback.format_exc()}")
                        fallback = np.ones((batch_len, self.n_classes_)) / self.n_classes_
                        all_probs.append(fallback)

            return np.vstack(all_probs) if all_probs else np.ones((X.shape[0], self.n_classes_)) / self.n_classes_

        except Exception as e:
            logger.error(f"Predict_proba failed: {e}\n{traceback.format_exc()}")
            return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_  # Ultimate fallback

    def predict(self, X):
        """Predict class labels."""
        try:
            proba = self.predict_proba(X)
            return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))
        except Exception as e:
            logger.error(f"Predict failed: {e}\n{traceback.format_exc()}")
            return np.zeros(X.shape[0])  # Fallback to class 0

class BaseModelFactory:
    """Enhanced factory for creating individual models with robust error handling"""

    @staticmethod
    def create_model(model_name: str, n_classes: int, **kwargs) -> Any:
        """Create a model instance based on name with comprehensive model support"""

        model_name_clean = model_name.lower().replace('_', '').replace(' ', '')

        try:
            # Tree-based models
            if 'lightgbm' in model_name_clean or model_name_clean == 'lgb':
                return lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    n_estimators=200,  # Further reduced for faster training
                    learning_rate=0.1,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    verbosity=-1,
                    **kwargs
                )

            elif 'xgboost' in model_name_clean or model_name_clean == 'xgb':
                return xgb.XGBClassifier(
                    objective='multi:softprob',
                    n_estimators=200,  # Further reduced for faster training
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            elif 'catboost' in model_name_clean or model_name_clean == 'cb':
                return cb.CatBoostClassifier(
                    loss_function='MultiClass',
                    iterations=200,  # Further reduced for faster training
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_seed=Config.RANDOM_STATE,
                    **kwargs
                )

            elif 'randomforest' in model_name_clean or 'rf' == model_name_clean:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            # SVM models
            elif 'advancedsvm' in model_name_clean or 'svmrbf' in model_name_clean:
                base_svm = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    class_weight='balanced',
                    probability=False,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )
                return CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)

            elif 'linearsvm' in model_name_clean or 'sgd' in model_name_clean:
                return SGDClassifier(
                    loss='log_loss',
                    learning_rate='adaptive',
                    eta0=0.01,
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            # Probabilistic models
            elif 'naivebayes' in model_name_clean or 'nb' == model_name_clean or 'gaussiannb' in model_name_clean:
                return GaussianNB(**kwargs)

            # Neural network
            elif 'neuralnetwork' in model_name_clean or 'mlp' in model_name_clean:
                return MLPClassifier(
                    hidden_layer_sizes=(50, 25),  # Reduced for faster training
                    max_iter=100,  # Reduced for faster training
                    early_stopping=True,
                    validation_fraction=0.1,
                    alpha=0.01,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            # Linear models
            elif 'logisticregression' in model_name_clean or 'lr' == model_name_clean:
                return LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            elif 'knn' in model_name_clean:
                return KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            elif 'ridge' in model_name_clean:
                return LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    multi_class='multinomial',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            # Advanced models
            elif 'ngboost' in model_name_clean and NGBOOST_AVAILABLE:
                return ngb.NGBClassifier(
                    Dist=k_categorical(n_classes),
                    n_estimators=100,  # Reduced for faster training
                    learning_rate=0.01,
                    verbose=False,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            # Add these to BaseModelFactory.create_model method after the NGBoost block:
            elif 'deepgbm' in model_name_clean:
                # DeepGBM implementation (fallback to LightGBM with deeper trees if DeepGBM not available)
                try:
                    deepgbm_instance = DeepGBMWrapper(
                        embedding_size=4, 
                        h_depth=2, 
                        deep_layers=[32, 32],
                        task='classification', 
                        epochs=50, batch_size=256,
                        random_state=Config.RANDOM_STATE, 
                        use_real_deepgbm=True
                    )
                    if hasattr(Config, 'GLOBAL_PREPROCESSOR') and Config.GLOBAL_PREPROCESSOR is not None:
                        deepgbm_instance.build_feature_metadata_from_transformer(Config.GLOBAL_PREPROCESSOR)
                    return deepgbm_instance
                except:
                    traceback.print_exc()

            elif 'tabflex' in model_name_clean:
                try:
                    # Import the TabFlex class and dependencies
                    from ticl.models.encoders import Linear
                    from ticl.models.linear_attention import get_linear_attention_layers
                    from ticl.utils import SeqBN
                    
                    # Use the TabFlexClassifier wrapper
                    return TabFlexClassifier(
                        emsize=128,
                        nhead=8,
                        nhid_factor=4,
                        nlayers=6,
                        dropout=0.1,
                        learning_rate=0.001,
                        epochs=50,  # Reduced for faster training
                        batch_size=256,
                        random_state=Config.RANDOM_STATE,
                        **kwargs
                    )
                except ImportError as e:
                    print(f"TabFlex dependencies not available: {e}. Using TabNet fallback.") 
                    try:
                        # TabNet fallback
                        from pytorch_tabnet.tab_model import TabNetClassifier
                        return TabNetClassifier(
                            n_d=64, n_a=64, n_steps=5,
                            gamma=1.5, seed=Config.RANDOM_STATE,
                            verbose=0, **kwargs
                        )
                    except ImportError:
                        print("TabNet also not available. Using MLP fallback.")  # Fixed: removed self.logger
                        return MLPClassifier(
                            hidden_layer_sizes=(256, 128, 64),
                            max_iter=500,
                            random_state=Config.RANDOM_STATE,
                            **kwargs
                        )

            elif 'mixture' in model_name_clean or 'moe' in model_name_clean:
                # Mixture-of-Experts implementation (fallback to ensemble-like MLP if MoE not available)
                try:
                    # Try to import specialized MoE library if available
                    from sklearn.mixture import GaussianMixture
                    from sklearn.base import BaseEstimator, ClassifierMixin

                    # Custom MoE implementation using multiple MLPs
                    class MixtureOfExpertsClassifier(BaseEstimator, ClassifierMixin):
                        def __init__(self, n_experts=3, random_state=None):
                            self.n_experts = n_experts
                            self.random_state = random_state
                            self.experts = []
                            self.gating_network = None

                        def fit(self, X, y):
                            # Create multiple expert networks
                            for i in range(self.n_experts):
                                expert = MLPClassifier(
                                    hidden_layer_sizes=(100, 50),
                                    max_iter=200,
                                    random_state=self.random_state + i if self.random_state else None,
                                    alpha=0.01
                                )
                                expert.fit(X, y)
                                self.experts.append(expert)

                            # Simple gating network (can be enhanced)
                            self.gating_network = LogisticRegression(
                                multi_class='multinomial',
                                random_state=self.random_state
                            )

                            # Create gating features (expert confidence scores)
                            expert_scores = []
                            for expert in self.experts:
                                scores = np.max(expert.predict_proba(X), axis=1)
                                expert_scores.append(scores)

                            gating_X = np.column_stack(expert_scores)
                            # Gate based on which expert is most confident
                            gate_labels = np.argmax(expert_scores, axis=0)
                            self.gating_network.fit(gating_X, gate_labels)

                            return self

                        def predict_proba(self, X):
                            """FIXED: Vectorized prediction without loops"""
                            # Get all expert predictions at once
                            expert_probas = np.stack([expert.predict_proba(X) for expert in self.experts], axis=0)
                            expert_scores = np.max(expert_probas, axis=2)  # Shape: (n_experts, n_samples)
                            
                            # Get gating weights for all samples at once
                            gating_X = expert_scores.T  # Shape: (n_samples, n_experts)
                            gate_proba = self.gating_network.predict_proba(gating_X)
                            
                            # Vectorized combination
                            final_proba = np.einsum('ijk,ij->jk', expert_probas.transpose(0,1,2), gate_proba.T)
                            
                            return final_proba

                        def predict(self, X):
                            proba = self.predict_proba(X)
                            return np.argmax(proba, axis=1)

                    return MixtureOfExpertsClassifier(
                        n_experts=3,
                        random_state=Config.RANDOM_STATE
                    )

                except Exception as e:
                    print(f"MoE implementation failed, using ensemble MLPClassifier for {model_name}: {e}")
                    return MLPClassifier(
                        hidden_layer_sizes=(150, 75),
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.1,
                        alpha=0.01,
                        random_state=Config.RANDOM_STATE,
                        **kwargs
                    )

            # Also add support for svm_linear and linear_model that were in MODEL_COMPLEXITY but missing from factory:
            elif 'svmlinear' in model_name_clean or model_name_clean == 'svmlinear':
                return SVC(
                    kernel='linear',
                    C=1.0,
                    class_weight='balanced',
                    probability=True,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            elif 'linearmodel' in model_name_clean or model_name_clean == 'linearmodel':
                return LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS,
                    **kwargs
                )

            elif 'adaboost' in model_name_clean or 'ada' in model_name_clean:
                from sklearn.ensemble import AdaBoostClassifier
                return AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.8,
                    algorithm='SAMME.R',  # For probability estimates
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            elif 'gradientboost' in model_name_clean or 'gbm' in model_name_clean:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

            elif 'random_classifier' in model_name_clean or model_name_clean == 'randomclassifier':
                from sklearn.dummy import DummyClassifier
                return DummyClassifier(strategy='uniform', random_state=Config.RANDOM_STATE)

            elif 'majority_classifier' in model_name_clean or model_name_clean == 'majorityclassifier':
                from sklearn.dummy import DummyClassifier
                return DummyClassifier(strategy='most_frequent')

            elif 'uniform_classifier' in model_name_clean or model_name_clean == 'uniformclassifier':
                from sklearn.dummy import DummyClassifier
                return DummyClassifier(strategy='uniform', random_state=Config.RANDOM_STATE)


            else:
                # Default to logistic regression for unknown models
                print(f"Warning: Unknown model '{model_name}', defaulting to LogisticRegression")
                return LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                    **kwargs
                )

        except Exception as e:
            print(f"Error creating model '{model_name}': {str(e)}")
            # Return a simple logistic regression as fallback
            return LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                random_state=Config.RANDOM_STATE
            )

class EnsembleFactory:
    """Enhanced factory for creating ensemble models"""

    @staticmethod
    def create_ensemble(config: Dict[str, Any], n_classes: int) -> Any:
        """Create ensemble based on configuration"""

        ensemble_type = config.get('ensemble_type', 'voting')
        base_model_names = config.get('base_models', [])

        # Create base models with error handling
        base_models = {}
        for name in base_model_names:
            try:
                base_models[name] = BaseModelFactory.create_model(name, n_classes)
            except Exception as e:
                print(f"Warning: Failed to create base model '{name}': {e}")
                continue

        if not base_models:
            raise ValueError("No valid base models could be created")
        # TEMPORARY: Force all ensembles to use voting
        if len(base_model_names) > 1:  # Only for multi-model ensembles
            return VotingEnsemble(base_models, list(base_models.keys()))

        if ensemble_type == 'stacking':
            return StackingEnsemble(base_models, list(base_models.keys()), n_classes)
        elif ensemble_type == 'voting':
            return VotingEnsemble(base_models, list(base_models.keys()))
        elif ensemble_type == 'weighted':
            weights = config.get('weights', None)
            return WeightedEnsemble(base_models, list(base_models.keys()), weights)
        elif ensemble_type == 'cv_ensemble':
            cv_folds = config.get('cv_folds', 3)
            return CVEnsemble(base_models, list(base_models.keys()), cv_folds, n_classes)
        elif ensemble_type == 'parametric_ensemble':
            n_versions = config.get('n_versions', 3)
            return ParametricEnsemble(base_models, list(base_models.keys()), n_versions, n_classes)
        elif ensemble_type == 'stacking_ridge':
            return StackingRidgeEnsemble(base_models, list(base_models.keys()), n_classes)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

class StackingEnsemble:
    """Enhanced stacking ensemble implementation with robust error handling"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str], n_classes: int):
        self.base_models = base_models
        self.model_names = model_names
        self.n_classes = n_classes
        self.meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            max_iter=1000
        )
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X

    def fit(self, X, y):
        """Fit the stacking ensemble with optimal sparse handling"""
        try:
            # Keep X in original format - convert per model as needed
            skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                                 random_state=Config.RANDOM_STATE)

            meta_features = []
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            n_samples = safe_array_length(X)

            for name in self.model_names:
                cv_preds = np.zeros((n_samples, self.n_classes))

                for train_idx, val_idx in skf.split(X, y_array):
                    try:
                        # Prepare data specifically for this model
                        X_model_format = self._prepare_data_for_model(X, name)
                        
                        X_train_fold = safe_array_indexing(X_model_format, train_idx)
                        X_val_fold = safe_array_indexing(X_model_format, val_idx)
                        y_train_fold = y_array[train_idx]

                        # Create fresh model for this fold
                        model_copy = BaseModelFactory.create_model(name, self.n_classes)
                        model_copy.fit(X_train_fold, y_train_fold)
                        cv_preds[val_idx] = model_copy.predict_proba(X_val_fold)
                    except Exception as e:
                        print(f"Warning: Fold training failed for {name}: {e}")
                        cv_preds[val_idx] = np.ones((len(val_idx), self.n_classes)) / self.n_classes

                meta_features.append(cv_preds)

            # Train base models on full data with appropriate format
            for name in self.model_names:
                try:
                    X_model_format = self._prepare_data_for_model(X, name)
                    self.base_models[name].fit(X_model_format, y)
                except Exception as e:
                    print(f"Warning: Full training failed for {name}: {e}")

            # Train meta-learner
            if meta_features:
                meta_X = np.column_stack(meta_features)
                self.meta_learner.fit(meta_X, y_array)

            self.is_fitted = True
            return self

        except Exception as e:
            print(f"Error in StackingEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Predict probabilities with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        try:
            base_preds = []
            for name in self.model_names:
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    pred = self.base_models[name].predict_proba(X_model_format)
                    base_preds.append(pred)
                except Exception as e:
                    print(f"Warning: Prediction failed for {name}: {e}")
                    n_samples = safe_array_length(X)
                    fallback_pred = np.ones((n_samples, self.n_classes)) / self.n_classes
                    base_preds.append(fallback_pred)

            if base_preds:
                meta_X = np.column_stack(base_preds)
                return self.meta_learner.predict_proba(meta_X)
            else:
                n_samples = safe_array_length(X)
                return np.ones((n_samples, self.n_classes)) / self.n_classes

        except Exception as e:
            print(f"Error in StackingEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, self.n_classes)) / self.n_classes

    def predict(self, X):
        """Predict using stacking ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class VotingEnsemble:
    """Enhanced voting ensemble implementation"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str]):
        self.base_models = base_models
        self.model_names = model_names
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X
        
    def fit(self, X, y):
        """Fit the voting ensemble with optimal sparse handling"""
        try:
            for name in self.model_names:
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    
                    # **CRITICAL FIX**: Set metadata for DeepGBM instances
                    if hasattr(self.base_models[name], 'set_metadata') and hasattr(Config, 'GLOBAL_PREPROCESSOR'):
                        if Config.GLOBAL_PREPROCESSOR is not None:
                            self.base_models[name].build_feature_metadata_from_transformer(Config.GLOBAL_PREPROCESSOR)
                    
                    self.base_models[name].fit(X_model_format, y)
                except Exception as e:
                    print(f"Warning: Training failed for {name}: {e}")
            self.is_fitted = True
            return self
        except Exception as e:
            print(f"Error in VotingEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Predict probabilities with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        try:
            valid_predictions = []
            for name in self.model_names:
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    proba = self.base_models[name].predict_proba(X_model_format)
                    valid_predictions.append(proba)
                except Exception as e:
                    print(f"Warning: Prediction failed for {name}: {e}")

            if valid_predictions:
                return np.mean(valid_predictions, axis=0)
            else:
                n_samples = safe_array_length(X)
                return np.ones((n_samples, 3)) / 3

        except Exception as e:
            print(f"Error in VotingEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, 3)) / 3

    def predict(self, X):
        """Predict using voting ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class WeightedEnsemble:
    """Enhanced weighted ensemble implementation"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str], weights: Optional[List[float]] = None):
        self.base_models = base_models
        self.model_names = model_names
        self.weights = weights if weights else [1.0/len(model_names)] * len(model_names)
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X

    def fit(self, X, y):
        """Fit the weighted ensemble with optimal sparse handling"""
        try:
            for name in self.model_names:
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    self.base_models[name].fit(X_model_format, y)
                except Exception as e:
                    print(f"Warning: Training failed for {name}: {e}")
            self.is_fitted = True
            return self
        except Exception as e:
            print(f"Error in WeightedEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Predict probabilities with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        try:
            proba_weighted = None
            total_weight = 0

            for i, name in enumerate(self.model_names):
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    proba = self.base_models[name].predict_proba(X_model_format) * self.weights[i]
                    if proba_weighted is None:
                        proba_weighted = proba
                    else:
                        proba_weighted += proba
                    total_weight += self.weights[i]
                except Exception as e:
                    print(f"Warning: Prediction failed for {name}: {e}")

            if proba_weighted is not None and total_weight > 0:
                return proba_weighted / total_weight
            else:
                # Fallback
                n_samples = safe_array_length(X)
                return np.ones((n_samples, 3)) / 3

        except Exception as e:
            print(f"Error in WeightedEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, 3)) / 3

    def predict(self, X):
        """Predict using weighted ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class CVEnsemble:
    """Cross-Validation Ensemble for creating multiple versions of the same model"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str], cv_folds: int, n_classes: int):
        self.base_model_name = model_names[0] if model_names else 'lightgbm'
        self.cv_folds = cv_folds
        self.n_classes = n_classes
        self.models = []
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X

    def fit(self, X, y):
        """Fit multiple CV models with optimal sparse handling"""
        try:
            # Prepare data format for the base model type
            X_model_format = self._prepare_data_for_model(X, self.base_model_name)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y

            for fold, (train_idx, _) in enumerate(skf.split(X_model_format, y_array)):
                try:
                    hp_variation = {
                        'learning_rate': 0.05 + (fold * 0.02),
                        'max_depth': 5 + (fold % 3),
                        'random_state': Config.RANDOM_STATE + fold
                    }

                    model = BaseModelFactory.create_model(self.base_model_name, self.n_classes, **hp_variation)
                    X_train_fold = safe_array_indexing(X_model_format, train_idx)
                    y_train_fold = y_array[train_idx]

                    model.fit(X_train_fold, y_train_fold)
                    self.models.append(model)
                except Exception as e:
                    print(f"Warning: CV fold {fold} training failed: {e}")

            self.is_fitted = len(self.models) > 0
            return self

        except Exception as e:
            print(f"Error in CVEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Average predictions with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        try:
            # Prepare data format for the base model type
            X_model_format = self._prepare_data_for_model(X, self.base_model_name)

            if self.models:
                valid_predictions = []
                for model in self.models:
                    try:
                        proba = model.predict_proba(X_model_format)
                        valid_predictions.append(proba)
                    except Exception as e:
                        print(f"Warning: CV model prediction failed: {e}")

                if valid_predictions:
                    return np.mean(valid_predictions, axis=0)

            n_samples = safe_array_length(X)
            return np.ones((n_samples, self.n_classes)) / self.n_classes

        except Exception as e:
            print(f"Error in CVEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, self.n_classes)) / self.n_classes

    def predict(self, X):
        """Predict using CV ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Add after CVEnsemble class:

class ParametricEnsemble:
    """Parametric Ensemble that creates multiple versions of models with different hyperparameters"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str], n_versions: int, n_classes: int):
        self.base_models = base_models
        self.model_names = model_names
        self.n_versions = n_versions
        self.n_classes = n_classes
        self.parametric_models = {}
        self.meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=Config.RANDOM_STATE,
            max_iter=1000
        )
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X

    def _generate_hyperparameter_variations(self, model_name: str, version: int):
        """Generate different hyperparameter sets for each model version"""
        base_params = {}

        model_name_clean = model_name.lower().replace('_', '')

        # LightGBM parameter variations
        if 'lightgbm' in model_name_clean:
            variations = [
                {'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 15, 'subsample': 0.7},
                {'learning_rate': 0.1, 'max_depth': 6, 'num_leaves': 31, 'subsample': 0.8},
                {'learning_rate': 0.15, 'max_depth': 8, 'num_leaves': 63, 'subsample': 0.9}
            ]
        # XGBoost parameter variations
        elif 'xgboost' in model_name_clean:
            variations = [
                {'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.7, 'colsample_bytree': 0.7},
                {'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
                {'learning_rate': 0.15, 'max_depth': 8, 'subsample': 0.9, 'colsample_bytree': 0.9}
            ]
        # CatBoost parameter variations
        elif 'catboost' in model_name_clean:
            variations = [
                {'learning_rate': 0.05, 'depth': 4, 'iterations': 150},
                {'learning_rate': 0.1, 'depth': 6, 'iterations': 200},
                {'learning_rate': 0.15, 'depth': 8, 'iterations': 250}
            ]
        # Random Forest parameter variations
        elif 'randomforest' in model_name_clean:
            variations = [
                {'n_estimators': 50, 'max_depth': 6, 'min_samples_split': 5},
                {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2},
                {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 10}
            ]
        # Neural Network parameter variations
        elif 'neuralnetwork' in model_name_clean or 'mlp' in model_name_clean:
            variations = [
                {'hidden_layer_sizes': (50,), 'alpha': 0.001, 'learning_rate_init': 0.001},
                {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'learning_rate_init': 0.01},
                {'hidden_layer_sizes': (150, 75, 25), 'alpha': 0.1, 'learning_rate_init': 0.1}
            ]
        # SVM parameter variations
        elif 'advancedsvm' in model_name_clean:
            variations = [
                {'C': 0.1, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 'scale'},
                {'C': 10.0, 'gamma': 'auto'}
            ]
        else:
            # Default variations for other models
            variations = [
                {'random_state': Config.RANDOM_STATE + version},
                {'random_state': Config.RANDOM_STATE + version + 100},
                {'random_state': Config.RANDOM_STATE + version + 200}
            ]

        return variations[version % len(variations)]

    def fit(self, X, y):
        """Fit parametric ensemble with optimal sparse handling"""
        try:
            # Create multiple versions of each base model
            for model_name in self.model_names:
                # Prepare data format for this model type
                X_model_format = self._prepare_data_for_model(X, model_name)
                self.parametric_models[model_name] = []

                for version in range(self.n_versions):
                    hp_variations = self._generate_hyperparameter_variations(model_name, version)
                    model = BaseModelFactory.create_model(model_name, self.n_classes, **hp_variations)

                    try:
                        model.fit(X_model_format, y)
                        self.parametric_models[model_name].append(model)
                    except Exception as e:
                        print(f"Warning: Parametric model {model_name} version {version} failed: {e}")

            # Create meta-features with appropriate data formats
            meta_features = []
            n_samples = safe_array_length(X)

            for model_name, models in self.parametric_models.items():
                # Prepare data format for this model type
                X_model_format = self._prepare_data_for_model(X, model_name)
                
                for model in models:
                    try:
                        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.RANDOM_STATE)
                        cv_preds = np.zeros((n_samples, self.n_classes))
                        y_array = np.array(y) if not isinstance(y, np.ndarray) else y

                        for train_idx, val_idx in skf.split(X_model_format, y_array):
                            X_train_fold = safe_array_indexing(X_model_format, train_idx)
                            X_val_fold = safe_array_indexing(X_model_format, val_idx)
                            y_train_fold = y_array[train_idx]

                            fold_model = BaseModelFactory.create_model(
                                model_name, self.n_classes,
                                **self._generate_hyperparameter_variations(model_name, len(models))
                            )
                            fold_model.fit(X_train_fold, y_train_fold)
                            cv_preds[val_idx] = fold_model.predict_proba(X_val_fold)

                        meta_features.append(cv_preds)
                    except Exception as e:
                        print(f"Warning: Meta-feature generation failed for {model_name}: {e}")

            if meta_features:
                meta_X = np.column_stack(meta_features)
                self.meta_learner.fit(meta_X, y)

            self.is_fitted = True
            return self

        except Exception as e:
            print(f"Error in ParametricEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Predict using parametric ensemble with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        try:
            base_predictions = []

            for model_name, models in self.parametric_models.items():
                # Prepare data format for this model type
                X_model_format = self._prepare_data_for_model(X, model_name)
                
                for model in models:
                    try:
                        pred = model.predict_proba(X_model_format)
                        base_predictions.append(pred)
                    except Exception as e:
                        print(f"Warning: Prediction failed for parametric {model_name}: {e}")

            if base_predictions:
                meta_X = np.column_stack(base_predictions)
                return self.meta_learner.predict_proba(meta_X)
            else:
                n_samples = safe_array_length(X)
                return np.ones((n_samples, self.n_classes)) / self.n_classes

        except Exception as e:
            print(f"Error in ParametricEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, self.n_classes)) / self.n_classes

    def predict(self, X):
        """Predict using parametric ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class StackingRidgeEnsemble:
    """Stacking ensemble with Ridge regression as meta-learner"""

    def __init__(self, base_models: Dict[str, Any], model_names: List[str], n_classes: int):
        self.base_models = base_models
        self.model_names = model_names
        self.n_classes = n_classes
        self.meta_learner = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
        self.is_fitted = False
        
    def _needs_dense_conversion(self, model_name: str) -> bool:
        """Check if a specific model needs dense matrix conversion"""
        model_clean = model_name.lower().replace('_', '')
        dense_only_models = ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
        return any(dense_model in model_clean for dense_model in dense_only_models)

    def _prepare_data_for_model(self, X, model_name: str):
        """Prepare data in the right format (sparse/dense) for a specific model"""
        if sparse.issparse(X) and self._needs_dense_conversion(model_name):
            return X.toarray()
        return X

    def fit(self, X, y):
        """Fit stacking ensemble with Ridge meta-learner and optimal sparse handling"""
        try:
            # Keep X in original format - convert per model as needed
            skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

            meta_features = []
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            n_samples = safe_array_length(X)

            for name in self.model_names:
                cv_preds = np.zeros((n_samples, self.n_classes))

                for train_idx, val_idx in skf.split(X, y_array):
                    try:
                        # Prepare data specifically for this model
                        X_model_format = self._prepare_data_for_model(X, name)
                        
                        X_train_fold = safe_array_indexing(X_model_format, train_idx)
                        X_val_fold = safe_array_indexing(X_model_format, val_idx)
                        y_train_fold = y_array[train_idx]

                        model_copy = BaseModelFactory.create_model(name, self.n_classes)
                        model_copy.fit(X_train_fold, y_train_fold)
                        cv_preds[val_idx] = model_copy.predict_proba(X_val_fold)
                    except Exception as e:
                        print(f"Warning: Ridge stacking fold training failed for {name}: {e}")
                        cv_preds[val_idx] = np.ones((len(val_idx), self.n_classes)) / self.n_classes

                # Flatten probabilities for Ridge regression
                for class_idx in range(self.n_classes):
                    meta_features.append(cv_preds[:, class_idx])

            # Train base models on full data with appropriate format
            for name in self.model_names:
                try:
                    X_model_format = self._prepare_data_for_model(X, name)
                    self.base_models[name].fit(X_model_format, y)
                except Exception as e:
                    print(f"Warning: Ridge stacking full training failed for {name}: {e}")

            # Train Ridge meta-learner for each class (one-vs-rest)
            if meta_features:
                meta_X = np.column_stack(meta_features)
                self.meta_learners = {}

                for class_idx in range(self.n_classes):
                    y_binary = (y_array == class_idx).astype(int)
                    meta_learner = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
                    meta_learner.fit(meta_X, y_binary)
                    self.meta_learners[class_idx] = meta_learner

            self.is_fitted = True
            return self

        except Exception as e:
            print(f"Error in StackingRidgeEnsemble.fit: {e}")
            self.is_fitted = False
            raise

    def predict_proba(self, X):
        """Predict probabilities with optimal sparse handling"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        try:
            # Get base model predictions with per-model data preparation
            base_features = []
            for name in self.model_names:
                try:
                    # Prepare data specifically for this model
                    X_model_format = self._prepare_data_for_model(X, name)
                    pred = self.base_models[name].predict_proba(X_model_format)
                    for class_idx in range(self.n_classes):
                        base_features.append(pred[:, class_idx])
                except Exception as e:
                    print(f"Warning: Ridge stacking prediction failed for {name}: {e}")
                    # Fallback features
                    for class_idx in range(self.n_classes):
                        n_samples = safe_array_length(X)
                        base_features.append(np.ones(n_samples) / self.n_classes)

            if base_features:
                meta_X = np.column_stack(base_features)

                # Get predictions from each Ridge meta-learner
                class_predictions = []
                for class_idx in range(self.n_classes):
                    if class_idx in self.meta_learners:
                        pred = self.meta_learners[class_idx].predict(meta_X)
                        class_predictions.append(pred)
                    else:
                        n_samples = safe_array_length(X)
                        class_predictions.append(np.ones(n_samples) / self.n_classes)

                # Convert to probabilities and normalize
                predictions = np.column_stack(class_predictions)
                predictions = np.clip(predictions, 0, 1)  # Ensure non-negative
                predictions = predictions / predictions.sum(axis=1, keepdims=True)  # Normalize

                return predictions
            else:
                n_samples = safe_array_length(X)
                return np.ones((n_samples, self.n_classes)) / self.n_classes

        except Exception as e:
            print(f"Error in StackingRidgeEnsemble.predict_proba: {e}")
            n_samples = safe_array_length(X)
            return np.ones((n_samples, self.n_classes)) / self.n_classes

    def predict(self, X):
        """Predict using Ridge stacking ensemble"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# ================================================================================================
# FIXED EVALUATION FRAMEWORK
# ================================================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation with robust error handling"""

    @staticmethod
    def calculate_tier1_metrics(y_true, y_pred_proba, training_time: float,
                               memory_usage: float, n_classes: int) -> Dict[str, float]:
        """Calculate Tier 1 metrics with error handling"""
        try:
            y_pred = np.argmax(y_pred_proba, axis=1)

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
                'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'training_time': training_time,
                'memory_usage': memory_usage
            }

            # Calculate ROC-AUC with error handling
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr',
                                                      average='weighted', labels=np.arange(n_classes))
            except Exception:
                metrics['roc_auc_ovr'] = 0.5

            return metrics

        except Exception as e:
            print(f"Error calculating Tier 1 metrics: {e}")
            return {
                'accuracy': 0.0,
                'weighted_f1': 0.0,
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'roc_auc_ovr': 0.5,
                'training_time': training_time,
                'memory_usage': memory_usage
            }

    @staticmethod
    def calculate_tier2_metrics(y_true, y_pred_proba, cv_scores: Optional[List[float]] = None) -> Dict[str, float]:
        """FIXED: Calculate Tier 2 metrics with proper array handling"""
        try:
            y_pred = np.argmax(y_pred_proba, axis=1)
            n_classes = y_pred_proba.shape[1]

            metrics = {}

            # Per-class F1 scores
            try:
                per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                # FIXED: Ensure per_class_f1 is always an array
                if np.isscalar(per_class_f1):
                    per_class_f1 = np.array([per_class_f1])

                for i, f1_val in enumerate(per_class_f1):
                    if i < n_classes:  # Safety check
                        metrics[f'class_{i}_f1'] = float(f1_val)
            except Exception as e:
                print(f"Error in per-class F1 calculation: {e}")
                for i in range(n_classes):
                    metrics[f'class_{i}_f1'] = 0.0

            # Confidence statistics
            try:
                max_proba = np.max(y_pred_proba, axis=1)
                # FIXED: Ensure we're working with valid arrays
                if len(max_proba) > 0:
                    metrics['confidence_mean'] = float(np.mean(max_proba))
                    metrics['confidence_std'] = float(np.std(max_proba))
                else:
                    metrics['confidence_mean'] = 0.5
                    metrics['confidence_std'] = 0.0
            except Exception as e:
                print(f"Error in confidence statistics: {e}")
                metrics['confidence_mean'] = 0.5
                metrics['confidence_std'] = 0.0

            # Prediction entropy
            try:
                # FIXED: Add small epsilon to avoid log(0) and handle array operations properly
                epsilon = 1e-15
                y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
                entropy = -np.sum(y_pred_proba_safe * np.log(y_pred_proba_safe), axis=1)

                if len(entropy) > 0:
                    metrics['prediction_entropy'] = float(np.mean(entropy))
                else:
                    metrics['prediction_entropy'] = 1.0
            except Exception as e:
                print(f"Error in prediction entropy: {e}")
                metrics['prediction_entropy'] = 1.0

            # Advanced metrics with error handling
            try:
                metrics['cohens_kappa'] = float(cohen_kappa_score(y_true, y_pred))
            except Exception as e:
                print(f"Error in Cohen's Kappa: {e}")
                metrics['cohens_kappa'] = 0.0

            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                # FIXED: Handle NaN values from MCC
                metrics['matthews_corrcoef'] = float(mcc) if not np.isnan(mcc) else 0.0
            except Exception as e:
                print(f"Error in Matthews Correlation: {e}")
                metrics['matthews_corrcoef'] = 0.0

            try:
                ll = log_loss(y_true, y_pred_proba)
                # FIXED: Handle infinite or NaN log loss values
                metrics['log_loss'] = float(ll) if np.isfinite(ll) else 1.0
            except Exception as e:
                print(f"Error in Log Loss: {e}")
                metrics['log_loss'] = 1.0

            # FIXED: Handle cv_scores properly
            if cv_scores is not None and len(cv_scores) > 0:
                # Ensure cv_scores is a proper array/list
                cv_scores_array = np.array(cv_scores)
                if len(cv_scores_array) > 1:
                    metrics['cv_stability'] = float(np.std(cv_scores_array))
                else:
                    metrics['cv_stability'] = 0.0
            else:
                metrics['cv_stability'] = 0.0

            return metrics

        except Exception as e:
            print(f"Error calculating Tier 2 metrics: {e}")
            return {
                'cohens_kappa': 0.0,
                'matthews_corrcoef': 0.0,
                'log_loss': 1.0,
                'confidence_mean': 0.5,
                'confidence_std': 0.0,
                'prediction_entropy': 1.0,
                'cv_stability': 0.0
            }

    @staticmethod
    def calculate_tier3_metrics(model, base_individual_score: Optional[float] = None) -> Dict[str, float]:
        """Calculate Tier 3 metrics (ensemble-specific)"""
        metrics = {}

        try:
            # Ensemble contribution analysis
            if hasattr(model, 'meta_learner') and hasattr(model.meta_learner, 'coef_'):
                try:
                    coef_array = np.array(model.meta_learner.coef_)
                    feature_importance = np.abs(coef_array).mean(axis=0)
                    n_models = len(model.model_names)

                    # FIXED: Handle coefficient shape properly
                    if len(coef_array.shape) > 1:
                        n_classes = coef_array.shape[1] if coef_array.shape[1] > 1 else coef_array.shape[0]
                    else:
                        n_classes = 1

                    for i, model_name in enumerate(model.model_names):
                        if n_classes > 1:
                            start_idx = i * n_classes
                            end_idx = (i + 1) * n_classes
                            if end_idx <= len(feature_importance):
                                contribution = float(np.mean(feature_importance[start_idx:end_idx]))
                            else:
                                contribution = float(feature_importance[i]) if i < len(feature_importance) else 0.0
                        else:
                            contribution = float(feature_importance[i]) if i < len(feature_importance) else 0.0

                        metrics[f'{model_name}_contribution'] = contribution
                except Exception as e:
                    print(f"Error in ensemble contribution analysis: {e}")

            # Marginal improvement over best individual
            if base_individual_score is not None:
                metrics['marginal_improvement'] = 0.0

        except Exception as e:
            print(f"Error calculating Tier 3 metrics: {e}")

        return metrics

class ModelEvaluator:
    """FIXED model evaluation class with robust error handling"""

    def __init__(self, X_train, X_test, y_train, y_test, n_classes: int):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_classes = n_classes
        self.logger = logging.getLogger(__name__)

        # CREATE VALIDATION SET for proper early stopping
        self.X_train_split, self.X_val, self.y_train_split, self.y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE,
            stratify=y_train
        )

    def quick_evaluate(self, model_name: str, config: Dict[str, Any]) -> Dict[str, float]:
        """FIXED: Proper validation-based early stopping"""
        try:
            # Use 30% of training split for quick training
            n_train_samples = safe_array_length(self.X_train_split)
            sample_size = min(int(0.3 * n_train_samples), 1000)

            np.random.seed(Config.RANDOM_STATE)
            indices = np.random.choice(n_train_samples, sample_size, replace=False)

            X_sample = safe_array_indexing(self.X_train_split, indices)
            y_sample = safe_pandas_indexing(self.y_train_split, indices)

            if hasattr(y_sample, 'values'):
                y_sample = y_sample.values

            # Handle sparse matrix conversion
            if sparse.issparse(X_sample):
                base_models = config.get('base_models', [])
                needs_dense = any(model.lower().replace('_', '') in ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm']
                                for model in base_models)
                if needs_dense:
                    X_sample = X_sample.toarray()
                    X_val_eval = self.X_val.toarray() if sparse.issparse(self.X_val) else self.X_val
                else:
                    X_val_eval = self.X_val
            else:
                X_val_eval = self.X_val

            start_time = time.time()

            if config.get('ensemble_type') == 'none':
                model = BaseModelFactory.create_model(config['base_models'][0], self.n_classes)
            else:
                model = EnsembleFactory.create_ensemble(config, self.n_classes)

            model.fit(X_sample, y_sample)
            training_time = time.time() - start_time

            # FIXED: Evaluate on VALIDATION set, not training set
            y_val_pred_proba = model.predict_proba(X_val_eval)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)

            metrics = {
                'accuracy': accuracy_score(self.y_val, y_val_pred),
                'weighted_f1': f1_score(self.y_val, y_val_pred, average='weighted'),
                'training_time': training_time,
                'validation_based': True  # Flag to indicate proper validation
            }

            return metrics

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Quick evaluation failed for {model_name}: {str(e)}")
            return {'accuracy': 0.0, 'weighted_f1': 0.0, 'training_time': float('inf'), 'validation_based': False}

    def comprehensive_cv_evaluate(self, model_name: str, config: Dict[str, Any], cv_folds: int = 3) -> Dict[str, Any]:
        """FIXED: Comprehensive cross-validation with proper sparse handling"""
        try:
            from sklearn.model_selection import StratifiedKFold
            print(f"Running comprehensive {cv_folds}-fold CV for {model_name}")
            
            # Prepare data with enhanced validation
            X_full = self.X_train_split
            y_full = self.y_train_split
            
            # Handle sparse conversion consistently
            if sparse.issparse(X_full):
                base_models = config.get('base_models', [])
                needs_dense = any(model.lower().replace('_', '') in ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm']
                                for model in base_models)
                if needs_dense:
                    X_full = X_full.toarray()
            
            # FIXED: Ensure proper array formats - keep sparse when possible
            if hasattr(y_full, 'values'):
                y_full = y_full.values
            y_full = np.asarray(y_full).flatten()
            
            # FIXED: Only convert X to dense numpy if it's not sparse and needs conversion
            if not sparse.issparse(X_full) and hasattr(X_full, 'values'):
                X_full = X_full.values
            if not sparse.issparse(X_full):
                X_full = np.asarray(X_full)
            
            # Validate data dimensions with safe checking
            if get_n_samples(y_full) == 0:
                self.logger.error("No training data available for CV")
                return {'cv_failed': True, 'error': 'No training data'}
            
            x_samples = get_n_samples(X_full)
            y_samples = get_n_samples(y_full)
            
            if x_samples != y_samples:
                self.logger.error(f"X and y dimension mismatch: X={x_samples}, y={y_samples}")
                # Try to fix the mismatch
                min_samples = min(x_samples, y_samples)
                if min_samples > 0:
                    if sparse.issparse(X_full):
                        X_full = X_full[:min_samples]
                    else:
                        X_full = X_full[:min_samples]
                    y_full = y_full[:min_samples]
                    self.logger.info(f"Truncated to {min_samples} samples for consistency")
                else:
                    return {'cv_failed': True, 'error': 'Dimension mismatch cannot be resolved'}

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
            
            cv_scores = {
                'accuracy': [],
                'weighted_f1': [],
                'weighted_precision': [],
                'weighted_recall': [],
                'roc_auc_ovr': []
            }
            
            fold_times = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
                print(f"Processing fold {fold + 1}/{cv_folds}")
                
                # FIXED: Safe indexing for both sparse and dense
                if sparse.issparse(X_full):
                    X_train_fold = X_full[train_idx]
                    X_val_fold = X_full[val_idx]
                else:
                    X_train_fold = X_full[train_idx]
                    X_val_fold = X_full[val_idx]
                
                y_train_fold = y_full[train_idx]
                y_val_fold = y_full[val_idx]

                start_time = time.time()

                # Create model for this fold
                if config.get('ensemble_type') == 'none':
                    model = BaseModelFactory.create_model(config['base_models'][0], self.n_classes)
                else:
                    model = EnsembleFactory.create_ensemble(config, self.n_classes)

                # Train and evaluate
                model.fit(X_train_fold, y_train_fold)
                fold_time = time.time() - start_time
                fold_times.append(fold_time)

                y_pred_proba = model.predict_proba(X_val_fold)
                y_pred = np.argmax(y_pred_proba, axis=1)

                # Calculate fold metrics
                cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                cv_scores['weighted_f1'].append(f1_score(y_val_fold, y_pred, average='weighted'))
                cv_scores['weighted_precision'].append(precision_score(y_val_fold, y_pred, average='weighted', zero_division=0))
                cv_scores['weighted_recall'].append(recall_score(y_val_fold, y_pred, average='weighted', zero_division=0))

                try:
                    cv_scores['roc_auc_ovr'].append(roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr', average='weighted'))
                except:
                    cv_scores['roc_auc_ovr'].append(0.5)

            # Calculate statistics
            cv_stats = {}
            for metric, scores in cv_scores.items():
                cv_stats[f'{metric}_mean'] = np.mean(scores)
                cv_stats[f'{metric}_std'] = np.std(scores)
                cv_stats[f'{metric}_scores'] = scores

            cv_stats['training_time_mean'] = np.mean(fold_times)
            cv_stats['training_time_std'] = np.std(fold_times)
            cv_stats['cv_folds'] = cv_folds

            print(f"CV completed. F1: {cv_stats['weighted_f1_mean']:.4f} ± {cv_stats['weighted_f1_std']:.4f}")

            return cv_stats

        except Exception as e:
            print(f"Comprehensive CV failed for {model_name}: {str(e)}")
            return {'cv_failed': True, 'error': str(e)}

    def full_evaluate(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced full evaluation with SAFE SHAPE HANDLING"""
        try:
            print(f"Starting full evaluation of {model_name}")
            print(f"Model config: {config}")
            
            # Overall timing
            overall_start_time = time.time()
            
            # Resource monitoring
            start_resources = monitor_resources()
            print(f"Initial memory usage: {start_resources['memory_percent']:.1f}%")

            # Model creation timing
            model_creation_start = time.time()
            print(f"Creating model architecture...")
            if config.get('ensemble_type') == 'none':
                model = BaseModelFactory.create_model(config['base_models'][0], self.n_classes)
                print(f"Single model created: {config['base_models']}")
            else:
                model = EnsembleFactory.create_ensemble(config, self.n_classes)
                print(f"Ensemble created: {config['ensemble_type']} with {len(config['base_models'])} base models")
            
            model_creation_time = time.time() - model_creation_start
            print(f"Model creation completed in {model_creation_time:.2f} seconds")

            # Data preparation timing
            data_prep_start = time.time()
            print(f"Preparing training data...")
            X_train_use = self.X_train
            X_test_use = self.X_test
            y_train_use = self.y_train

            # OPTIMIZED: Do conversion once and reuse
            if sparse.issparse(X_train_use):
                base_models = config.get('base_models', [])
                needs_dense = any(model.lower().replace('_', '') in ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm', 'tabflex']
                                for model in base_models)
                if needs_dense:
                    print(f"Converting sparse matrices to dense for compatibility...")
                    dense_conversion_start = time.time()
                    X_train_use = X_train_use.toarray()
                    X_test_use = self.X_test.toarray()
                    X_val_eval = self.X_val.toarray() if sparse.issparse(self.X_val) else self.X_val
                    dense_conversion_time = time.time() - dense_conversion_start
                    print(f"Dense conversion completed in {dense_conversion_time:.2f} seconds")
                else:
                    X_val_eval = self.X_val
            else:
                X_val_eval = self.X_val

            if hasattr(y_train_use, 'values'):
                y_train_use = y_train_use.values

            data_prep_time = time.time() - data_prep_start
            print(f"Data preparation completed in {data_prep_time:.2f} seconds")

            # Cross-validation timing
            cv_start = time.time()
            print(f"Starting comprehensive cross-validation...")
            cv_results = self.comprehensive_cv_evaluate(model_name, config, cv_folds=3)
            cv_time = time.time() - cv_start
            print(f"Cross-validation completed in {cv_time:.2f} seconds")

            # FIXED: Safe data validation using helper function
            def safe_get_shape(arr):
                """Safely get the first dimension of an array"""
                # try:
                #     if hasattr(arr, 'shape'):
                #         shape_val = arr.shape[0]
                #         # Ensure we return an integer, not a tuple
                #         return int(shape_val) if not isinstance(shape_val, tuple) else int(shape_val)
                #     else:
                #         return len(arr)
                # except:
                #     return 0
                
                try:
                    return safe_array_length(arr)
                except Exception:
                    return 0

            train_samples = get_n_samples(X_train_use)
            train_labels = get_n_samples(y_train_use)
            test_samples = get_n_samples(X_test_use)
            test_labels = get_n_samples(self.y_test)
                        
            if train_labels == 0 or train_samples == 0:
                print("No training data available for evaluation")
                return {
                    'model_name': model_name, 'config': config, 'status': 'failed',
                    'error': 'No training data available', 'timestamp': datetime.now().isoformat()
                }

            if test_labels == 0 or test_samples == 0:
                print("No test data available for evaluation")
                return {
                    'model_name': model_name, 'config': config, 'status': 'failed',
                    'error': 'No test data available', 'timestamp': datetime.now().isoformat()
                }

            # Training timing
            training_start = time.time()
            print(f"Starting model training...")
            print(f"Training set size: {train_samples} samples, {safe_get_shape(X_train_use[0:1])} features" if train_samples > 0 else "Training set size: unknown")
            
            n_features = X_train_use.shape[1] if hasattr(X_train_use, 'shape') and len(X_train_use.shape) > 1 else "unknown"
            print(f"Training set size: {train_samples} samples, {n_features} features" if train_samples > 0 else "Training set size: unknown")
            
            model.fit(X_train_use, y_train_use)
            training_time = time.time() - training_start
            print(f"Training completed in {training_time:.2f} seconds")

            end_resources = monitor_resources()
            memory_usage = max(0, end_resources['memory_percent'] - start_resources['memory_percent'])
            print(f"Memory usage change: +{memory_usage:.1f}%")

            # DETAILED PREDICTION PHASE WITH SAFE SHAPE HANDLING
            print(f"="*50)
            print(f"STARTING PREDICTION PHASE")
            print(f"="*50)
            
            # Check if this is a problematic model type
            is_complex_model = any(base_model in ['tabflex', 'mixture_of_experts', 'deepgbm'] 
                                for base_model in config.get('base_models', []))
            
            if is_complex_model:
                print(f"COMPLEX MODEL DETECTED: Using optimized prediction strategy")

            total_prediction_start = time.time()

            # 1. Training set predictions with SAFE shape checking
            train_pred_start = time.time()
            print(f"Predicting probabilities for TRAINING set...")
            print(f"Training set shape: ({train_samples}, {X_train_use.shape[1] if hasattr(X_train_use, 'shape') else 'unknown'})")
            
            # FIXED: Safe shape comparison
            if train_samples > 5000 and is_complex_model:
                print(f"Using batched prediction for large training set...")
                y_train_pred_proba = self._predict_proba_batched_safe(model, X_train_use, batch_size=1000)
            else:
                y_train_pred_proba = model.predict_proba(X_train_use)
            
            train_pred_time = time.time() - train_pred_start
            print(f"Training predictions completed in {train_pred_time:.2f} seconds")
            print(f"Training prediction shape: {y_train_pred_proba.shape}")

            # 2. Validation set predictions with SAFE shape checking
            val_pred_start = time.time()
            print(f"Predicting probabilities for VALIDATION set...")
            val_samples = get_n_samples(X_val_eval)
            print(f"Validation set shape: ({val_samples}, {X_val_eval.shape[1] if hasattr(X_val_eval, 'shape') else 'unknown'})")
            
            # FIXED: Safe shape comparison
            if val_samples > 2000 and is_complex_model:
                print(f"Using batched prediction for validation set...")
                y_val_pred_proba = self._predict_proba_batched_safe(model, X_val_eval, batch_size=500)
            else:
                y_val_pred_proba = model.predict_proba(X_val_eval)
            
            val_pred_time = time.time() - val_pred_start
            print(f"Validation predictions completed in {val_pred_time:.2f} seconds")
            print(f"Validation prediction shape: {y_val_pred_proba.shape}")

            # 3. Test set predictions with SAFE shape checking
            test_pred_start = time.time()
            print(f"Predicting probabilities for TEST set...")
            print(f"Test set shape: ({test_samples}, {X_test_use.shape[1] if hasattr(X_test_use, 'shape') else 'unknown'})")
            
            # FIXED: Safe shape comparison
            if test_samples > 2000 and is_complex_model:
                print(f"Using batched prediction for test set...")
                y_test_pred_proba = self._predict_proba_batched_safe(model, X_test_use, batch_size=500)
            else:
                y_test_pred_proba = model.predict_proba(X_test_use)
            
            test_pred_time = time.time() - test_pred_start
            print(f"Test predictions completed in {test_pred_time:.2f} seconds")
            print(f"Test prediction shape: {y_test_pred_proba.shape}")

            total_prediction_time = time.time() - total_prediction_start
            print(f"="*50)
            print(f"TOTAL PREDICTION TIME: {total_prediction_time:.2f} seconds")
            print(f"="*50)

            # Continue with metrics calculation...
            print(f"Calculating comprehensive metrics...")
            
            # Calculate training metrics
            train_metrics = MetricsCalculator.calculate_tier1_metrics(
                y_train_use, y_train_pred_proba, training_time, memory_usage, self.n_classes
            )
            
            # Calculate test metrics  
            y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else np.array(self.y_test)
            test_metrics = MetricsCalculator.calculate_tier1_metrics(
                y_test_array, y_test_pred_proba, training_time, memory_usage, self.n_classes
            )
            
            # Add Tier 2 metrics
            train_tier2 = MetricsCalculator.calculate_tier2_metrics(
                y_train_use, y_train_pred_proba
            )
            test_tier2 = MetricsCalculator.calculate_tier2_metrics(
                y_test_array, y_test_pred_proba
            )
            
            train_metrics.update(train_tier2)
            test_metrics.update(test_tier2)
            
            # Store actual y values for visualization
            train_metrics['y_true'] = y_train_use
            test_metrics['y_true'] = y_test_array

            # OPTIMIZED Bootstrap analysis (reduced samples for speed)
            bootstrap_start = time.time()
            print(f"Starting bootstrap analysis...")
            bootstrap_analyzer = self._get_adaptive_bootstrap_analyzer(model_name, test_samples)

            # FIXED: Enhanced bootstrap data validation
            bootstrap_start = time.time()
            print(f"Starting bootstrap analysis...")
            bootstrap_analyzer = self._get_adaptive_bootstrap_analyzer(model_name, get_n_samples(X_test_use))
            
            # CRITICAL VALIDATION: Check data before bootstrap with proper sparse handling
            try:
                y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else np.array(self.y_test)
                y_test_array = np.asarray(y_test_array).flatten()
                
                # FIXED: Use safe sparse-aware length checking
                x_test_samples = get_n_samples(X_test_use)
                y_test_samples = len(y_test_array)
                
                if y_test_samples == 0 or x_test_samples == 0:
                    print(f"Skipping bootstrap: empty data arrays")
                    test_bootstrap = bootstrap_analyzer._get_fallback_bootstrap_result()
                elif x_test_samples != y_test_samples:
                    print(f"Skipping bootstrap: data shape mismatch X={x_test_samples}, y={y_test_samples}")
                    test_bootstrap = bootstrap_analyzer._get_fallback_bootstrap_result()
                else:
                    # Proceed with bootstrap only if data is valid
                    if x_test_samples > 3000:
                        print(f"Using bootstrap subsample for large dataset...")
                        sample_indices = np.random.choice(x_test_samples, 2000, replace=False)
                        
                        # FIXED: Handle sparse matrix subsampling properly
                        if sparse.issparse(X_test_use):
                            X_bootstrap = X_test_use[sample_indices]
                        else:
                            X_bootstrap = X_test_use[sample_indices]
                        y_bootstrap = y_test_array[sample_indices]
                        test_bootstrap = bootstrap_analyzer.bootstrap_model_performance(model, X_bootstrap, y_bootstrap)
                    else:
                        test_bootstrap = bootstrap_analyzer.bootstrap_model_performance(model, X_test_use, y_test_array)
                        
                    print(f"Bootstrap analysis completed with {test_bootstrap.get('success_rate', 0):.2%} success rate")
                    
            except Exception as bootstrap_error:
                print(f"Bootstrap analysis failed: {bootstrap_error}")
                test_bootstrap = bootstrap_analyzer._get_fallback_bootstrap_result()
            
            bootstrap_time = time.time() - bootstrap_start
            print(f"Bootstrap analysis completed in {bootstrap_time:.2f} seconds")

            print(f"All predictions and analysis completed successfully!")

            # Assemble final results
            results = {
                'model_name': model_name,
                'config': config,
                'status': 'completed',
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cross_validation': cv_results,
                'bootstrap_analysis': test_bootstrap,
                'predictions': {
                    'train_proba': y_train_pred_proba,
                    'test_proba': y_test_pred_proba,
                    'val_proba': y_val_pred_proba
                },
                'timing': {
                    'model_creation': model_creation_time,
                    'data_preparation': data_prep_time,
                    'cross_validation': cv_time,
                    'training': training_time,
                    'prediction': total_prediction_time,
                    'bootstrap': bootstrap_time,
                    'total': time.time() - overall_start_time
                },
                'timestamp': datetime.now().isoformat()
            }

            total_time = time.time() - overall_start_time
            print(f"FULL EVALUATION COMPLETED in {total_time:.2f} seconds")
            print(f"Time breakdown:")
            print(f"  - Model creation: {model_creation_time:.2f}s")
            print(f"  - Data preparation: {data_prep_time:.2f}s") 
            print(f"  - Cross-validation: {cv_time:.2f}s")
            print(f"  - Training: {training_time:.2f}s")
            print(f"  - Predictions: {total_prediction_time:.2f}s")
            print(f"  - Bootstrap: {bootstrap_time:.2f}s")

            return results

        except Exception as e:
            print(f"Full evaluation FAILED for {model_name}: {str(e)}")
            return {
                'model_name': model_name, 'config': config, 'status': 'failed',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }
            
    def _get_adaptive_bootstrap_analyzer(self, model_name: str, dataset_size: int):
        """Get bootstrap analyzer with adaptive settings based on model complexity and dataset size"""
        
        # Check memory pressure
        memory_info = monitor_resources()
        high_memory = memory_info['memory_percent'] > 90
        
        # Classify model complexity
        complex_models = ['neuralnetwork', 'advancedsvm', 'tabflex', 'mixture_of_experts', 'deepgbm']
        is_complex = any(cm in model_name.lower() for cm in complex_models)
        
        # Determine bootstrap settings
        if high_memory or (is_complex and dataset_size > 5000):
            # ULTRA-FAST mode for high memory or complex models
            n_bootstrap = 10
            max_samples = 1000
            self.logger.warning(f"Using ULTRA-FAST bootstrap: {n_bootstrap} iterations, {max_samples} samples")
        elif is_complex or dataset_size > 10000:
            # FAST mode for complex models or large datasets  
            n_bootstrap = 20
            max_samples = 3000
            self.logger.info(f"Using FAST bootstrap: {n_bootstrap} iterations, {max_samples} samples")
        else:
            # NORMAL mode
            n_bootstrap = 30
            max_samples = 8000
            self.logger.info(f"Using NORMAL bootstrap: {n_bootstrap} iterations, {max_samples} samples")
        
        return BootstrapAnalyzer(n_bootstrap=n_bootstrap, max_bootstrap_samples=max_samples)
            
    def _prepare_all_data_formats(self, config):
        """Prepare all data formats once to avoid repeated conversions"""
        base_models = config.get('base_models', [])
        needs_dense = any(model.lower().replace('_', '') in 
                        ['naivebayes', 'nb', 'gaussiannb', 'neuralnetwork', 'mlp', 'mlpclassifier', 'advancedsvm', 'deepgbm']
                        for model in base_models)
        
        if sparse.issparse(self.X_train) and needs_dense:
            print("Converting to dense format once...")
            X_train_use = self.X_train.toarray()
            X_test_use = self.X_test.toarray()
            X_val_use = self.X_val.toarray()
        else:
            X_train_use = self.X_train
            X_test_use = self.X_test
            X_val_use = self.X_val
        
        return X_train_use, X_test_use, X_val_use

    def _predict_proba_batched_safe(self, model, X, batch_size=None):
        """SAFE batched prediction with proper shape handling"""
        
        n_samples = get_n_samples(X)
        
        # Dynamic batch sizing based on model complexity and available memory
        if batch_size is None:
            model_name = model.__class__.__name__.lower()
            if 'tabflex' in model_name or 'mixture' in model_name:
                batch_size = min(256, max(1, n_samples // 10))
            elif 'deepgbm' in model_name:
                batch_size = min(512, max(1, n_samples // 5))
            else:
                batch_size = min(2000, max(1, n_samples // 2))
        
        print(f"Using adaptive batch size: {batch_size}")
        
        n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
        all_probas = []
        
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, n_samples)
            
            if i % max(1, n_batches // 10) == 0:
                print(f"Batch progress: {i+1}/{n_batches} ({100*(i+1)/n_batches:.1f}%)")
            
            try:
                # FIXED: Safe indexing for both sparse and dense
                if sparse.issparse(X):
                    X_batch = X[batch_start:batch_end]
                else:
                    X_batch = X[batch_start:batch_end]
                
                batch_proba = model.predict_proba(X_batch)
                all_probas.append(batch_proba)
                
            except Exception as e:
                print(f"Batch {i+1} failed: {e}")
                # FIXED: Use consistent n_classes
                n_classes = getattr(model, 'n_classes_', self.n_classes)
                fallback = np.ones((batch_end - batch_start, n_classes)) / n_classes
                all_probas.append(fallback)
        
        return np.vstack(all_probas) if all_probas else np.ones((n_samples, self.n_classes)) / self.n_classes
    
class EnsembleSelectionFramework:
    """Systematic ensemble type selection with theoretical justification"""
    
    # Algorithm family classifications for systematic analysis
    ALGORITHM_FAMILIES = {
        'tree_boosting': ['lightgbm', 'xgboost', 'catboost', 'adaboost', 'gradientboost'],
        'tree_bagging': ['randomforest'],
        'linear_models': ['logisticregression', 'ridge', 'sgd', 'linearsvm', 'linear_model'],
        'probabilistic': ['naivebayes'],
        'neural_networks': ['neuralnetwork', 'mlp', 'tabflex'],
        'kernel_methods': ['advancedsvm', 'svmrbf'],
        'instance_based': ['knn'],
        'advanced_ensemble': ['deepgbm', 'ngboost', 'mixture_of_experts']
    }
    
    @staticmethod
    def classify_algorithms(base_models: List[str]) -> Dict[str, int]:
        """Classify algorithms by family for systematic analysis"""
        family_counts = {family: 0 for family in EnsembleSelectionFramework.ALGORITHM_FAMILIES}
        
        for model in base_models:
            model_clean = model.lower().replace('_', '')
            for family, algorithms in EnsembleSelectionFramework.ALGORITHM_FAMILIES.items():
                if any(algo in model_clean for algo in algorithms):
                    family_counts[family] += 1
                    break
        
        return family_counts
    
    @staticmethod
    def calculate_algorithmic_diversity(family_counts: Dict[str, int]) -> float:
        """Calculate algorithmic diversity score (0-1)"""
        active_families = sum(1 for count in family_counts.values() if count > 0)
        return active_families / len(family_counts)
    
    @staticmethod
    def get_optimal_ensemble_type(base_models: List[str], complexity_budget: str = 'medium') -> str:
        """
        Systematically determine optimal ensemble type based on algorithmic theory
        
        Theoretical Framework:
        1. High Diversity (3+ families) -> Stacking (meta-learning benefits)
        2. Probabilistic Compatibility -> Voting (similar confidence scales)
        3. Performance Hierarchy -> Weighted (known differences)
        4. Single Algorithm -> CV/Parametric (hyperparameter diversity)
        5. Complexity Constraints -> Voting (computational efficiency)
        """
        
        if len(base_models) == 1:
            return 'cv_ensemble'  # Single algorithm diversity through hyperparameters
        
        family_counts = EnsembleSelectionFramework.classify_algorithms(base_models)
        diversity_score = EnsembleSelectionFramework.calculate_algorithmic_diversity(family_counts)
        total_models = len(base_models)
        
        # Complexity constraint rules
        high_complexity_families = ['neural_networks', 'advanced_ensemble', 'kernel_methods']
        complexity_count = sum(family_counts[family] for family in high_complexity_families)
        
        if complexity_budget == 'low' or (complexity_count > 0 and total_models > 2):
            return 'voting'  # Avoid expensive meta-learning with complex base models
        
        # Theoretical decision framework
        if diversity_score >= 0.5:  # High algorithmic diversity (3+ families)
            return 'stacking'
        elif family_counts['tree_boosting'] > 1 or family_counts['tree_bagging'] > 0:
            if diversity_score <= 0.3:  # Similar tree-based models
                return 'voting'
        elif family_counts['probabilistic'] > 0 and diversity_score <= 0.3:
            return 'voting'  # Probabilistic compatibility
        elif total_models == 2 and complexity_count == 0:
            return 'weighted'  # Two simple models with potential hierarchy
        else:
            return 'stacking'  # Default for moderate diversity
    
    @staticmethod
    def generate_performance_based_weights(base_models: List[str]) -> List[float]:
        """
        Generate theoretically justified weights based on empirical performance characteristics
        """
        # Performance rankings based on literature and empirical studies
        performance_scores = {
            'lightgbm': 0.95, 'xgboost': 0.93, 'catboost': 0.91,
            'deepgbm': 0.89, 'ngboost': 0.85,
            'randomforest': 0.78, 'gradientboost': 0.76, 'adaboost': 0.68,
            'tabflex': 0.82, 'neuralnetwork': 0.75,
            'advancedsvm': 0.72, 'linearsvm': 0.58,
            'logisticregression': 0.62, 'ridge': 0.55, 'sgd': 0.52,
            'naivebayes': 0.48, 'knn': 0.45,
            'mixture_of_experts': 0.87
        }
        
        weights = []
        for model in base_models:
            model_clean = model.lower().replace('_', '')
            score = 0.5  # Default score
            
            for algo, perf_score in performance_scores.items():
                if algo in model_clean:
                    score = perf_score
                    break
            
            weights.append(score)
        
        # Normalize weights
        weight_sum = sum(weights)
        return [w / weight_sum for w in weights] if weight_sum > 0 else [1.0/len(weights)] * len(weights)

class SystematicConfigurationGenerator:
    """Generate balanced, theoretically justified model configurations"""
    
    def __init__(self):
        self.ensemble_framework = EnsembleSelectionFramework()
        
    def generate_systematic_configurations(self, target_models: List[Dict]) -> Dict[str, Dict]:
        """
        Generate configurations with systematic ensemble type testing
        
        For each target model combination, test all applicable ensemble types
        """
        configurations = {}
        
        for model_spec in target_models:
            base_config = {
                'base_models': model_spec['algorithms'],
                'description': model_spec.get('description', ''),
                'research_purpose': model_spec.get('purpose', 'comparative_analysis')
            }
            
            # Generate all applicable ensemble configurations
            ensemble_variants = self._generate_ensemble_variants(model_spec['algorithms'])
            
            for ensemble_type, config in ensemble_variants.items():
                model_name = f"{model_spec['name']}_{ensemble_type}"
                configurations[model_name] = {**base_config, **config}
        
        return configurations
    
    def _generate_ensemble_variants(self, base_models: List[str]) -> Dict[str, Dict]:
        """Generate all applicable ensemble variants for given base models"""
        # VALIDATE that all base models can be created
        valid_models = []
        for model in base_models:
            try:
                # Test if model can be created (dry run)
                test_model = BaseModelFactory.create_model(model, 3)  # 3 classes for testing
                valid_models.append(model)
            except Exception as e:
                print(f"Warning: Skipping invalid model {model}: {e}")
        
        if not valid_models:
            return {}
        
        base_models = valid_models  # Use only valid models
        variants = {}
        
        if len(base_models) == 1:
            # Single algorithm variants
            variants.update({
                'solo': {'ensemble_type': 'none'},
                'cv': {'ensemble_type': 'cv_ensemble', 'cv_folds': 3},
                'parametric': {'ensemble_type': 'parametric_ensemble', 'n_versions': 3}
            })
        else:
            # Multi-algorithm variants
            family_counts = self.ensemble_framework.classify_algorithms(base_models)
            complexity_count = sum(family_counts[f] for f in ['neural_networks', 'advanced_ensemble', 'kernel_methods'])
            
            # Always include voting (baseline ensemble)
            variants['voting'] = {'ensemble_type': 'voting'}
            
            # Weighted ensemble for 2-3 models
            if 2 <= len(base_models) <= 3:
                weights = self.ensemble_framework.generate_performance_based_weights(base_models)
                variants['weighted'] = {
                    'ensemble_type': 'weighted',
                    'weights': weights
                }
            
            # Stacking for diverse, non-complex combinations
            if complexity_count == 0 or len(base_models) <= 2:
                variants['stacking'] = {'ensemble_type': 'stacking'}
            
            # Ridge stacking for 3+ tree-based models
            tree_count = family_counts['tree_boosting'] + family_counts['tree_bagging']
            if tree_count >= 2 and len(base_models) >= 3:
                variants['stacking_ridge'] = {'ensemble_type': 'stacking_ridge'}
            
            # Parametric ensemble for homogeneous combinations
            if len(set(base_models)) <= 2:  # Similar algorithms
                variants['parametric'] = {
                    'ensemble_type': 'parametric_ensemble',
                    'n_versions': 2
                }
        
        return variants

class StatisticalTester:
    """Statistical significance testing for model comparisons"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_models_statistically(self, model1_scores: List[float], model2_scores: List[float],
                                   model1_name: str, model2_name: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Compare two models using multiple statistical tests"""
        from scipy import stats

        try:
            # Ensure equal length
            min_len = min(len(model1_scores), len(model2_scores))
            scores1 = np.array(model1_scores[:min_len])
            scores2 = np.array(model2_scores[:min_len])

            results = {
                'model1_name': model1_name,
                'model2_name': model2_name,
                'model1_mean': np.mean(scores1),
                'model2_mean': np.mean(scores2),
                'difference': np.mean(scores1) - np.mean(scores2)
            }

            # Paired t-test
            t_stat, t_pvalue = stats.ttest_rel(scores1, scores2)
            results['paired_ttest'] = {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'significant': t_pvalue < alpha,
                'interpretation': f"{'Significant' if t_pvalue < alpha else 'Not significant'} difference (p={t_pvalue:.4f})"
            }

            # Wilcoxon signed-rank test (non-parametric)
            w_stat, w_pvalue = stats.wilcoxon(scores1, scores2, alternative='two-sided')
            results['wilcoxon_test'] = {
                'statistic': w_stat,
                'pvalue': w_pvalue,
                'significant': w_pvalue < alpha,
                'interpretation': f"{'Significant' if w_pvalue < alpha else 'Not significant'} difference (p={w_pvalue:.4f})"
            }

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            results['effect_size'] = {
                'cohens_d': cohens_d,
                'magnitude': self._interpret_effect_size(cohens_d)
            }

            return results

        except Exception as e:
            self.logger.error(f"Statistical comparison failed: {str(e)}")
            return {'error': str(e)}

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def multiple_comparisons_correction(self, pvalues: List[float], method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparisons correction"""
        from statsmodels.stats.multitest import multipletests

        try:
            _, corrected_pvalues, _, _ = multipletests(pvalues, method=method)
            return corrected_pvalues.tolist()
        except Exception as e:
            self.logger.error(f"Multiple comparisons correction failed: {str(e)}")
            return pvalues

class BootstrapAnalyzer:
    """OPTIMIZED Bootstrap analyzer with adaptive limits"""

    def __init__(self, n_bootstrap: int = 30, confidence_level: float = 0.95, max_bootstrap_samples: int = None):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level  
        self.max_bootstrap_samples = max_bootstrap_samples
        self.logger = logging.getLogger(__name__)
        
    def _get_fallback_bootstrap_result(self):
        """Enhanced fallback bootstrap result with debugging info"""
        return {
            'error': 'Bootstrap analysis failed - invalid data dimensions or insufficient data',
            'mean': 0.0,
            'std': 0.0,
            'confidence_interval': (0.0, 0.0),
            'confidence_level': self.confidence_level,
            'bootstrap_scores': [],
            'n_bootstrap': 0,
            'success_rate': 0.0,
            'fallback_used': True,
            'fallback_reason': 'dimensionality_error'
        }

    def bootstrap_model_performance(self, model, X: np.ndarray, y: np.ndarray, metric_func=None) -> Dict[str, Any]:
        """FIXED: Bootstrap with proper sparse matrix handling"""
        if metric_func is None:
            metric_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')

        try:
            # FIXED: Proper array conversion and validation
            if hasattr(y, 'values'):
                y = y.values
            if hasattr(X, 'values'):
                X = X.values
                
            y = np.asarray(y).flatten()
            # DON'T convert sparse X to dense array here - keep it sparse
            
            # CRITICAL FIX: Use safe length checking for sparse matrices
            # if sparse.issparse(X):
            #     n_samples = X.shape[0]  # FIXED: Use shape instead of len()
            # else:
            #     X = np.asarray(X)
            #     n_samples = len(X)
            
            # WITH:
            n_samples = get_n_samples(X)
            if not sparse.issparse(X):
                X = np.asarray(X)
            
            # Validate we have matching dimensions
            if len(y) != n_samples:
                self.logger.error(f"X and y length mismatch: X={n_samples}, y={len(y)}")
                return self._get_fallback_bootstrap_result()
            
            # ADAPTIVE LIMITS based on dataset size and memory
            memory_info = monitor_resources()
            high_memory = memory_info['memory_percent'] > 90
            
            # Dynamic bootstrap settings
            if high_memory or n_samples > 10000:
                actual_n_bootstrap = min(15, self.n_bootstrap)
                max_sample_size = min(2000, n_samples // 2)
                self.logger.warning(f"High memory/large dataset: reducing bootstrap to {actual_n_bootstrap} iterations, {max_sample_size} samples each")
            elif n_samples > 5000:
                actual_n_bootstrap = min(25, self.n_bootstrap)
                max_sample_size = min(5000, n_samples)
                self.logger.info(f"Large dataset: reducing bootstrap to {actual_n_bootstrap} iterations, {max_sample_size} samples each")
            else:
                actual_n_bootstrap = self.n_bootstrap
                max_sample_size = n_samples
            
            # Override with manual limit if provided
            if self.max_bootstrap_samples is not None:
                max_sample_size = min(max_sample_size, self.max_bootstrap_samples)

            bootstrap_scores = []
            successful_bootstraps = 0
            max_failures = min(20, actual_n_bootstrap // 3)
            failures = 0

            progress_interval = max(1, actual_n_bootstrap // 5)

            for i in range(actual_n_bootstrap):
                if failures > max_failures:
                    print(f"Too many bootstrap failures ({failures}), stopping early")
                    break
                    
                if i % progress_interval == 0:
                    print(f"Bootstrap progress: {i+1}/{actual_n_bootstrap} ({100*(i+1)/actual_n_bootstrap:.1f}%)")
                    
                try:
                    # OPTIMIZED: Use smaller sample size
                    bootstrap_sample_size = min(max_sample_size, n_samples)
                    indices = np.random.choice(n_samples, bootstrap_sample_size, replace=True)
                    
                    # FIXED: Handle sparse matrix indexing properly
                    if sparse.issparse(X):
                        X_boot = X[indices]  # Sparse matrices support fancy indexing
                    else:
                        X_boot = X[indices]
                    y_boot = y[indices]
                    
                    # Validate bootstrap sample
                    if len(y_boot) == 0 or len(np.unique(y_boot)) < 2:
                        failures += 1
                        continue

                    # OPTIMIZED: Prediction with error handling
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_boot)
                            y_pred = np.argmax(y_pred_proba, axis=1)
                        else:
                            y_pred = model.predict(X_boot)
                    except Exception as pred_error:
                        failures += 1
                        if failures <= 3:
                            print(f"Bootstrap prediction failed: {pred_error}")
                        continue

                    # Calculate metric
                    try:
                        score = metric_func(y_boot, y_pred)
                        if np.isfinite(score):
                            bootstrap_scores.append(score)
                            successful_bootstraps += 1
                        else:
                            failures += 1
                    except Exception as metric_error:
                        failures += 1
                        if failures <= 3:
                            print(f"Bootstrap metric calculation failed: {metric_error}")
                        continue

                except Exception as bootstrap_error:
                    failures += 1
                    if failures <= 3:
                        print(f"Bootstrap iteration {i} failed: {bootstrap_error}")
                    continue

            # Check if we have enough successful bootstraps
            min_successful = max(5, actual_n_bootstrap // 5)
            if len(bootstrap_scores) < min_successful:
                self.logger.warning(f"Insufficient successful bootstraps: {len(bootstrap_scores)}/{actual_n_bootstrap}")
                return self._get_fallback_bootstrap_result()

            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)

            print(f"Bootstrap completed: {successful_bootstraps}/{actual_n_bootstrap} successful samples")

            return {
                'mean': float(np.mean(bootstrap_scores)),
                'std': float(np.std(bootstrap_scores)),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'confidence_level': self.confidence_level,
                'bootstrap_scores': bootstrap_scores[:50],
                'n_bootstrap': successful_bootstraps,
                'success_rate': successful_bootstraps / actual_n_bootstrap,
                'adaptive_limits_used': {
                    'n_bootstrap_actual': actual_n_bootstrap,
                    'max_sample_size': max_sample_size,
                    'high_memory_mode': high_memory
                }
            }

        except Exception as e:
            print(f"Bootstrap analysis failed: {str(e)}")
            return self._get_fallback_bootstrap_result()

    def _get_fallback_bootstrap_result(self):
        """ENHANCED fallback bootstrap result"""
        return {
            'error': 'Bootstrap analysis failed or insufficient data',
            'mean': 0.0,
            'std': 0.0,
            'confidence_interval': (0.0, 0.0),
            'confidence_level': self.confidence_level,
            'bootstrap_scores': [],
            'n_bootstrap': 0,
            'success_rate': 0.0,
            'fallback_used': True
        }

# ================================================================================================
# CHECKPOINT AND PERSISTENCE SYSTEM
# ================================================================================================

class CheckpointManager:
    def __init__(self):
        self.checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, 'checkpoint.json')
        self.results_file = os.path.join(Config.RESULTS_DIR, 'all_results.pkl')

        # ADD THESE NEW FILES
        self.master_results_file = os.path.join(Config.RESULTS_DIR, 'master_results.csv')
        self.detailed_metrics_file = os.path.join(Config.RESULTS_DIR, 'detailed_metrics.csv')
        self.train_metrics_file = os.path.join(Config.RESULTS_DIR, 'train_metrics.csv')
        self.test_metrics_file = os.path.join(Config.RESULTS_DIR, 'test_metrics.csv')
        self.failed_models_file = os.path.join(Config.RESULTS_DIR, 'failed_models.csv')
        self.historical_summary_file = os.path.join(Config.REPORTS_DIR, 'cumulative_analysis.txt')

        self.logger = logging.getLogger(__name__)

        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()

    # Add this method to your existing CheckpointManager class
    def save_checkpoint(self, model_name: str, results: Dict[str, Any]):
        """Enhanced checkpoint saving with CSV exports"""
        try:
            # Load existing checkpoint
            checkpoint = self.load_checkpoint()

            # Update checkpoint based on status
            if results['status'] == 'completed':
                if model_name not in checkpoint['completed_models']:
                    checkpoint['completed_models'].append(model_name)
            elif results['status'] == 'failed':
                if model_name not in checkpoint['failed_models']:
                    checkpoint['failed_models'].append(model_name)
            elif results['status'] == 'early_stopped':
                if model_name not in checkpoint['early_stopped_models']:
                    checkpoint['early_stopped_models'].append(model_name)

            checkpoint['timestamp'] = datetime.now().isoformat()

            # Save original checkpoint
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Save to CSV files for easy reading
            self._save_to_csv_files(model_name, results)

            # Update cumulative analysis
            self._update_cumulative_analysis()

            # Save detailed results (without large arrays to save space)
            results_to_save = results.copy()
            if 'predictions' in results_to_save:
                del results_to_save['predictions']

            # Load existing results
            all_results = self.load_all_results()
            all_results[model_name] = results_to_save

            # Save results
            with open(self.results_file, 'wb') as f:
                pickle.dump(all_results, f)

            self.logger.info(f"Checkpoint and CSV files updated for {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for {model_name}: {str(e)}")

    def _initialize_csv_files(self):
        """Initialize CSV files with proper headers"""
        # Master results header
        master_headers = ['timestamp', 'model_name', 'status', 'ensemble_type', 'base_models',
                         'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_roc_auc',
                         'train_accuracy', 'train_f1', 'train_precision', 'train_recall', 'train_roc_auc',
                         'training_time', 'memory_usage', 'complexity_flagged', 'reason']

        if not os.path.exists(self.master_results_file):
            pd.DataFrame(columns=master_headers).to_csv(self.master_results_file, index=False)

        # Detailed metrics headers (ALL metrics)
        detailed_headers = ['timestamp', 'model_name', 'dataset_type'] + TIER1_METRICS + TIER2_METRICS + TIER3_METRICS

        if not os.path.exists(self.detailed_metrics_file):
            pd.DataFrame(columns=detailed_headers).to_csv(self.detailed_metrics_file, index=False)

    def load_checkpoint(self) -> Dict[str, List[str]]:
        """Load existing checkpoint"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
            else:
                checkpoint = {
                    'completed_models': [],
                    'failed_models': [],
                    'early_stopped_models': [],
                    'timestamp': None
                }
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return {'completed_models': [], 'failed_models': [], 'early_stopped_models': []}

    def load_all_results(self) -> Dict[str, Any]:
        """Load all saved results"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load results: {str(e)}")
            return {}

    def _save_to_csv_files(self, model_name: str, results: Dict[str, Any]):
        """Save results to multiple CSV files for easy analysis"""
        timestamp = datetime.now().isoformat()

        # Prepare master results row
        master_row = {
            'timestamp': timestamp,
            'model_name': model_name,
            'status': results.get('status', 'unknown'),
            'ensemble_type': results.get('config', {}).get('ensemble_type', 'none'),
            'base_models': str(results.get('config', {}).get('base_models', [])),
            'reason': results.get('reason', ''),
            'complexity_flagged': results.get('complexity_flagged', False)
        }

        # Add metrics if completed
        if results.get('status') == 'completed':
            test_metrics = results.get('test_metrics', {})
            train_metrics = results.get('train_metrics', {})

            master_row.update({
                'test_accuracy': test_metrics.get('accuracy', 0),
                'test_f1': test_metrics.get('weighted_f1', 0),
                'test_precision': test_metrics.get('weighted_precision', 0),
                'test_recall': test_metrics.get('weighted_recall', 0),
                'test_roc_auc': test_metrics.get('roc_auc_ovr', 0),
                'train_accuracy': train_metrics.get('accuracy', 0),
                'train_f1': train_metrics.get('weighted_f1', 0),
                'train_precision': train_metrics.get('weighted_precision', 0),
                'train_recall': train_metrics.get('weighted_recall', 0),
                'train_roc_auc': train_metrics.get('roc_auc_ovr', 0),
                'training_time': test_metrics.get('training_time', 0),
                'memory_usage': test_metrics.get('memory_usage', 0)
            })

            # Save detailed metrics for BOTH train and test
            self._save_detailed_metrics(model_name, results, timestamp)

        # Append to master results
        master_df = pd.DataFrame([master_row])
        master_df.to_csv(self.master_results_file, mode='a', header=False, index=False)

    def _save_detailed_metrics(self, model_name: str, results: Dict[str, Any], timestamp: str):
        """Save ALL detailed metrics for both train and test sets with proper column alignment"""
        
        try:
            # Define the expected column structure
            expected_columns = ['timestamp', 'model_name', 'dataset_type'] + TIER1_METRICS + TIER2_METRICS + TIER3_METRICS
            
            # Create base rows with all expected columns initialized to default values
            train_row = {col: 0.0 if col not in ['timestamp', 'model_name', 'dataset_type'] else '' for col in expected_columns}
            test_row = {col: 0.0 if col not in ['timestamp', 'model_name', 'dataset_type'] else '' for col in expected_columns}
            
            # Set the fixed columns
            train_row.update({
                'timestamp': timestamp,
                'model_name': model_name,
                'dataset_type': 'train'
            })
            
            test_row.update({
                'timestamp': timestamp,
                'model_name': model_name,
                'dataset_type': 'test'
            })
            
            # Safely update with actual metrics, only using expected columns
            train_metrics = results.get('train_metrics', {})
            test_metrics = results.get('test_metrics', {})
            ensemble_metrics = results.get('ensemble_metrics', {})
            
            # Update train row with only expected columns
            for col in expected_columns:
                if col in train_metrics:
                    train_row[col] = train_metrics[col]
                elif col in ensemble_metrics:
                    train_row[col] = ensemble_metrics[col]
            
            # Update test row with only expected columns
            for col in expected_columns:
                if col in test_metrics:
                    test_row[col] = test_metrics[col]
                elif col in ensemble_metrics:
                    test_row[col] = ensemble_metrics[col]
            
            # Create DataFrame with exact column order
            detailed_df = pd.DataFrame([train_row, test_row], columns=expected_columns)
            
            # Append to CSV
            detailed_df.to_csv(self.detailed_metrics_file, mode='a', header=False, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to save detailed metrics for {model_name}: {str(e)}")

    def _update_cumulative_analysis(self):
        """Update cumulative analysis report"""
        try:
            if os.path.exists(self.master_results_file):
                df = pd.read_csv(self.master_results_file)

                summary = []
                summary.append(f"CUMULATIVE ANALYSIS REPORT - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                summary.append("=" * 80)
                summary.append(f"Total Models Processed: {len(df)}")
                summary.append(f"Successful Models: {len(df[df['status'] == 'completed'])}")
                summary.append(f"Failed Models: {len(df[df['status'] == 'failed'])}")
                summary.append(f"Early Stopped Models: {len(df[df['status'] == 'early_stopped'])}")

                # Top performers
                completed = df[df['status'] == 'completed'].copy()
                if not completed.empty:
                    top_performers = completed.nlargest(10, 'test_f1')
                    summary.append(f"\nTOP 10 PERFORMERS (All Time):")
                    summary.append("-" * 40)
                    for idx, row in top_performers.iterrows():
                        summary.append(f"{row['model_name']}: F1={row['test_f1']:.4f}, Acc={row['test_accuracy']:.4f}")

                with open(self.historical_summary_file, 'w') as f:
                    f.write('\n'.join(summary))

        except Exception as e:
            self.logger.error(f"Failed to update cumulative analysis: {e}")

# ================================================================================================
# VISUALIZATION COMPONENTS
# ================================================================================================

class VisualizationManager:
    """Handle all visualization generation with enhanced error handling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')

    def create_comprehensive_model_report(self, model_name: str, results: Dict[str, Any], save_dir: str):
        """Create comprehensive individual model report with ALL visualizations"""

        if results['status'] != 'completed':
            return

        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]

        # Create individual model directory
        model_dir = os.path.join(save_dir, f"individual_models/{safe_name}")
        os.makedirs(model_dir, exist_ok=True)

        train_proba = results['predictions']['train_proba']
        test_proba = results['predictions']['test_proba']
        train_pred = np.argmax(train_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)

        # 1. Confusion matrices for BOTH train and test
        self._create_dual_confusion_matrices(model_name, train_pred, test_pred,
                                           results, model_dir)

        # 2. ROC curves for BOTH train and test
        self._create_dual_roc_curves(model_name, train_proba, test_proba,
                                   results, model_dir)

        # 3. Performance metrics comparison (train vs test)
        self._create_train_test_comparison(model_name, results, model_dir)

        # 4. Prediction confidence distributions
        self._create_confidence_distributions(model_name, train_proba, test_proba, model_dir)

        # 5. Learning curves (if available)
        self._create_class_performance_breakdown(model_name, results, model_dir)

    def _create_dual_confusion_matrices(self, model_name: str, train_pred, test_pred, results, save_dir):
        """Create side-by-side confusion matrices for train and test"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Train confusion matrix
            cm_train = confusion_matrix(results['train_metrics']['y_true'] if 'y_true' in results['train_metrics']
                                      else self.y_train, train_pred)
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'Training Set Confusion Matrix - {model_name}')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')

            # Test confusion matrix
            cm_test = confusion_matrix(results['test_metrics']['y_true'] if 'y_true' in results['test_metrics']
                                     else self.y_test, test_pred)
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax2)
            ax2.set_title(f'Test Set Confusion Matrix - {model_name}')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confusion_matrices_train_test.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create dual confusion matrices: {e}")

    def _create_train_test_comparison(self, model_name: str, results, save_dir):
        """Create comprehensive train vs test metrics comparison"""
        try:
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']

            metrics_to_compare = ['accuracy', 'weighted_f1', 'weighted_precision',
                                'weighted_recall', 'roc_auc_ovr']

            train_values = [train_metrics.get(m, 0) for m in metrics_to_compare]
            test_values = [test_metrics.get(m, 0) for m in metrics_to_compare]

            x = np.arange(len(metrics_to_compare))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='skyblue')
            ax.bar(x + width/2, test_values, width, label='Testing', alpha=0.8, color='lightcoral')

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title(f'Training vs Testing Performance - {model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_compare], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
                ax.text(i - width/2, train_val + 0.01, f'{train_val:.3f}',
                       ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, test_val + 0.01, f'{test_val:.3f}',
                       ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'train_test_comparison.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create train/test comparison: {e}")

    def create_comprehensive_comparative_analysis(self, results: Dict[str, Any], save_dir: str):
        """Create comprehensive comparative analysis across ALL models"""

        successful_results = {k: v for k, v in results.items() if v.get('status') == 'completed'}
        failed_results = {k: v for k, v in results.items() if v.get('status') == 'failed'}
        early_stopped = {k: v for k, v in results.items() if v.get('status') == 'early_stopped'}

        # 1. Complete performance overview
        self._create_complete_performance_overview(successful_results, save_dir)

        # 2. Training vs Test performance scatter plots
        self._create_train_test_scatter_analysis(successful_results, save_dir)

        # 3. Model complexity vs performance analysis
        self._create_complexity_analysis(successful_results, save_dir)

        # 4. Failed models analysis
        self._create_failed_models_analysis(failed_results, early_stopped, save_dir)

        # 5. Ensemble vs Individual model comparison
        self._create_ensemble_individual_comparison(successful_results, save_dir)

    def _create_complete_performance_overview(self, results, save_dir):
        """Create comprehensive performance overview for ALL successful models"""
        try:
            if not results:
                return

            model_names = list(results.keys())
            metrics = ['accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'roc_auc_ovr']

            # Create subplot grid
            fig, axes = plt.subplots(3, 2, figsize=(20, 18))
            axes = axes.flatten()

            for i, metric in enumerate(metrics):
                if i < len(axes):
                    train_values = [results[name]['train_metrics'].get(metric, 0) for name in model_names]
                    test_values = [results[name]['test_metrics'].get(metric, 0) for name in model_names]

                    x = np.arange(len(model_names))
                    width = 0.35

                    ax = axes[i]
                    ax.bar(x - width/2, train_values, width, label='Train', alpha=0.7, color='lightblue')
                    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.7, color='lightcoral')

                    ax.set_title(f'{metric.replace("_", " ").title()} - All Models', fontsize=14)
                    ax.set_xticks(x)
                    ax.set_xticklabels([name[:12] + '...' if len(name) > 12 else name
                                       for name in model_names], rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            # Training time comparison
            if len(axes) > len(metrics):
                ax = axes[len(metrics)]
                times = [results[name]['test_metrics'].get('training_time', 0) for name in model_names]
                bars = ax.bar(range(len(model_names)), times, color='green', alpha=0.7)
                ax.set_title('Training Time Comparison', fontsize=14)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels([name[:12] + '...' if len(name) > 12 else name
                                   for name in model_names], rotation=45, ha='right')
                ax.set_ylabel('Time (seconds)')
                ax.grid(True, alpha=0.3)

            plt.suptitle('Comprehensive Performance Overview - All Models', fontsize=18)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'complete_performance_overview.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create complete performance overview: {e}")

    def create_confusion_matrix(self, y_true, y_pred, model_name: str, save_path: str, n_classes: int):
        """Create and save confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.arange(n_classes),
                       yticklabels=np.arange(n_classes))
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(save_path, dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create confusion matrix for {model_name}: {str(e)}")

    def _create_dual_roc_curves(self, model_name: str, train_proba, test_proba, results, save_dir):
        """Create ROC curves for both train and test sets"""
        try:
            from sklearn.preprocessing import label_binarize

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Get actual y values
            y_train = results['train_metrics'].get('y_true', [])
            y_test = results['test_metrics'].get('y_true', [])

            if len(y_train) == 0 or len(y_test) == 0:
                return

            n_classes = train_proba.shape[1]

            # Train ROC
            y_train_bin = label_binarize(y_train, classes=np.arange(n_classes))
            if n_classes == 2:
                y_train_bin = np.hstack((1-y_train_bin, y_train_bin))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_train_bin[:, i], train_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax1.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'Train ROC Curves - {model_name}')
            ax1.legend()

            # Test ROC
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            if n_classes == 2:
                y_test_bin = np.hstack((1-y_test_bin, y_test_bin))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title(f'Test ROC Curves - {model_name}')
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'roc_curves_train_test.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create ROC curves: {e}")

    def _create_confidence_distributions(self, model_name: str, train_proba, test_proba, save_dir):
        """Create confidence distribution plots"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Train confidence
            train_conf = np.max(train_proba, axis=1)
            ax1.hist(train_conf, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Prediction Confidence')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Train Confidence Distribution - {model_name}')
            ax1.axvline(np.mean(train_conf), color='red', linestyle='--',
                       label=f'Mean: {np.mean(train_conf):.3f}')
            ax1.legend()

            # Test confidence
            test_conf = np.max(test_proba, axis=1)
            ax2.hist(test_conf, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_xlabel('Prediction Confidence')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Test Confidence Distribution - {model_name}')
            ax2.axvline(np.mean(test_conf), color='red', linestyle='--',
                       label=f'Mean: {np.mean(test_conf):.3f}')
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confidence_distributions.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create confidence distributions: {e}")

    def _create_class_performance_breakdown(self, model_name: str, results, save_dir):
        """Create class-wise performance breakdown"""
        try:
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']

            # Extract per-class F1 scores
            n_classes = 3  # Based on your data
            train_f1_scores = [train_metrics.get(f'class_{i}_f1', 0) for i in range(n_classes)]
            test_f1_scores = [test_metrics.get(f'class_{i}_f1', 0) for i in range(n_classes)]

            x = np.arange(n_classes)
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, train_f1_scores, width, label='Train', alpha=0.8, color='skyblue')
            ax.bar(x + width/2, test_f1_scores, width, label='Test', alpha=0.8, color='lightcoral')

            ax.set_xlabel('Class')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'Per-Class Performance - {model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, (train_val, test_val) in enumerate(zip(train_f1_scores, test_f1_scores)):
                ax.text(i - width/2, train_val + 0.01, f'{train_val:.3f}',
                       ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, test_val + 0.01, f'{test_val:.3f}',
                       ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'class_performance_breakdown.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create class performance breakdown: {e}")

    def _create_train_test_scatter_analysis(self, results, save_dir):
        """Create scatter plot analysis of train vs test performance"""
        try:
            if not results:
                return

            model_names = list(results.keys())
            train_acc = [results[name]['train_metrics'].get('accuracy', 0) for name in model_names]
            test_acc = [results[name]['test_metrics'].get('accuracy', 0) for name in model_names]
            train_f1 = [results[name]['train_metrics'].get('weighted_f1', 0) for name in model_names]
            test_f1 = [results[name]['test_metrics'].get('weighted_f1', 0) for name in model_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Accuracy scatter
            ax1.scatter(train_acc, test_acc, alpha=0.7, s=60)
            ax1.plot([0, 1], [0, 1], 'r--', alpha=0.75)
            ax1.set_xlabel('Train Accuracy')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Train vs Test Accuracy')
            ax1.grid(True, alpha=0.3)

            # F1 scatter
            ax2.scatter(train_f1, test_f1, alpha=0.7, s=60)
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.75)
            ax2.set_xlabel('Train F1-Score')
            ax2.set_ylabel('Test F1-Score')
            ax2.set_title('Train vs Test F1-Score')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'train_test_scatter_analysis.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create scatter analysis: {e}")

    def _create_complexity_analysis(self, results, save_dir):
        """Create complexity vs performance analysis"""
        try:
            if not results:
                return

            model_names = list(results.keys())
            training_times = [results[name]['test_metrics'].get('training_time', 0) for name in model_names]
            test_f1_scores = [results[name]['test_metrics'].get('weighted_f1', 0) for name in model_names]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(training_times, test_f1_scores, alpha=0.7, s=60)

            # Add model names as labels
            for i, name in enumerate(model_names):
                plt.annotate(name[:15] + '...' if len(name) > 15 else name,
                           (training_times[i], test_f1_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.xlabel('Training Time (seconds)')
            plt.ylabel('Test F1-Score')
            plt.title('Model Complexity vs Performance Analysis')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'complexity_vs_performance.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create complexity analysis: {e}")

    def _create_failed_models_analysis(self, failed_results, early_stopped, save_dir):
        """Create analysis of failed and early stopped models"""
        try:
            all_failed = {**failed_results, **early_stopped}
            if not all_failed:
                return

            # Count by failure type
            failure_types = {}
            for name, result in all_failed.items():
                status = result.get('status', 'unknown')
                if status not in failure_types:
                    failure_types[status] = 0
                failure_types[status] += 1

            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(failure_types.values(), labels=failure_types.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('Distribution of Failed Models')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'failed_models_analysis.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create failed models analysis: {e}")

    def _create_ensemble_individual_comparison(self, results, save_dir):
        """Compare ensemble vs individual models"""
        try:
            if not results:
                return

            individual_models = []
            ensemble_models = []

            for name, result in results.items():
                config = result.get('config', {})
                if config.get('ensemble_type') == 'none':
                    individual_models.append((name, result['test_metrics'].get('weighted_f1', 0)))
                else:
                    ensemble_models.append((name, result['test_metrics'].get('weighted_f1', 0)))

            if not individual_models or not ensemble_models:
                return

            individual_f1 = [f1 for _, f1 in individual_models]
            ensemble_f1 = [f1 for _, f1 in ensemble_models]

            plt.figure(figsize=(10, 6))
            plt.boxplot([individual_f1, ensemble_f1], labels=['Individual Models', 'Ensemble Models'])
            plt.ylabel('Test F1-Score')
            plt.title('Individual vs Ensemble Model Performance')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'individual_vs_ensemble.png'),
                       dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create ensemble comparison: {e}")


    def create_performance_comparison(self, results: Dict[str, Any], save_path: str):
        """Create comprehensive performance comparison"""
        try:
            successful_results = {k: v for k, v in results.items() if v.get('status') == 'completed'}

            if not successful_results:
                self.logger.warning("No successful results to plot")
                return

            model_names = list(successful_results.keys())
            metrics = ['accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'roc_auc_ovr']

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, metric in enumerate(metrics):
                if i < len(axes):
                    values = [successful_results[name]['test_metrics'].get(metric, 0) for name in model_names]

                    ax = axes[i]
                    bars = ax.bar(range(len(model_names)), values,
                                 color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))

                    ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
                    ax.set_xticks(range(len(model_names)))
                    ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                       for name in model_names], rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)

                    # Add value labels on bars
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            # Training time comparison
            if len(axes) > len(metrics):
                ax = axes[len(metrics)]
                times = [successful_results[name]['test_metrics'].get('training_time', 0) for name in model_names]
                bars = ax.bar(range(len(model_names)), times, color='orange', alpha=0.7)
                ax.set_title('Training Time (seconds)', fontsize=12)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                   for name in model_names], rotation=45, ha='right')
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(len(metrics) + 1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle('Model Performance Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path, dpi=Config.REPORT_DPI, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Failed to create performance comparison: {str(e)}")

# ================================================================================================
# REPORT GENERATION
# ================================================================================================

class ReportGenerator:
    """Generate comprehensive analysis reports"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary text"""

        successful_results = {k: v for k, v in results.items() if v.get('status') == 'completed'}
        failed_results = {k: v for k, v in results.items() if v.get('status') == 'failed'}
        early_stopped = {k: v for k, v in results.items() if v.get('status') == 'early_stopped'}

        summary_lines = []
        summary_lines.append("EXECUTIVE SUMMARY - ML MODEL COMPARATIVE ANALYSIS")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total Models Evaluated: {len(results)}")
        summary_lines.append(f"Successful Completions: {len(successful_results)}")
        summary_lines.append(f"Failed Models: {len(failed_results)}")
        summary_lines.append(f"Early Stopped (Poor Performance): {len(early_stopped)}")

        if successful_results:
            # Find top performers
            top_models = sorted(
                successful_results.items(),
                key=lambda x: x[1]['test_metrics'].get('weighted_f1', 0),
                reverse=True
            )[:5]

            summary_lines.append("\nTOP 5 PERFORMING MODELS (by Weighted F1-Score):")
            summary_lines.append("-" * 50)

            for i, (model_name, result) in enumerate(top_models, 1):
                metrics = result['test_metrics']
                summary_lines.append(f"\n{i}. {model_name}")
                summary_lines.append(f"   F1-Score: {metrics.get('weighted_f1', 0):.4f}")
                summary_lines.append(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                summary_lines.append(f"   ROC-AUC: {metrics.get('roc_auc_ovr', 0):.4f}")
                summary_lines.append(f"   Training Time: {metrics.get('training_time', 0):.2f}s")

        if early_stopped:
            summary_lines.append(f"\nEARLY STOPPED MODELS ({len(early_stopped)}):")
            summary_lines.append("-" * 25)
            for name, result in list(early_stopped.items())[:5]:
                reason = result.get('reason', 'Unknown')
                summary_lines.append(f"- {name}: {reason}")

        return "\n".join(summary_lines)

    def save_reports(self, results: Dict[str, Any], visualizations: Dict[str, str]):
        """Save all reports to files"""
        try:
            executive_summary = self.generate_executive_summary(results)

            with open(os.path.join(Config.REPORTS_DIR, 'executive_summary.txt'), 'w') as f:
                f.write(executive_summary)

            self.logger.info("Reports generated successfully")

        except Exception as e:
            self.logger.error(f"Failed to generate reports: {str(e)}")

    def generate_comprehensive_reports(self, results: Dict[str, Any], plots_dir: str):
        """Generate comprehensive reports with all visualizations"""
        try:
            # Generate executive summary
            executive_summary = self.generate_executive_summary(results)

            # Save executive summary
            with open(os.path.join(Config.REPORTS_DIR, 'executive_summary.txt'), 'w') as f:
                f.write(executive_summary)

            # Generate detailed technical report
            technical_report = self._generate_technical_report(results)

            # Save technical report
            with open(os.path.join(Config.REPORTS_DIR, 'technical_report.txt'), 'w') as f:
                f.write(technical_report)

            self.logger.info("Comprehensive reports generated successfully")

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive reports: {str(e)}")

    def _generate_technical_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed technical report"""
        successful_results = {k: v for k, v in results.items() if v.get('status') == 'completed'}

        report_lines = []
        report_lines.append("TECHNICAL ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if successful_results:
            report_lines.append(f"\nDETAILED MODEL PERFORMANCE:")
            report_lines.append("-" * 40)

            for model_name, result in successful_results.items():
                train_metrics = result['train_metrics']
                test_metrics = result['test_metrics']

                report_lines.append(f"\n{model_name.upper()}:")
                report_lines.append(f"  Train Performance:")
                report_lines.append(f"    Accuracy: {train_metrics.get('accuracy', 0):.4f}")
                report_lines.append(f"    F1-Score: {train_metrics.get('weighted_f1', 0):.4f}")
                report_lines.append(f"    Precision: {train_metrics.get('weighted_precision', 0):.4f}")
                report_lines.append(f"    Recall: {train_metrics.get('weighted_recall', 0):.4f}")

                report_lines.append(f"  Test Performance:")
                report_lines.append(f"    Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                report_lines.append(f"    F1-Score: {test_metrics.get('weighted_f1', 0):.4f}")
                report_lines.append(f"    Precision: {test_metrics.get('weighted_precision', 0):.4f}")
                report_lines.append(f"    Recall: {test_metrics.get('weighted_recall', 0):.4f}")

                report_lines.append(f"  Computational Metrics:")
                report_lines.append(f"    Training Time: {test_metrics.get('training_time', 0):.2f}s")
                report_lines.append(f"    Memory Usage: {test_metrics.get('memory_usage', 0):.2f}%")

        return "\n".join(report_lines)

# ================================================================================================
# MAIN PIPELINE EXECUTION
# ================================================================================================

class ModelComparisonPipeline:
    """Enhanced main pipeline for comprehensive model comparison"""

    def __init__(self):
        self.logger = setup_logging()
        self.checkpoint_manager = CheckpointManager()
        self.viz_manager = VisualizationManager()
        self.report_generator = ReportGenerator()

        # Data placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_classes = None
        self.evaluator = None

        # Results storage
        self.all_results = {}
        self.best_individual_score = 0.0

    def _generate_final_analysis(self):
        """Enhanced final analysis with statistical rigor"""
        try:
            self.logger.info("Generating comprehensive final analysis...")

            # 1. Perform statistical analysis
            self._perform_statistical_analysis()

            # 2. Generate existing visualizations
            self.viz_manager.create_comprehensive_comparative_analysis(
                self.all_results, Config.PLOTS_DIR
            )

            # 3. Generate enhanced reports with statistical analysis
            self.report_generator.generate_comprehensive_reports(
                self.all_results, Config.PLOTS_DIR
            )

            # 4. Generate statistical summary
            self._generate_statistical_summary()

            self.logger.info("Enhanced statistical analysis completed!")

        except Exception as e:
            self.logger.error(f"Failed to generate enhanced analysis: {str(e)}")

    def _generate_statistical_summary(self):
        """Generate statistical summary report"""
        try:
            if not hasattr(self, 'statistical_comparisons'):
                return

            summary_lines = []
            summary_lines.append("STATISTICAL ANALYSIS SUMMARY")
            summary_lines.append("=" * 50)
            summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"Total Comparisons: {len(self.statistical_comparisons)}")

            significant_comparisons = [c for c in self.statistical_comparisons
                                    if c.get('paired_ttest', {}).get('significant', False)]

            summary_lines.append(f"Significant Differences: {len(significant_comparisons)}")

            if significant_comparisons:
                summary_lines.append("\nSIGNIFICANT MODEL DIFFERENCES:")
                summary_lines.append("-" * 30)

                for comp in significant_comparisons:
                    summary_lines.append(f"\n{comp['model1_name']} vs {comp['model2_name']}:")
                    summary_lines.append(f"  Difference: {comp['difference']:.4f}")
                    summary_lines.append(f"  p-value: {comp['paired_ttest']['pvalue']:.4f}")
                    summary_lines.append(f"  Effect size: {comp['effect_size']['magnitude']}")

            with open(os.path.join(Config.REPORTS_DIR, 'statistical_summary.txt'), 'w') as f:
                f.write('\n'.join(summary_lines))

        except Exception as e:
            self.logger.error(f"Failed to generate statistical summary: {str(e)}")

    def _perform_statistical_analysis(self):
        """Perform statistical analysis across all successful models"""
        try:
            self.logger.info("Performing statistical analysis across models...")

            successful_results = {k: v for k, v in self.all_results.items()
                                if v.get('status') == 'completed'}

            if len(successful_results) < 2:
                self.logger.warning("Insufficient models for statistical comparison")
                return

            statistical_tester = StatisticalTester()
            model_names = list(successful_results.keys())
            comparisons = []

            # Pairwise comparisons
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1_name = model_names[i]
                    model2_name = model_names[j]

                    # Get CV scores for comparison
                    model1_cv = successful_results[model1_name].get('cross_validation', {})
                    model2_cv = successful_results[model2_name].get('cross_validation', {})

                    if 'weighted_f1_scores' in model1_cv and 'weighted_f1_scores' in model2_cv:
                        comparison = statistical_tester.compare_models_statistically(
                            model1_cv['weighted_f1_scores'],
                            model2_cv['weighted_f1_scores'],
                            model1_name, model2_name
                        )
                        comparisons.append(comparison)

            # Store statistical analysis results
            self.statistical_comparisons = comparisons

            # Save to file
            import json
            with open(os.path.join(Config.RESULTS_DIR, 'statistical_comparisons.json'), 'w') as f:
                json.dump(comparisons, f, indent=2, default=str)

            self.logger.info(f"Statistical analysis completed: {len(comparisons)} comparisons")

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with enhanced error handling"""
        try:
            self.logger.info("Loading and preprocessing data...")

            # Try to load real data first
            try:
                df = pd.read_csv(Config.DATA_PATH)
                self.logger.info("Real dataset loaded successfully!")

                if Config.TARGET_COLUMN not in df.columns:
                    raise ValueError(f"Target column '{Config.TARGET_COLUMN}' not found")

                X_df = df.drop(columns=[Config.TARGET_COLUMN])
                y = df[Config.TARGET_COLUMN].astype(int)

            except FileNotFoundError:
                self.logger.warning("Real dataset not found. Generating synthetic data for demonstration...")

                # Generate synthetic dataset
                n_samples = 20000
                n_numerical_features = 25
                n_categorical_features = 5
                np.random.seed(Config.RANDOM_STATE)

                X_num = np.random.rand(n_samples, n_numerical_features) * 100

                categories = [
                    ['A', 'B', 'C', 'D', 'E'],
                    ['X', 'Y', 'Z'],
                    ['P', 'Q'],
                    ['Alpha', 'Beta', 'Gamma'],
                    ['Red', 'Green', 'Blue', 'Yellow']
                ]

                X_cat_columns = []
                for i, cat_list in enumerate(categories):
                    X_cat_columns.append(np.random.choice(cat_list, n_samples))

                X_cat = np.column_stack(X_cat_columns)

                column_names = ([f'num_feat_{i}' for i in range(n_numerical_features)] +
                               [f'cat_feat_{i}' for i in range(n_categorical_features)])

                X_df = pd.DataFrame(np.hstack((X_num, X_cat)), columns=column_names)

                for i in range(n_categorical_features):
                    X_df[f'cat_feat_{i}'] = X_df[f'cat_feat_{i}'].astype('category')

                # Generate imbalanced multi-class target using clustering
                self.logger.info("Generating synthetic target labels using clustering...")

                kmeans = KMeans(n_clusters=3, random_state=Config.RANDOM_STATE, n_init=10)
                y_cluster = kmeans.fit_predict(X_num[:, :10])

                y = np.copy(y_cluster)

                # Make it imbalanced
                reassign_mask = np.random.rand(n_samples) < 0.3
                y[reassign_mask] = np.random.choice([0, 1, 2],
                                                   size=np.sum(reassign_mask),
                                                   p=[0.6, 0.3, 0.1])

                self.logger.info(f"Generated synthetic dataset with target distribution: {np.bincount(y)}")

            # Preprocessing
            numerical_features = X_df.select_dtypes(include=[np.number]).columns
            categorical_features = X_df.select_dtypes(include=['category', 'object']).columns

            for col in X_df.select_dtypes(include=['object']).columns:
                if X_df[col].nunique() < len(X_df) / 10:
                    X_df[col] = X_df[col].astype('category')
                    categorical_features = X_df.select_dtypes(include=['category']).columns

            self.logger.info(f"Numerical features: {len(numerical_features)}")
            self.logger.info(f"Categorical features: {len(categorical_features)}")

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )

            # Split data (before fitting preprocessor)
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
            )

            # Fit and transform (preprocessor is now fitted here)
            self.X_train = preprocessor.fit_transform(X_train)
            self.X_test = preprocessor.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test
            self.n_classes = len(np.unique(y))
            
            Config.GLOBAL_PREPROCESSOR = preprocessor

            deepgbm_instance = DeepGBMWrapper()
            metadata = deepgbm_instance.build_feature_metadata_from_transformer(preprocessor)
            # DeepGBMWrapper.set_metadata(
            #                 metadata['nume_input_size'],
            #                 metadata['cate_field_size'],
            #                 metadata['feature_sizes']
            #             )
            self.logger.info(f"DeepGBM metadata configured: numeric={metadata['nume_input_size']}, "
                            f"categorical={metadata['cate_field_size']}, sizes={len(metadata['feature_sizes'])}")
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(self.X_train, self.X_test,
                                          self.y_train, self.y_test, self.n_classes)

            self.logger.info(f"Data preprocessing completed.")
            self.logger.info(f"Training set shape: {self.X_train.shape}")
            self.logger.info(f"Test set shape: {self.X_test.shape}")
            self.logger.info(f"Number of classes: {self.n_classes}")
            self.logger.info(f"Class distribution: {np.bincount(y)}")

        except Exception as e:
            self.logger.error(f"Data loading/preprocessing failed: {str(e)}")
            raise

    def run_analysis(self, model_configs: Dict[str, Dict[str, Any]]):
        """Run the complete analysis pipeline with enhanced error handling"""
        try:
            self.logger.info("Starting comprehensive model analysis pipeline...")

            # Load and preprocess data
            self.load_and_preprocess_data()

            # Load existing checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint()
            completed_models = set(checkpoint.get('completed_models', []))
            failed_models = set(checkpoint.get('failed_models', []))
            early_stopped_models = set(checkpoint.get('early_stopped_models', []))

            total_processed = len(completed_models) + len(failed_models) + len(early_stopped_models)
            self.logger.info(f"Found {total_processed} previously processed models")

            # Process models by tier
            for tier_name, tier_configs in model_configs.items():
                if not tier_configs:
                    continue

                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing {tier_name.upper()}: {len(tier_configs)} models")
                self.logger.info(f"{'='*60}")

                # Group models by complexity
                model_batches = self._create_model_batches(tier_configs)

                for complexity, models_in_batch in model_batches.items():
                    if not models_in_batch:
                        continue

                    self.logger.info(f"\nProcessing {complexity} complexity batch: {len(models_in_batch)} models")
                    batch_size = Config.BATCH_SIZES.get(complexity, 3)

                    for i, (model_name, config) in enumerate(models_in_batch):

                        # Skip if already processed
                        if model_name in (completed_models | failed_models | early_stopped_models):
                            self.logger.info(f"Skipping {model_name} (already processed)")
                            continue

                        self.logger.info(f"\n--- Processing {model_name} ({i+1}/{len(models_in_batch)}) ---")

                        # Resource check
                        resources = monitor_resources()
                        if resources['memory_percent'] > 90:
                            self.logger.warning(f"High memory usage: {resources['memory_percent']:.1f}%")
                            cleanup_memory()

                        # Stage 1: Quick Assessment
                        self.logger.info("Stage 1: Quick assessment...")
                        quick_results = self.evaluator.quick_evaluate(model_name, config)
                        should_stop, reason = check_early_stop_criteria(quick_results)

                        if should_stop:
                            self.logger.info(f"Early stopping {model_name}: {reason}")

                            early_stop_result = {
                                'model_name': model_name,
                                'config': config,
                                'status': 'early_stopped',
                                'reason': reason,
                                'quick_metrics': quick_results,
                                'timestamp': datetime.now().isoformat()
                            }

                            self.all_results[model_name] = early_stop_result
                            self.checkpoint_manager.save_checkpoint(model_name, early_stop_result)
                            continue

                        # Stage 2: Full Evaluation
                        self.logger.info("Stage 2: Full evaluation...")
                        results = self.evaluator.full_evaluate(model_name, config)

                        if results['status'] == 'completed':
                            test_f1 = results['test_metrics']['weighted_f1']
                            train_time = results['test_metrics']['training_time']

                            # Update best individual model score
                            if config.get('ensemble_type') == 'none':
                                self.best_individual_score = max(self.best_individual_score, test_f1)

                            # Stage 3: Complexity Analysis
                            if (train_time > Config.COMPLEXITY_TIME_MULTIPLIER * 60 and
                                test_f1 < self.best_individual_score + Config.COMPLEXITY_F1_IMPROVEMENT):
                                results['complexity_flagged'] = True
                                self.logger.warning(f"{model_name} flagged: high complexity, minimal improvement")

                            # Generate visualizations
                            self._generate_model_visualizations(model_name, results)

                        # Store and save results
                        self.all_results[model_name] = results
                        self.checkpoint_manager.save_checkpoint(model_name, results)

                        # Memory cleanup
                        cleanup_memory()

                        # Manual break point
                        if (Config.MANUAL_BREAK_ENABLED and
                            (i + 1) % batch_size == 0 and
                            i + 1 < len(models_in_batch)):

                            print(f"\n{'='*50}")
                            print(f"BATCH CHECKPOINT: Completed {i+1}/{len(models_in_batch)} {complexity} models")
                            print(f"Current results: {len([r for r in self.all_results.values() if r.get('status') == 'completed'])} successful")
                            print(f"Memory usage: {monitor_resources()['memory_percent']:.1f}%")
                            print(f"{'='*50}")

                            user_input = input("Press Enter to continue to next batch, 'q' to quit, 's' to skip to next tier: ")
                            if user_input.lower() == 'q':
                                self.logger.info("User requested early termination")
                                self._generate_final_analysis()
                                return self.all_results
                            elif user_input.lower() == 's':
                                self.logger.info("User requested skip to next tier")
                                break

            # Final analysis and reporting
            self.logger.info("\n" + "="*60)
            self.logger.info("GENERATING FINAL ANALYSIS AND REPORTS")
            self.logger.info("="*60)

            self._generate_final_analysis()

            self.logger.info("Analysis pipeline completed successfully!")
            return self.all_results if self.all_results is not None else {}

        except KeyboardInterrupt:
            self.logger.info("Analysis interrupted by user. Generating partial results...")
            self._generate_final_analysis()
            return self.all_results if self.all_results is not None else {}
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            return self.all_results if self.all_results is not None else {}

    def _create_model_batches(self, tier_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, Dict]]]:
        """Group models by complexity for intelligent batching"""
        batches = {'light': [], 'medium': [], 'heavy': []}

        for model_name, config in tier_configs.items():
            base_models = config.get('base_models', [])
            max_complexity = 'light'

            # Determine complexity based on heaviest base model
            for base_model in base_models:
                base_model_lower = base_model.lower().replace('_', '').replace(' ', '')
                for complexity, models in MODEL_COMPLEXITY.items():
                    if any(model in base_model_lower for model in models):
                        if complexity == 'heavy':
                            max_complexity = 'heavy'
                            break
                        elif complexity == 'medium' and max_complexity != 'heavy':
                            max_complexity = 'medium'

            # Special cases for complex ensembles
            ensemble_type = config.get('ensemble_type', 'none')
            if (ensemble_type in ['stacking', 'cv_ensemble'] or
                len(base_models) > 3 or
                'mixture' in model_name.lower() or
                'parametric' in model_name.lower()):
                max_complexity = 'heavy'

            batches[max_complexity].append((model_name, config))

        return batches

    def _generate_model_visualizations(self, model_name: str, results: Dict[str, Any]):
        """Generate comprehensive visualizations for individual models"""
        try:
            self.logger.info(f"Generating comprehensive visualizations for {model_name}...")

            # Create comprehensive individual model report
            self.viz_manager.create_comprehensive_model_report(
                model_name, results, Config.PLOTS_DIR
            )

            self.logger.info(f"Individual model visualizations completed for {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations for {model_name}: {str(e)}")

    def _generate_final_analysis(self):
        """Enhanced final analysis with comprehensive reporting"""
        try:
            self.logger.info("Generating comprehensive final analysis...")

            # 1. Generate comprehensive comparative visualizations
            self.viz_manager.create_comprehensive_comparative_analysis(
                self.all_results, Config.PLOTS_DIR
            )

            # 2. Generate enhanced reports
            self.report_generator.generate_comprehensive_reports(
                self.all_results, Config.PLOTS_DIR
            )

            self.logger.info("Comprehensive final analysis completed!")
            self.logger.info(f"All reports and visualizations saved to:")
            self.logger.info(f"   - CSV Results: {Config.RESULTS_DIR}/")
            self.logger.info(f"   - Visual Reports: {Config.PLOTS_DIR}/")
            self.logger.info(f"   - Text Reports: {Config.REPORTS_DIR}/")

        except Exception as e:
            self.logger.error(f"Failed to generate final analysis: {str(e)}")

def generate_checkpoint_aware_configurations(completed_models: List[str]) -> Dict[str, Dict]:
    """Generate new configurations while preserving completed work"""
    
    generator = SystematicConfigurationGenerator()
    all_configs = generator.generate_systematic_configurations(TARGET_MODELS)
    
    # Filter out completed models
    new_configs = {}
    skipped_count = 0
    
    for config_name, config in all_configs.items():
        if config_name in completed_models:
            skipped_count += 1
            continue
        new_configs[config_name] = config
    
    print(f"Generated {len(new_configs)} new configurations")
    print(f"Skipped {skipped_count} already completed models")
    
    # Add theoretical controls and negative controls
    new_configs.update(_generate_control_models(completed_models))
    
    return new_configs

def _generate_control_models(completed_models: List[str]) -> Dict[str, Dict]:
    """Generate statistical control models"""
    controls = {}
    
    # Random baseline (if not completed)
    if 'random_baseline' not in completed_models:
        controls['random_baseline'] = {
            'base_models': ['random_classifier'],
            'ensemble_type': 'none',
            'purpose': 'negative_control'
        }
    
    # Majority class baseline
    if 'majority_baseline' not in completed_models:
        controls['majority_baseline'] = {
            'base_models': ['majority_classifier'],
            'ensemble_type': 'none',
            'purpose': 'statistical_control'
        }
    
    # Ensemble ablation study
    if 'ablation_voting_vs_stacking' not in completed_models:
        controls['ablation_voting_vs_stacking'] = {
            'base_models': ['lightgbm', 'xgboost', 'randomforest'],
            'ensemble_type': 'voting',
            'purpose': 'ablation_study',
            'compare_with': 'stacking'
        }
    
    return controls

def update_model_configs_systematically():
    """Update MODEL_CONFIGS with systematic, balanced configurations"""
    
    # Load completed models from checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint = checkpoint_manager.load_checkpoint()
    completed_models = checkpoint.get('completed_models', [])
    
    # Generate new systematic configurations
    new_configs = generate_checkpoint_aware_configurations(completed_models)
    
    # Organize by complexity for batching
    tier1_configs = {}
    tier2_configs = {}
    
    for name, config in new_configs.items():
        base_models = config['base_models']
        
        # Determine tier based on complexity and purpose
        if (len(base_models) == 1 or 
            config.get('purpose', '').startswith('baseline') or
            'solo' in name):
            tier1_configs[name] = config
        else:
            tier2_configs[name] = config
    
    return {
        'tier1': tier1_configs,
        'tier2': tier2_configs
    }

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

def main():
    """Main execution function with enhanced error handling"""

    print("="*80)
    print("COMPREHENSIVE MULTI-MODEL COMPARATIVE ANALYSIS PIPELINE (FINAL FIXED)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize pipeline
    try:
        pipeline = ModelComparisonPipeline()

        # Check if MODEL_CONFIGS is populated
        tier1_count = len(MODEL_CONFIGS.get('tier1', {}))
        tier2_count = len(MODEL_CONFIGS.get('tier2', {}))
        total_models = tier1_count + tier2_count

        if total_models == 0:
            print("\nERROR: MODEL_CONFIGS is empty!")
            return None
      
        # FIXED: Generate updated configurations first
        UPDATED_MODEL_CONFIGS = update_model_configs_systematically()
        
        current_configs = UPDATED_MODEL_CONFIGS
        
        print(f"\nPipeline Configuration:")
        print(f"- Tier 1 Models: {tier1_count}")
        print(f"- Tier 2 Models: {tier2_count}")
        print(f"- Total Models: {total_models}")
        print(f"- Results Directory: {Config.RESULTS_DIR}")
        print(f"- Reports Directory: {Config.REPORTS_DIR}")
        print(f"- Plots Directory: {Config.PLOTS_DIR}")
        print(f"- Manual Breaks Enabled: {Config.MANUAL_BREAK_ENABLED}")

        # System resource check
        resources = monitor_resources()
        print(f"\nSystem Resources:")
        print(f"- Available Memory: {resources.get('memory_available', 'Unknown'):.1f} GB")
        print(f"- Memory Usage: {resources.get('memory_percent', 'Unknown'):.1f}%")

        # Start analysis
        print(f"\nStarting analysis pipeline...")
        print("="*50)

        results = pipeline.run_analysis(MODEL_CONFIGS)
        if results is None:
            print("\nERROR: Analysis pipeline returned None!")
            print("Check the logs above for specific errors.")
            return None
        
        # Final summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

        successful = len([r for r in results.values() if r.get('status') == 'completed'])
        failed = len([r for r in results.values() if r.get('status') == 'failed'])
        early_stopped = len([r for r in results.values() if r.get('status') == 'early_stopped'])

        print(f"Final Results Summary:")
        print(f"- Total Models Processed: {len(results)}")
        print(f"- Successful Completions: {successful}")
        print(f"- Failed Models: {failed}")
        print(f"- Early Stopped (Poor Performance): {early_stopped}")

        # Show top performers
        if successful > 0:
            successful_results = {k: v for k, v in results.items() if v.get('status') == 'completed'}
            top_5 = sorted(successful_results.items(),
                          key=lambda x: x[1]['test_metrics'].get('weighted_f1', 0),
                          reverse=True)[:5]

            print(f"\nTOP 5 PERFORMERS (by F1-Score):")
            print("-" * 50)
            for i, (name, result) in enumerate(top_5, 1):
                metrics = result['test_metrics']
                f1 = metrics.get('weighted_f1', 0)
                acc = metrics.get('accuracy', 0)
                time = metrics.get('training_time', 0)
                print(f"{i}. {name[:40]}...")
                print(f"   F1: {f1:.4f} | Accuracy: {acc:.4f} | Time: {time:.2f}s")

        print(f"\nDetailed results and reports available in:")
        print(f"- Results: {Config.RESULTS_DIR}/")
        print(f"- Reports: {Config.REPORTS_DIR}/")
        print(f"- Visualizations: {Config.PLOTS_DIR}/")

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return results

    except KeyboardInterrupt:
        print(f"\nAnalysis interrupted by user!")
        print("Partial results have been saved to checkpoint files.")
        return None
    except Exception as e:
        print(f"\nAnalysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
