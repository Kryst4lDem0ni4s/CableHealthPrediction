import os
import re
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
warnings.filterwarnings("ignore")

# Configuration
RESULTS_DIR = 'analysis_res'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Small-font, dense-plots defaults for 600+ models
plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 150,
    'font.size': 5,  # Reduced for density
    'axes.titlesize': 6,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.constrained_layout.use': True
})

# Helpers: parsing, classification, labels (from your code, unchanged except minor fixes)
BASE_ALGO_TOKENS = {
    'lightgbm', 'xgboost', 'catboost', 'randomforest',
    'adaboost', 'gradientboost', 'gradientclassifier',
    'logisticregression', 'ridge', 'sgd', 'linearsvm', 'linearmodel',
    'advancedsvm', 'naivebayes', 'ngboost',
    'knn', 'neuralnetwork', 'tabflex', 'deepgbm',
    'mixture_of_experts', 'mixtureofexperts'
}
ENSEMBLE_HINTS = {
    'stacking_ridge': 'stacking',
    'stackingridge': 'stacking',
    'stacking': 'stacking',
    'parametric_ensemble': 'parametric',
    'parametric': 'parametric',
    'cv_ensemble': 'cv',
    'cvensemble': 'cv',
    'weighted': 'weighted',
    'voting': 'voting',
    'ensemble': 'voting'  # generic "ensemble" falls back to voting baseline
}
IGNORE_TOKENS = {
    'solo', 'ensemble', 'voting', 'weighted', 'stacking', 'stackingridge',
    'parametric', 'parametricensemble', 'cv', 'cvensemble', 'cv_ensemble',
    'ridge', 'classifier', 'linearmodel'  # keep in ignore for suffix cues
}
FAMILY_MAP = {
    # boosting & tree
    'lightgbm': 'tree_boosting', 'xgboost': 'tree_boosting',
    'catboost': 'tree_boosting', 'adaboost': 'traditional_boosting',
    'gradientboost': 'traditional_boosting', 'gradientclassifier': 'traditional_boosting',
    'randomforest': 'tree_bagging',
    # linear
    'logisticregression': 'linear_models', 'ridge': 'linear_models',
    'sgd': 'linear_models', 'linearsvm': 'linear_models', 'linearmodel': 'linear_models',
    # kernel
    'advancedsvm': 'kernel_methods',
    # probabilistic
    'naivebayes': 'probabilistic', 'ngboost': 'probabilistic_boosting',
    # instance/neural/advanced
    'knn': 'instance_based',
    'neuralnetwork': 'neural_networks', 'tabflex': 'neural_networks',
    'deepgbm': 'advanced_ensemble', 'mixture_of_experts': 'advanced_ensemble',
    'mixtureofexperts': 'advanced_ensemble'
}
TEST_METRICS = [
    'accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall',
    'roc_auc_ovr', 'training_time', 'memory_usage', 'per_class_f1',
    'cohens_kappa', 'matthews_corrcoef', 'log_loss',
    'confidence_mean', 'confidence_std', 'cv_stability', 'prediction_entropy'
]
TIER3_OPTIONAL = ['individual_contribution', 'ensemble_diversity', 'marginal_improvement', 'computational_overhead']

# Data loading with duplicate handling (keep latest by timestamp)
def load_data(file_path='C:/Users/Khwaish/.vscode/CableHealthExp/resnew/analysis_results/detailed_metrics.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # Sort by timestamp descending and drop duplicates, keeping latest
    df = df.sort_values(by='timestamp', ascending=False)
    df = df.drop_duplicates(subset=['model_name', 'dataset_type'], keep='first')
    # Coerce numerics
    for c in df.columns:
        if c not in ['timestamp', 'model_name', 'dataset_type']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# Model/ensemble parsing & classification
def infer_ensemble_type(name: str) -> str:
    name_l = name.lower()
    for key, etype in ENSEMBLE_HINTS.items():
        if key in name_l:
            return etype
    if 'solo' in name_l:
        return 'none'
    return 'none'

def infer_base_algos(name: str) -> List[str]:
    name_l = name.lower()
    tokens = re.split(r'[_\-]+', name_l)
    tokens = [t if t != 'moe' else 'mixture_of_experts' for t in tokens]
    base = [t for t in tokens if t in BASE_ALGO_TOKENS]
    seen = set()
    base_clean = [b for b in base if not (b in seen or seen.add(b))]
    if not base_clean and 'solo' in name_l:
        return ['solo']
    return base_clean

def infer_ensemble_size(name: str) -> int:
    base = [b for b in infer_base_algos(name) if b != 'solo']
    return max(1, len(base)) if base else 1

def infer_family_label(name: str) -> str:
    base = [b for b in infer_base_algos(name) if b != 'solo']
    fams = set(FAMILY_MAP.get(b, 'other') for b in base) if base else set()
    if len(fams) == 0:
        base_guess = name.lower().replace('_solo', '').strip('_')
        return FAMILY_MAP.get(base_guess, 'other')
    if len(fams) == 1:
        return next(iter(fams))
    return 'mixed'

def enrich_classification(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df[['model_name']]
        .drop_duplicates()
        .assign(
            model_name=lambda x: x['model_name'].astype(str),  # Ensure string type
            ensemble_type=lambda x: x['model_name'].apply(infer_ensemble_type),
            ensemble_size=lambda x: x['model_name'].apply(infer_ensemble_size),
            family=lambda x: x['model_name'].apply(infer_family_label),
            is_ensemble=lambda x: x['ensemble_type'].ne('none') | x['ensemble_size'].gt(1),
        )
    )
    return df.merge(meta, on='model_name', how='left')

# Train/Test wide and diagnostics
def make_train_test_wide(df: pd.DataFrame, key_metrics: List[str], pivot_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if pivot_cols is None:
        pivot_cols = ['model_name']
    try:
        for col in pivot_cols + ['dataset_type'] + key_metrics:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple, dict)) else x)
        duplicates = df[df.duplicated(subset=pivot_cols + ['dataset_type'], keep=False)]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} duplicate entries for pivot. Aggregating...")
        aggregated = df.groupby(pivot_cols + ['dataset_type'])[key_metrics].mean().reset_index()
        wide = aggregated.pivot(index=pivot_cols, columns='dataset_type', values=key_metrics)
        wide.columns = [f"{col}_{dataset_type}" for col, dataset_type in wide.columns]
        wide = wide.reset_index()
        wide = wide.fillna(0)
        return wide
    except Exception as e:
        print(f"Error in make_train_test_wide: {e}")
        return pd.DataFrame()

def add_fit_diagnostics(wide: pd.DataFrame) -> pd.DataFrame:
    # Overfitting gap
    if all(c in wide.columns for c in ['weighted_f1_train', 'weighted_f1_test']):
        wide['overfit_gap_f1'] = wide['weighted_f1_train'] - wide['weighted_f1_test']
    else:
        wide['overfit_gap_f1'] = np.nan
    
    # Underfitting score
    if all(c in wide.columns for c in ['weighted_f1_train', 'weighted_f1_test']):
        wide['underfit_score_f1'] = wide[['weighted_f1_train', 'weighted_f1_test']].min(axis=1)
    else:
        wide['underfit_score_f1'] = np.nan
    
    # Complexity
    for dset in ['test']:
        tt_col = f"training_time_{dset}"
        mu_col = f"memory_usage_{dset}"
        wide['z_time'] = 0.0
        if tt_col in wide.columns:
            tt = wide[tt_col]
            wide['z_time'] = (tt - tt.mean()) / (tt.std(ddof=0) + 1e-9)
        wide['z_mem'] = 0.0
        if mu_col in wide.columns:
            mu = wide[mu_col]
            wide['z_mem'] = (mu - mu.mean()) / (mu.std(ddof=0) + 1e-9)
    
    ensemble_size = wide.get('ensemble_size', pd.Series([1] * len(wide)))
    wide['complexity_score'] = wide['z_time'] + wide['z_mem'] + 0.25 * (ensemble_size - 1)
    
    return wide

# Analysis Functions
def analyze_best_models(df_test, metric='weighted_f1', top_n=20):
    return df_test.sort_values(by=metric, ascending=False).head(top_n)

def analyze_best_metrics(df_test):
    metrics = TEST_METRICS
    return {m: df_test[m].max() for m in metrics if m in df_test.columns}

def analyze_best_ensembles(df_test, metric='weighted_f1', top_n=20):
    ensembles = df_test[df_test['is_ensemble']]
    return ensembles.sort_values(by=metric, ascending=False).head(top_n)

def analyze_complex_models(wide, time_threshold_percentile=75, memory_threshold_percentile=75):
    med_f1 = np.nanmedian(wide['weighted_f1_test'])
    hi_time = wide['training_time_test'] >= np.nanpercentile(wide['training_time_test'], time_threshold_percentile)
    hi_mem = wide['memory_usage_test'] >= np.nanpercentile(wide['memory_usage_test'], memory_threshold_percentile)
    flagged = wide[(hi_time | hi_mem) & (wide['weighted_f1_test'] <= (med_f1 + 0.01))].copy()
    flagged['complexity_reason'] = 'High time/memory with marginal F1 gain'
    return flagged.sort_values('complexity_score', ascending=False)

def analyze_fitting(wide, threshold=0.05):
    overfitted = wide[wide['overfit_gap_f1'] > threshold].sort_values('overfit_gap_f1', ascending=False)
    underfitted = wide[wide['underfit_score_f1'] < threshold].sort_values('underfit_score_f1', ascending=True)
    return overfitted, underfitted

def analyze_baselines(df_test):
    return df_test[df_test['model_name'].str.contains('baseline|solo', case=False)]

def family_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    return (
        test_df
        .groupby('family')
        .agg(
            n=('model_name', 'nunique'),
            mean_f1=('weighted_f1', 'mean'),
            std_f1=('weighted_f1', 'std'),
            mean_acc=('accuracy', 'mean'),
            mean_time=('training_time', 'mean'),
            mean_mem=('memory_usage', 'mean')
        )
        .reset_index()
        .sort_values('mean_f1', ascending=False)
    )

def metric_effectiveness(test_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = test_df.select_dtypes(include=['number']).columns.tolist()
    rows = []
    for m in numeric_cols:
        if m == 'weighted_f1' or m not in TEST_METRICS + TIER3_OPTIONAL:
            continue
        a = test_df['weighted_f1'].astype(float)
        b = test_df[m].astype(float)
        if a.notna().sum() > 3 and b.notna().sum() > 3:
            rho, p = stats.spearmanr(a, b, nan_policy='omit')
            rows.append((m, rho, p))
    return pd.DataFrame(rows, columns=['metric', 'spearman_rho', 'p_value']).sort_values('spearman_rho', ascending=False)

# Batched Visualizations (generate PNG per page instead of multi-page PDF)
def batched_multirow_bar_png(df: pd.DataFrame, value_col: str, label_col: str, title: str, per_page: int, base_filename: str, color: str = 'tab:blue'):
    if df.empty:
        return
    n = len(df)
    pages = int(np.ceil(n / per_page))
    for p in range(pages):
        start = p * per_page
        end = min((p + 1) * per_page, n)
        sub = df.iloc[start:end]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.2 * len(sub) + 1)))  # Dynamic height
        sns.barplot(ax=ax, data=sub, y=label_col, x=value_col, color=color, alpha=0.85)
        ax.set_title(f"{title} (Page {p+1}/{pages})")
        ax.set_xlabel(value_col.replace('_', ' ').title())
        ax.set_ylabel('')
        # Add value labels with small font
        for i, v in enumerate(sub[value_col]):
            ax.text(v, i, f" {v:.3f}", va='center', ha='left', fontsize=4)
        plt.subplots_adjust(left=0.3, right=0.95, bottom=0.1, top=0.95)  # Clearer gaps
        out_path = os.path.join(PLOTS_DIR, f'{base_filename}_page_{p+1}.png')
        fig.savefig(out_path)
        plt.close(fig)

def correlation_heatmap(test_df: pd.DataFrame, out_png: str):
    try:
        corr = test_df.corr(numeric_only=True)
        if corr.empty:
            print("No numeric columns for heatmap.")
            return
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0.0, square=False, cbar_kws={'shrink': 0.7}, annot=True, fmt='.2f', annot_kws={"size": 4})
        ax.set_title('Metric Correlation Heatmap (Test Set)')
        fig.savefig(os.path.join(PLOTS_DIR, out_png))
        plt.close(fig)
    except Exception as e:
        print(f"Error in correlation_heatmap: {e}")

def scatter_complexity_vs_perf(test_df: pd.DataFrame, out_png: str, hue_col: str = 'ensemble_size'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue_col in test_df.columns:
        sns.scatterplot(data=test_df, x='training_time', y='weighted_f1', hue=hue_col, ax=ax, s=20, alpha=0.7)
    else:
        print(f"Warning: Hue column '{hue_col}' not found. Plotting without hue.")
        sns.scatterplot(data=test_df, x='training_time', y='weighted_f1', ax=ax, s=20, alpha=0.7)
    ax.set_title('Complexity vs Performance (Test)')
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Weighted F1')
    fig.savefig(os.path.join(PLOTS_DIR, out_png))
    plt.close(fig)

# Generate TXT and PDF reports
def generate_txt_report(analysis_results, table_paths, context, out_txt: str):
    with open(out_txt, 'w') as f:
        f.write(f"Comparative Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*72 + "\n")
        f.write(json.dumps(context, indent=2) + "\n")
        for section, content in analysis_results.items():
            f.write(f"\n{section.upper()}\n")
            if isinstance(content, pd.DataFrame):
                f.write(content.to_string(index=False) + "\n")
            else:
                f.write(str(content) + "\n")
        f.write("\nArtifacts:\n")
        for k, v in table_paths.items():
            f.write(f"- {k}: {v}\n")

def generate_pdf_report(analysis_results, plots_dir, table_paths, context, out_pdf: str):
    c = canvas.Canvas(out_pdf, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Comparative Analysis Report")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(100, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 30
    # Overview
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Overview")
    y -= 20
    c.setFont("Helvetica", 9)
    for k, v in context.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 15
    # Sections with tables (text-based)
    for section, content in analysis_results.items():
        if y < 100:
            c.showPage()
            y = height - 50
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, section)
        y -= 20
        c.setFont("Helvetica", 8)
        if isinstance(content, pd.DataFrame):
            lines = content.to_string(index=False).split('\n')
            for line in lines[:20]:  # Limit to first 20 rows for space
                c.drawString(60, y, line)
                y -= 12
                if y < 100:
                    c.showPage()
                    y = height - 50
        else:
            c.drawString(60, y, str(content))
            y -= 20
    # Embed plots (only PNG images)
    for plot_file in os.listdir(plots_dir):
        if plot_file.endswith('.png'):  # Skip PDFs
            if y < 300:
                c.showPage()
                y = height - 50
            c.drawString(50, y, f"Plot: {plot_file}")
            y -= 20
            plot_path = os.path.join(plots_dir, plot_file)
            c.drawInlineImage(plot_path, 50, y - 200, width=500, height=200)
            y -= 220
    # Appendices: Table paths
    c.showPage()
    y = height - 50
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Appendices: Table Paths")
    y -= 20
    c.setFont("Helvetica", 9)
    for k, v in table_paths.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 15
    c.save()

# Export tables
def export_tables(wide: pd.DataFrame) -> Dict[str, str]:
    paths = {}
    def save_tbl(df_tbl: pd.DataFrame, filename: str) -> str:
        out = os.path.join(RESULTS_DIR, filename)
        df_tbl.to_csv(out, index=False)
        return out
    all_sorted = wide.sort_values('weighted_f1_test', ascending=False)
    ens_sorted = all_sorted[all_sorted['is_ensemble']]  # Ensure this line comes before usage
    paths['top20_ensembles'] = save_tbl(ens_sorted.head(20), 'top20_ensembles.csv')
    paths['worst20_ensembles'] = save_tbl(ens_sorted.tail(20), 'worst20_ensembles.csv')
    # Over/Underfitting (using ens_sorted for ensembles)
    overfit_hi = ens_sorted.sort_values('overfit_gap_f1', ascending=False).head(20)
    overfit_lo = ens_sorted.sort_values('overfit_gap_f1', ascending=True).head(20)
    underfit_hi = ens_sorted.sort_values('underfit_score_f1', ascending=True).head(20)
    underfit_lo = ens_sorted.sort_values('underfit_score_f1', ascending=False).head(20)
    paths['highest_overfit_ensembles'] = save_tbl(overfit_hi, 'highest_overfit_ensembles.csv')
    paths['lowest_overfit_ensembles'] = save_tbl(overfit_lo, 'lowest_overfit_ensembles.csv')
    paths['highest_underfit_ensembles'] = save_tbl(underfit_hi, 'highest_underfit_ensembles.csv')
    paths['lowest_underfit_ensembles'] = save_tbl(underfit_lo, 'lowest_underfit_ensembles.csv')
    # Ensemble types
    for et in ['voting', 'stacking', 'parametric', 'cv', 'weighted']:
        sub = all_sorted[all_sorted['ensemble_type'] == et]
        if not sub.empty:
            paths[f'all_{et}_ensembles'] = save_tbl(sub, f'all_{et}_ensembles.csv')
    # Size-based
    for op, name in [
        (wide['ensemble_size'] == 1, 'solo'),
        (wide['ensemble_size'] == 2, 'double'),
        (wide['ensemble_size'] == 3, 'triple'),
        (wide['ensemble_size'] >= 4, 'quadplus')
    ]:
        sub = all_sorted[op]
        if not sub.empty:
            paths[f'compare_{name}'] = save_tbl(sub, f'compare_{name}_only.csv')
    return paths

# Main analysis pipeline
def run_analysis(file_path='detailed_metrics.csv', per_page: int = 50):  # Increased per_page for density
    df = load_data(file_path)
    df = enrich_classification(df)
    test_df = df[df['dataset_type'] == 'test'].copy()
    train_df = df[df['dataset_type'] == 'train'].copy()
    key_metrics = [c for c in TEST_METRICS + TIER3_OPTIONAL if c in df.columns]
    wide = make_train_test_wide(df, key_metrics)
    meta_cols = ['model_name', 'ensemble_size', 'ensemble_type', 'family', 'is_ensemble']
    meta_df = df[meta_cols].copy()
    for col in meta_cols:
        if meta_df[col].dtype == 'object':
            meta_df[col] = meta_df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple, dict)) else x)
    meta = meta_df.drop_duplicates()
    wide = wide.merge(meta, on='model_name', how='left', suffixes=('', '_dup'))
    # Clean up any duplicate columns from merge
    for col in meta_cols:
        dup_col = f'{col}_dup'
        if dup_col in wide.columns:
            wide[col] = wide[col].combine_first(wide[dup_col])
            wide = wide.drop(columns=[dup_col])
    wide = add_fit_diagnostics(wide)
    # Analyses
    analysis_results = {}
    analysis_results['Best Models'] = analyze_best_models(test_df)
    analysis_results['Best Metrics'] = analyze_best_metrics(test_df)
    analysis_results['Best Ensembles'] = analyze_best_ensembles(test_df)
    analysis_results['Too Complex Models'] = analyze_complex_models(wide)
    overfitted, underfitted = analyze_fitting(wide)
    analysis_results['Overfitted Models'] = overfitted
    analysis_results['Underfitted Models'] = underfitted
    analysis_results['Baseline Models'] = analyze_baselines(test_df)
    analysis_results['Family Summary'] = family_summary(test_df)
    analysis_results['Metric Effectiveness'] = metric_effectiveness(test_df)
    # Compute sorted DataFrames for export and visualizations
    all_sorted = wide.sort_values('weighted_f1_test', ascending=False)
    ens_sorted = all_sorted[all_sorted['is_ensemble']]
    # Compute overfit/underfit subsets (for tables and categories)
    overfit_hi = ens_sorted.sort_values('overfit_gap_f1', ascending=False).head(20)
    overfit_lo = ens_sorted.sort_values('overfit_gap_f1', ascending=True).head(20)
    underfit_hi = ens_sorted.sort_values('underfit_score_f1', ascending=True).head(20)
    underfit_lo = ens_sorted.sort_values('underfit_score_f1', ascending=False).head(20)
    table_paths = export_tables(wide)
    # Visualizations (batched for all categories)
    categories = {
        'solo_only': wide['ensemble_size'] == 1,
        'double_only': wide['ensemble_size'] == 2,
        'triple_only': wide['ensemble_size'] == 3,
        'quadplus_only': wide['ensemble_size'] >= 4,
        'top20_ensembles': wide['model_name'].isin(ens_sorted.head(20)['model_name']),
        'worst20_ensembles': wide['model_name'].isin(ens_sorted.tail(20)['model_name']),
        'highest_overfit_ensembles': wide['model_name'].isin(overfit_hi['model_name']),
        'lowest_overfit_ensembles': wide['model_name'].isin(overfit_lo['model_name']),
        'highest_underfit_ensembles': wide['model_name'].isin(underfit_hi['model_name']),
        'lowest_underfit_ensembles': wide['model_name'].isin(underfit_lo['model_name']),
        'all_voting_ensembles': wide['ensemble_type'] == 'voting',
        'all_stacking_ensembles': wide['ensemble_type'] == 'stacking',
        'all_parametric_ensembles': wide['ensemble_type'] == 'parametric'
    }
    for label, mask in categories.items():
        sub = wide[mask].sort_values('weighted_f1_test', ascending=False)[['model_name', 'weighted_f1_test']]
        if not sub.empty:
            batched_multirow_bar_png(sub, 'weighted_f1_test', 'model_name', f'Weighted F1 (Test) - {label.replace("_", " ").title()}', per_page, f'f1_test_{label}')
    # Family stats visualization (bar plot)
    fam = family_summary(test_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fam, x='mean_f1', y='family', ax=ax)
    ax.set_title('Family Summary by Mean F1')
    fig.savefig(os.path.join(PLOTS_DIR, 'family_summary.png'))
    plt.close(fig)
    # Complexity vs Performance
    test_df = test_df.merge(wide[['model_name', 'ensemble_size']], on='model_name', how='left', suffixes=('', '_dup'))
    if 'ensemble_size_dup' in test_df.columns:
        test_df['ensemble_size'] = test_df['ensemble_size'].combine_first(test_df['ensemble_size_dup'])
        test_df = test_df.drop(columns=['ensemble_size_dup'])
    if 'ensemble_size' not in test_df.columns:
        print("Warning: 'ensemble_size' not added to test_df. Using default value.")
        test_df['ensemble_size'] = 1  # Fallback to default if merge fails
    scatter_complexity_vs_perf(test_df, 'complexity_vs_performance_test.png')
    # Correlation Heatmap
    correlation_heatmap(test_df, 'correlation_heatmap_test.png')
    # Report context
    context = {
        'n_models': df['model_name'].nunique(),
        'n_rows': len(df),
        'train_rows': len(train_df),
        'test_rows': len(test_df)
    }
    # Generate reports
    txt_path = os.path.join(REPORTS_DIR, 'analysis_report.txt')
    pdf_path = os.path.join(REPORTS_DIR, 'analysis_report.pdf')
    generate_txt_report(analysis_results, table_paths, context, txt_path)
    generate_pdf_report(analysis_results, PLOTS_DIR, table_paths, context, pdf_path)
    print(f"Analysis complete. Results in: {RESULTS_DIR}")
    print(f"TXT Report: {txt_path}")
    print(f"PDF Report: {pdf_path}")

if __name__ == "__main__":
    run_analysis('C:/Users/Khwaish/.vscode/CableHealthExp/resnew/analysis_results/detailed_metrics.csv')
