"""
CTGAN-Based Cable Health Dataset Generation
==========================================
This implementation uses CTGAN to generate realistic synthetic cable health data
based on a small sample dataset, preserving domain relationships and patterns.

Key Improvements over Random Generation:
- Preserves feature correlations and dependencies
- Maintains realistic target variable relationships
- Incorporates domain knowledge through learned patterns
- Generates high-quality synthetic data for ML model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# CTGAN imports
try:
    from ctgan import CTGAN
    print("✅ CTGAN library imported successfully")
except ImportError:
    print("❌ CTGAN not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "ctgan"])
    from ctgan import CTGAN
    print("✅ CTGAN installed and imported successfully")

class CTGANCableHealthGenerator:
    """
    Advanced Cable Health Dataset Generator using CTGAN
    
    This class uses Conditional Tabular GAN to learn patterns from a small
    sample dataset and generate realistic synthetic cable health data that
    preserves domain relationships and statistical properties.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize CTGAN model
        self.ctgan_model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.original_data = None
        self.label_encoders = {}
        
        # Define categorical features for CTGAN
        self.categorical_features = [
            'MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory',
            'FloodZoneRisk', 'CableHealthScore'
        ]
        
        print(f"✅ CTGAN Cable Health Generator initialized with random_state={random_state}")
    
    def load_and_analyze_sample_data(self, filepath):
        """
        Load and analyze the sample dataset to understand its structure and patterns
        """
        print("\n📂 Loading and Analyzing Sample Dataset...")
        
        # Load the sample data
        df = pd.read_csv(filepath)
        self.original_data = df.copy()
        
        print(f"Sample dataset shape: {df.shape}")
        print(f"Features: {len(df.columns)}")
        
        # Analyze feature types
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'CableID' in numerical_features:
            numerical_features.remove('CableID')  # Remove ID column
        
        categorical_features = [col for col in self.categorical_features if col in df.columns]
        
        self.numerical_columns = numerical_features
        self.categorical_columns = categorical_features
        
        print(f"Numerical features: {len(self.numerical_columns)}")
        print(f"Categorical features: {len(self.categorical_columns)}")
        
        # Analyze target distribution
        print(f"\n🎯 Target Variable Analysis:")
        target_dist = df['CableHealthScore'].value_counts().sort_index()
        for i, count in target_dist.items():
            status = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical'][i]
            print(f"  Class {i} ({status}): {count} samples ({count/len(df)*100:.1f}%)")
        
        # ENHANCED: Robust feature-target correlation analysis with multiple fallback strategies
        print(f"\n🔍 Feature-Target Correlations:")
        if len(self.numerical_columns) > 0:
            try:
                # Strategy 1: Standard correlation approach
                correlations = df[self.numerical_columns + ['CableHealthScore']].corr()['CableHealthScore'].drop('CableHealthScore')
                
                # Debug: Check what type correlations is
                print(f"  Debug: correlations type = {type(correlations)}")
                print(f"  Debug: correlations shape = {getattr(correlations, 'shape', 'No shape attribute')}")
                
                # Handle different data types
                if isinstance(correlations, pd.DataFrame):
                    print("  ⚠️  Correlations returned as DataFrame, converting to Series...")
                    # If it's a DataFrame, try to squeeze it to Series
                    correlations = correlations.squeeze()
                    
                    # If still DataFrame, take the first column
                    if isinstance(correlations, pd.DataFrame):
                        correlations = correlations.iloc[:, 0]
                
                elif isinstance(correlations, pd.Series):
                    print("  ✅ Correlations is a Series (expected)")
                else:
                    print(f"  ⚠️  Unexpected correlations type: {type(correlations)}")
                    raise TypeError(f"Unexpected correlations type: {type(correlations)}")
                
                # Now safely sort the Series
                if isinstance(correlations, pd.Series):
                    top_corr = correlations.abs().sort_values(ascending=False).head(10)
                    
                    for feature, corr in top_corr.items():
                        direction = "↗️" if correlations[feature] > 0 else "↘️"
                        print(f"  {direction} {feature}: {corr:.3f}")
                else:
                    raise TypeError("Could not convert correlations to Series")
                    
            except Exception as e:
                print(f"  ⚠️  Standard correlation failed: {e}")
                print("  🔧 Trying fallback correlation methods...")
                
                # Fallback Strategy 1: Manual correlation calculation
                try:
                    print("  Fallback 1: Manual correlation calculation...")
                    correlations_manual = {}
                    target_values = df['CableHealthScore']
                    
                    for feature in self.numerical_columns:
                        feature_values = df[feature]
                        # Calculate Pearson correlation manually
                        corr_coef = np.corrcoef(feature_values, target_values)[0, 1]
                        if not np.isnan(corr_coef):
                            correlations_manual[feature] = corr_coef
                    
                    # Sort by absolute correlation
                    sorted_correlations = sorted(correlations_manual.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)
                    
                    print("  ✅ Manual correlation calculation successful:")
                    for feature, corr in sorted_correlations[:10]:
                        direction = "↗️" if corr > 0 else "↘️"
                        print(f"    {direction} {feature}: {abs(corr):.3f}")
                        
                except Exception as e2:
                    print(f"  ⚠️  Manual correlation failed: {e2}")
                    
                    # Fallback Strategy 2: Individual feature correlations
                    try:
                        print("  Fallback 2: Individual feature correlations...")
                        individual_correlations = {}
                        
                        for feature in self.numerical_columns:
                            try:
                                corr_val = df[feature].corr(df['CableHealthScore'])
                                if not np.isnan(corr_val):
                                    individual_correlations[feature] = corr_val
                            except Exception:
                                continue
                        
                        if individual_correlations:
                            # Sort by absolute correlation
                            sorted_individual = sorted(individual_correlations.items(), 
                                                    key=lambda x: abs(x[1]), reverse=True)
                            
                            print("  ✅ Individual correlation calculation successful:")
                            for feature, corr in sorted_individual[:10]:
                                direction = "↗️" if corr > 0 else "↘️"
                                print(f"    {direction} {feature}: {abs(corr):.3f}")
                        else:
                            print("  ❌ No valid correlations found")
                            
                    except Exception as e3:
                        print(f"  ❌ All correlation methods failed: {e3}")
                        print("  📊 Showing basic feature statistics instead:")
                        
                        # Fallback Strategy 3: Basic statistics
                        for feature in self.numerical_columns[:10]:
                            mean_val = df[feature].mean()
                            std_val = df[feature].std()
                            print(f"    📈 {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
        else:
            print("  ⚠️  No numerical features available for correlation analysis")
        
        return df
    
    def preprocess_for_ctgan(self, df):
        """
        Preprocess data specifically for CTGAN training
        """
        print("\n🔧 Preprocessing Data for CTGAN...")
        
        df_processed = df.copy()
        
        # Remove CableID as it's not needed for pattern learning
        if 'CableID' in df_processed.columns:
            df_processed = df_processed.drop('CableID', axis=1)
        
        # Ensure categorical columns are properly typed
        for col in self.categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # Handle any missing values (though sample data shouldn't have any)
        if df_processed.isnull().sum().sum() > 0:
            print("⚠️  Missing values detected, filling with median/mode...")
            for col in df_processed.columns:
                if col in self.numerical_columns:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        print(f"✅ Data preprocessed for CTGAN training")
        print(f"Final shape: {df_processed.shape}")
        
        return df_processed
    
    def train_ctgan_model(self, df, epochs=300, batch_size=500):
        """
        Train CTGAN model on the sample dataset
        
        Theory: CTGAN learns the joint distribution of features and their
        relationships, enabling generation of realistic synthetic data that
        preserves statistical properties and domain relationships.
        """
        print(f"\n🚀 Training CTGAN Model...")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        
        # Preprocess data
        df_processed = self.preprocess_for_ctgan(df)
        
        # FIXED: Calculate optimal batch_size and pac for small datasets
        n_samples = len(df_processed)
        
        # Adjust batch size for small datasets
        if n_samples < batch_size:
            batch_size = n_samples
            print(f"⚠️  Adjusted batch_size to {batch_size} (dataset size)")
        
        # FIXED: Calculate pac that divides evenly into batch_size
        # Find the largest divisor of batch_size that's <= 10
        possible_pac_values = [i for i in range(1, 11) if batch_size % i == 0]
        optimal_pac = max(possible_pac_values) if possible_pac_values else 1
        
        print(f"🔧 Optimal PAC value: {optimal_pac} (batch_size {batch_size} is divisible)")
        
        # Configure CTGAN with optimized parameters for small datasets
        try:
            self.ctgan_model = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                generator_dim=(128, 128),  # Reduced for small data
                discriminator_dim=(128, 128),  # Reduced for small data
                generator_lr=2e-4,
                discriminator_lr=2e-4,
                discriminator_steps=1,
                log_frequency=True,
                verbose=True,
                pac=optimal_pac,  # FIXED: Use calculated optimal pac
                embedding_dim=64  # Reduced for small data
            )
            print(f"✅ CTGAN model configured successfully")
        except Exception as e:
            print(f"⚠️  Standard CTGAN configuration failed: {e}")
            print("🔧 Trying minimal configuration...")
            
            # Fallback with minimal configuration
            minimal_pac = 1  # Most conservative pac value
            minimal_batch_size = min(batch_size, n_samples)
            
            self.ctgan_model = CTGAN(
                epochs=epochs//2,  # Reduced epochs
                batch_size=minimal_batch_size,
                generator_dim=(64, 64),  # Very small architecture
                discriminator_dim=(64, 64),
                generator_lr=1e-3,  # Higher learning rate for faster convergence
                discriminator_lr=1e-3,
                discriminator_steps=1,
                log_frequency=True,
                verbose=True,
                pac=minimal_pac,  # FIXED: Use pac=1 for maximum compatibility
                embedding_dim=32  # Minimal embedding
            )
            print(f"✅ CTGAN model configured with minimal settings")
        
        # Identify categorical columns for CTGAN
        categorical_columns_for_ctgan = [col for col in self.categorical_columns if col in df_processed.columns]
        
        print(f"Training on {len(df_processed)} samples...")
        print(f"Categorical columns for CTGAN: {categorical_columns_for_ctgan}")
        print(f"Final batch_size: {self.ctgan_model._batch_size}")
        print(f"Final pac: {self.ctgan_model.pac}")
        
        # Train the model
        try:
            self.ctgan_model.fit(df_processed, categorical_columns_for_ctgan)
            print("✅ CTGAN model trained successfully!")
        except Exception as e:
            print(f"❌ CTGAN training failed: {e}")
            print("🔧 Trying with ultra-minimal configuration...")
            
            # Ultra-minimal fallback
            try:
                self.ctgan_model = CTGAN(
                    epochs=50,  # Very few epochs
                    batch_size=n_samples,  # Use full dataset as batch
                    generator_dim=(32,),  # Single layer
                    discriminator_dim=(32,),  # Single layer
                    generator_lr=1e-2,  # High learning rate
                    discriminator_lr=1e-2,
                    discriminator_steps=1,
                    log_frequency=False,
                    verbose=True,
                    pac=1,  # Minimal pac
                    embedding_dim=16  # Very small embedding
                )
                
                self.ctgan_model.fit(df_processed, categorical_columns_for_ctgan)
                print("✅ CTGAN model trained with ultra-minimal configuration!")
                
            except Exception as e2:
                print(f"❌ All CTGAN configurations failed: {e2}")
                print("🔧 Consider using a larger sample dataset (>50 rows) for better CTGAN performance")
                raise e2
        
        return self.ctgan_model

    
    def generate_synthetic_data(self, num_samples=20000, validate_quality=True):
        """
        Generate synthetic cable health data using trained CTGAN model
        """
        print(f"\n📈 Generating {num_samples:,} Synthetic Samples...")
        
        if self.ctgan_model is None:
            raise ValueError("CTGAN model not trained. Call train_ctgan_model() first.")
        
        # Generate synthetic data
        synthetic_data = self.ctgan_model.sample(num_samples)
        
        # Add CableID column
        synthetic_data['CableID'] = [f"CAB{i+1:05}" for i in range(len(synthetic_data))]
        
        # Reorder columns to match original structure
        if self.original_data is not None:
            original_columns = self.original_data.columns.tolist()
            available_columns = [col for col in original_columns if col in synthetic_data.columns]
            synthetic_data = synthetic_data[available_columns]
        
        # Post-process to ensure data quality
        synthetic_data = self.post_process_synthetic_data(synthetic_data)
        
        if validate_quality:
            self.validate_synthetic_data_quality(synthetic_data)
        
        print(f"✅ Generated {len(synthetic_data):,} high-quality synthetic samples")
        
        return synthetic_data
    
    def post_process_synthetic_data(self, df):
        """
        Post-process synthetic data to ensure realistic constraints and relationships
        """
        print("🔧 Post-processing synthetic data...")
        
        df_processed = df.copy()
        
        # Ensure categorical variables are within valid ranges
        if 'MaterialType' in df_processed.columns:
            df_processed['MaterialType'] = df_processed['MaterialType'].clip(0, 1).round().astype(int)
        
        if 'InstallationType' in df_processed.columns:
            df_processed['InstallationType'] = df_processed['InstallationType'].clip(0, 2).round().astype(int)
        
        if 'FailureRootCauseMajorCategory' in df_processed.columns:
            df_processed['FailureRootCauseMajorCategory'] = df_processed['FailureRootCauseMajorCategory'].clip(0, 2).round().astype(int)
        
        if 'FloodZoneRisk' in df_processed.columns:
            df_processed['FloodZoneRisk'] = df_processed['FloodZoneRisk'].clip(0, 1).round().astype(int)
        
        if 'CableHealthScore' in df_processed.columns:
            df_processed['CableHealthScore'] = df_processed['CableHealthScore'].clip(0, 2).round().astype(int)
        
        # Ensure numerical constraints
        numerical_constraints = {
            'AssetAgeYears': (1, 30),
            'RatedLifespanYears': (20, 30),
            'YearsRemainingWarranty': (0, 15),
            'NumberOfRepairs': (0, 20),
            'NumberOfFailuresPast3Years': (0, 20),
            'FailureRatePerYear': (0, 1),
            'AvgVoltageDeviationPercent': (1, 8),
            'PeakLoadKW': (100, 800),
            'OverloadEventCountPast3Years': (0, 15),
            'SwitchingEventsPerYear': (5, 25),
            'FCITriggerRatePerYear': (0, 1),
            'LastFCITriggerRecencyDays': (1, 365),
            'PartialDischargeSeverityScore': (0, 1),
            'PDThresholdBreachesPast1Year': (0, 10),
            'SensitiveCustomerCount': (0, 15),
            'ConnectedLoadKW': (1000, 10000),
            'InspectionDefectSeverityScore': (0, 1),
            'TimeSinceLastInspectionDays': (1, 365),
            'AvgGroundTemperatureCelsius': (10, 40),
            'SoilCorrosivityIndex': (0, 1),
            'CriticalityScore': (0, 1),
            'EstimatedDowntimeCostPerFailure': (1000, 50000),
            'TemperatureAnomaliesPast1Year': (0, 10),
            'VibrationAnomaliesPast1Year': (0, 5)
        }
        
        for col, (min_val, max_val) in numerical_constraints.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].clip(min_val, max_val)
                
                # Round integer columns
                if col in ['AssetAgeYears', 'RatedLifespanYears', 'YearsRemainingWarranty', 
                          'NumberOfRepairs', 'NumberOfFailuresPast3Years', 'OverloadEventCountPast3Years',
                          'SwitchingEventsPerYear', 'LastFCITriggerRecencyDays', 'PDThresholdBreachesPast1Year',
                          'SensitiveCustomerCount', 'ConnectedLoadKW', 'TimeSinceLastInspectionDays',
                          'AvgGroundTemperatureCelsius', 'EstimatedDowntimeCostPerFailure',
                          'TemperatureAnomaliesPast1Year', 'VibrationAnomaliesPast1Year', 'PeakLoadKW']:
                    df_processed[col] = df_processed[col].round().astype(int)
                else:
                    df_processed[col] = df_processed[col].round(3)
        
        # Apply domain-specific constraints
        # Ensure RatedLifespanYears is realistic (20, 25, or 30)
        if 'RatedLifespanYears' in df_processed.columns:
            df_processed['RatedLifespanYears'] = df_processed['RatedLifespanYears'].apply(
                lambda x: min([20, 25, 30], key=lambda y: abs(y - x))
            )
        
        # Ensure warranty years don't exceed asset age
        if all(col in df_processed.columns for col in ['YearsRemainingWarranty', 'AssetAgeYears', 'RatedLifespanYears']):
            max_warranty = df_processed['RatedLifespanYears'] - df_processed['AssetAgeYears']
            df_processed['YearsRemainingWarranty'] = np.minimum(
                df_processed['YearsRemainingWarranty'], 
                np.maximum(0, max_warranty)
            )
        
        print("✅ Post-processing completed")
        
        return df_processed
    
    def validate_synthetic_data_quality(self, synthetic_data):
        """
        Validate the quality of generated synthetic data
        """
        print("\n🔍 Validating Synthetic Data Quality...")
        
        if self.original_data is None:
            print("⚠️  No original data available for comparison")
            return
        
        # Compare basic statistics
        print("\n📊 Statistical Comparison:")
        
        # Numerical features comparison
        numerical_features = [col for col in self.numerical_columns if col in synthetic_data.columns]
        
        stats_comparison = pd.DataFrame({
            'Original_Mean': self.original_data[numerical_features].mean(),
            'Synthetic_Mean': synthetic_data[numerical_features].mean(),
            'Original_Std': self.original_data[numerical_features].std(),
            'Synthetic_Std': synthetic_data[numerical_features].std()
        })
        
        stats_comparison['Mean_Diff'] = abs(stats_comparison['Original_Mean'] - stats_comparison['Synthetic_Mean'])
        stats_comparison['Std_Diff'] = abs(stats_comparison['Original_Std'] - stats_comparison['Synthetic_Std'])
        
        print("Top 10 features by statistical similarity:")
        top_similar = stats_comparison.nsmallest(10, 'Mean_Diff')
        for feature in top_similar.index[:5]:
            print(f"  ✅ {feature}: Mean diff = {stats_comparison.loc[feature, 'Mean_Diff']:.3f}")
        
        # Target distribution comparison
        print(f"\n🎯 Target Distribution Comparison:")
        original_target_dist = self.original_data['CableHealthScore'].value_counts(normalize=True).sort_index()
        synthetic_target_dist = synthetic_data['CableHealthScore'].value_counts(normalize=True).sort_index()
        
        for i in range(3):
            orig_prop = original_target_dist.get(i, 0)
            synth_prop = synthetic_target_dist.get(i, 0)
            status = ['🔵 Healthy', '🟠 At Risk', '🔴 Critical'][i]
            print(f"  {status}: Original {orig_prop:.1%} vs Synthetic {synth_prop:.1%}")
        
        # Correlation preservation
        if len(numerical_features) > 1:
            original_corr = self.original_data[numerical_features].corr()
            synthetic_corr = synthetic_data[numerical_features].corr()
            
            corr_diff = abs(original_corr - synthetic_corr).mean().mean()
            print(f"\n🔗 Correlation Preservation:")
            print(f"  Average correlation difference: {corr_diff:.3f}")
            
            if corr_diff < 0.1:
                print("  ✅ Excellent correlation preservation")
            elif corr_diff < 0.2:
                print("  ✅ Good correlation preservation")
            else:
                print("  ⚠️  Moderate correlation preservation")
    
    def create_visualization_report(self, synthetic_data, save_plots=True):
        """
        Create comprehensive visualization report comparing original and synthetic data
        """
        print("\n📊 Creating Visualization Report...")
        
        if self.original_data is None:
            print("⚠️  No original data available for visualization comparison")
            return
        
        # Set up the plotting
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('CTGAN Synthetic Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Target distribution comparison
        ax = axes[0, 0]
        original_target = self.original_data['CableHealthScore'].value_counts().sort_index()
        synthetic_target = synthetic_data['CableHealthScore'].value_counts().sort_index()
        
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, original_target.values, width, label='Original', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, synthetic_target.values, width, label='Synthetic', alpha=0.8, color='lightcoral')
        ax.set_xlabel('Cable Health Score')
        ax.set_ylabel('Count')
        ax.set_title('Target Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Healthy', 'At Risk', 'Critical'])
        ax.legend()
        
        # 2-4. Feature distribution comparisons
        numerical_features = [col for col in self.numerical_columns if col in synthetic_data.columns]
        important_features = ['AssetAgeYears', 'FailureRatePerYear', 'NumberOfRepairs', 
                            'PartialDischargeSeverityScore', 'AvgVoltageDeviationPercent']
        
        plot_features = [f for f in important_features if f in numerical_features][:5]
        
        for i, feature in enumerate(plot_features):
            if i < 5:
                row = (i + 1) // 3
                col = (i + 1) % 3
                ax = axes[row, col]
                
                ax.hist(self.original_data[feature], bins=10, alpha=0.7, label='Original', 
                       color='skyblue', density=True)
                ax.hist(synthetic_data[feature], bins=10, alpha=0.7, label='Synthetic', 
                       color='lightcoral', density=True)
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.set_title(f'{feature} Distribution')
                ax.legend()
        
        # 6. Correlation heatmap comparison
        if len(plot_features) >= 3:
            ax = axes[2, 0]
            original_corr = self.original_data[plot_features[:3]].corr()
            sns.heatmap(original_corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Original Data Correlations')
            
            ax = axes[2, 1]
            synthetic_corr = synthetic_data[plot_features[:3]].corr()
            sns.heatmap(synthetic_corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            ax.set_title('Synthetic Data Correlations')
        
        # 8. Feature importance comparison using a quick model
        ax = axes[2, 2]
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Train on original data
            rf_orig = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            X_orig = self.original_data[plot_features]
            y_orig = self.original_data['CableHealthScore']
            rf_orig.fit(X_orig, y_orig)
            
            # Train on synthetic data
            rf_synth = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            X_synth = synthetic_data[plot_features]
            y_synth = synthetic_data['CableHealthScore']
            rf_synth.fit(X_synth, y_synth)
            
            # Compare feature importances
            x_pos = np.arange(len(plot_features))
            ax.bar(x_pos - 0.2, rf_orig.feature_importances_, 0.4, label='Original', alpha=0.8)
            ax.bar(x_pos + 0.2, rf_synth.feature_importances_, 0.4, label='Synthetic', alpha=0.8)
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_features, rotation=45)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature importance\ncomparison failed:\n{e}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ctgan_synthetic_data_quality_report.png', dpi=300, bbox_inches='tight')
            print("📁 Visualization report saved as 'ctgan_synthetic_data_quality_report.png'")
        
        plt.show()

def main_ctgan_generation():
    """
    Main execution function for CTGAN-based cable health dataset generation
    """
    print("🚀 Starting CTGAN Cable Health Dataset Generation...")
    print("=" * 70)
    
    # Initialize generator
    generator = CTGANCableHealthGenerator(random_state=42)
    
    # Step 1: Load and analyze sample data
    sample_data = generator.load_and_analyze_sample_data('cable_health_sample_ordinal_encoded - cable_health_sample_ordinal_encoded.csv')
    
    # Step 2: Train CTGAN model
    ctgan_model = generator.train_ctgan_model(
        sample_data, 
        epochs=300,  # Adjust based on your computational resources
        batch_size=25  # Small batch size for small dataset
    )
    
    # Step 3: Generate synthetic data
    synthetic_data = generator.generate_synthetic_data(
        num_samples=20000,
        validate_quality=True
    )
    
    # Step 4: Save synthetic data
    output_file = "cable_health_ctgan_synthetic_20k.csv"
    synthetic_data.to_csv(output_file, index=False)
    print(f"✅ Synthetic dataset saved to {output_file}")
    
    # Step 5: Create visualization report
    generator.create_visualization_report(synthetic_data, save_plots=True)
    
    # Step 6: Quick model evaluation comparison
    print("\n🧪 Quick Model Evaluation Comparison...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
        
        # Prepare data for modeling
        feature_columns = [col for col in synthetic_data.columns if col not in ['CableID', 'CableHealthScore']]
        
        # Original data evaluation
        if len(sample_data) > 10:  # Only if we have enough samples
            X_orig = sample_data[feature_columns]
            y_orig = sample_data['CableHealthScore']
            
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_orig, y_orig, test_size=0.3, random_state=42, stratify=y_orig
            )
            
            rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_orig.fit(X_train_orig, y_train_orig)
            y_pred_orig = rf_orig.predict(X_test_orig)
            orig_accuracy = accuracy_score(y_test_orig, y_pred_orig)
            
            print(f"📊 Original Data Model Accuracy: {orig_accuracy:.3f}")
        
        # Synthetic data evaluation
        X_synth = synthetic_data[feature_columns]
        y_synth = synthetic_data['CableHealthScore']
        
        X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
            X_synth, y_synth, test_size=0.2, random_state=42, stratify=y_synth
        )
        
        rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_synth.fit(X_train_synth, y_train_synth)
        y_pred_synth = rf_synth.predict(X_test_synth)
        synth_accuracy = accuracy_score(y_test_synth, y_pred_synth)
        
        print(f"📊 Synthetic Data Model Accuracy: {synth_accuracy:.3f}")
        
        # Classification report for synthetic data
        print(f"\n📋 Synthetic Data Classification Report:")
        print(classification_report(y_test_synth, y_pred_synth, 
                                  target_names=['Healthy', 'At Risk', 'Critical']))
        
    except Exception as e:
        print(f"⚠️  Model evaluation failed: {e}")
    
    print("\n🎉 CTGAN Dataset Generation Completed Successfully!")
    print("=" * 70)
    print(f"📁 Generated dataset: {output_file}")
    print(f"📊 Dataset size: {len(synthetic_data):,} rows")
    print(f"🔧 Features: {len(feature_columns)}")
    print(f"🎯 Target classes: 3 (Healthy, At Risk, Critical)")
    
    return generator, synthetic_data

if __name__ == "__main__":
    # Execute CTGAN generation
    generator, synthetic_dataset = main_ctgan_generation()
