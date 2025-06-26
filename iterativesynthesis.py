"""
Method 1: Iterative Learning and Synthesis for Cable Health Dataset Generation
==============================================================================
This approach uses the existing 25-row dataset to learn patterns and iteratively
generate new synthetic data with improved target labeling based on learned relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class IterativeCableHealthGenerator:
    """
    Iterative cable health dataset generator using domain knowledge and pattern learning
    """
    
    def __init__(self, seed_data_path, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Load seed dataset
        self.seed_data = pd.read_csv(seed_data_path)
        print(f"✅ Loaded seed dataset with {len(self.seed_data)} rows")
        
        # Initialize components
        self.pattern_model = None
        self.feature_distributions = {}
        self.correlation_matrix = None
        self.domain_rules = self._define_domain_rules()
        
        # Feature categories for targeted synthesis
        self.critical_features = [
            'AssetAgeYears', 'NumberOfRepairs', 'NumberOfFailuresPast3Years',
            'FailureRatePerYear', 'PartialDischargeSeverityScore', 'InspectionDefectSeverityScore'
        ]
        
        self.risk_features = [
            'OverloadEventCountPast3Years', 'PDThresholdBreachesPast1Year',
            'AvgVoltageDeviationPercent', 'TimeSinceLastInspectionDays'
        ]
        
        self.environmental_features = [
            'SoilCorrosivityIndex', 'FloodZoneRisk', 'AvgGroundTemperatureCelsius'
        ]
    
    def _define_domain_rules(self):
        """
        Define domain expert knowledge rules for cable health assessment
        
        Theory: These rules encode real-world cable degradation patterns and
        maintenance decision criteria used by utility companies.
        """
        return {
            'critical_conditions': {
                # Multiple failures indicate critical condition
                'high_failure_rate': lambda row: row['FailureRatePerYear'] > 0.8 and row['NumberOfFailuresPast3Years'] > 8,
                
                # Old cables with high repair count
                'aging_with_repairs': lambda row: row['AssetAgeYears'] > 20 and row['NumberOfRepairs'] > 6,
                
                # Severe partial discharge issues
                'severe_pd': lambda row: row['PartialDischargeSeverityScore'] > 0.7 and row['PDThresholdBreachesPast1Year'] > 2,
                
                # High defect severity with recent inspection
                'critical_defects': lambda row: row['InspectionDefectSeverityScore'] > 0.8 and row['TimeSinceLastInspectionDays'] < 180,
                
                # Combination of electrical stress factors
                'electrical_stress': lambda row: (row['AvgVoltageDeviationPercent'] > 6 and 
                                                row['OverloadEventCountPast3Years'] > 7 and
                                                row['PeakLoadKW'] > 600)
            },
            
            'at_risk_conditions': {
                # Moderate aging with some issues
                'moderate_aging': lambda row: (row['AssetAgeYears'] > 15 and 
                                             row['AssetAgeYears'] <= 25 and 
                                             row['NumberOfRepairs'] > 2),
                
                # Increasing failure trend
                'failure_trend': lambda row: (row['FailureRatePerYear'] > 0.4 and 
                                            row['FailureRatePerYear'] <= 0.8 and
                                            row['NumberOfFailuresPast3Years'] > 3),
                
                # Moderate partial discharge
                'moderate_pd': lambda row: (row['PartialDischargeSeverityScore'] > 0.3 and 
                                          row['PartialDischargeSeverityScore'] <= 0.7),
                
                # Environmental stress
                'environmental_risk': lambda row: (row['SoilCorrosivityIndex'] > 0.6 and 
                                                 row['FloodZoneRisk'] == 1 and
                                                 row['AvgGroundTemperatureCelsius'] > 25),
                
                # Overdue inspection with some anomalies
                'inspection_overdue': lambda row: (row['TimeSinceLastInspectionDays'] > 300 and
                                                 (row['TemperatureAnomaliesPast1Year'] > 2 or
                                                  row['VibrationAnomaliesPast1Year'] > 0))
            },
            
            'healthy_conditions': {
                # New cables with minimal issues
                'new_cable': lambda row: (row['AssetAgeYears'] <= 10 and 
                                        row['NumberOfRepairs'] <= 2 and
                                        row['FailureRatePerYear'] <= 0.3),
                
                # Well-maintained cables
                'well_maintained': lambda row: (row['TimeSinceLastInspectionDays'] <= 180 and
                                              row['InspectionDefectSeverityScore'] <= 0.3 and
                                              row['NumberOfFailuresPast3Years'] <= 2),
                
                # Low electrical stress
                'low_stress': lambda row: (row['AvgVoltageDeviationPercent'] <= 4 and
                                         row['OverloadEventCountPast3Years'] <= 3 and
                                         row['PartialDischargeSeverityScore'] <= 0.3)
            }
        }
    
    def learn_patterns_from_seed(self):
        """
        Learn statistical patterns and relationships from the seed dataset
        """
        print("\n🧠 Learning patterns from seed dataset...")
        
        # Prepare features (exclude ID and target)
        feature_cols = [col for col in self.seed_data.columns if col not in ['CableID', 'CableHealthScore']]
        X = self.seed_data[feature_cols]
        y = self.seed_data['CableHealthScore']
        
        # Train pattern recognition model
        self.pattern_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.pattern_model.fit(X, y)
        
        # Learn feature distributions for each class
        for health_score in [0, 1, 2]:
            class_data = self.seed_data[self.seed_data['CableHealthScore'] == health_score]
            self.feature_distributions[health_score] = {}
            
            for feature in feature_cols:
                if feature in ['MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory', 'FloodZoneRisk']:
                    # Categorical features - store value counts
                    self.feature_distributions[health_score][feature] = class_data[feature].value_counts(normalize=True).to_dict()
                else:
                    # Numerical features - store statistical parameters
                    self.feature_distributions[health_score][feature] = {
                        'mean': class_data[feature].mean(),
                        'std': class_data[feature].std(),
                        'min': class_data[feature].min(),
                        'max': class_data[feature].max(),
                        'q25': class_data[feature].quantile(0.25),
                        'q75': class_data[feature].quantile(0.75)
                    }
        
        # Calculate correlation matrix
        self.correlation_matrix = X.corr()
        
        # Evaluate seed model performance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        self.pattern_model.fit(X_train, y_train)
        y_pred = self.pattern_model.predict(X_test)
        seed_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Seed model accuracy: {seed_accuracy:.3f}")
        print(f"✅ Learned distributions for {len(feature_cols)} features")
        
        return seed_accuracy
    
    def apply_domain_rules(self, row):
        """
        Apply domain expert rules to determine cable health score
        """
        # Check critical conditions first (highest priority)
        critical_score = 0
        for rule_name, rule_func in self.domain_rules['critical_conditions'].items():
            if rule_func(row):
                critical_score += 1
        
        if critical_score >= 2:  # Multiple critical conditions
            return 2
        
        # Check at-risk conditions
        at_risk_score = 0
        for rule_name, rule_func in self.domain_rules['at_risk_conditions'].items():
            if rule_func(row):
                at_risk_score += 1
        
        if critical_score >= 1 or at_risk_score >= 2:  # One critical or multiple at-risk
            return 1 if at_risk_score >= critical_score else 2
        
        # Check healthy conditions
        healthy_score = 0
        for rule_name, rule_func in self.domain_rules['healthy_conditions'].items():
            if rule_func(row):
                healthy_score += 1
        
        if healthy_score >= 2:  # Multiple healthy indicators
            return 0
        elif at_risk_score >= 1:  # Some risk factors
            return 1
        else:
            return 0  # Default to healthy if no strong indicators
    
    def synthesize_correlated_features(self, target_class, base_features):
        """
        Synthesize features that maintain realistic correlations
        """
        synthetic_row = base_features.copy()
        
        # Apply correlation-based adjustments
        if target_class == 2:  # Critical
            # Increase correlated risk factors
            if 'AssetAgeYears' in synthetic_row:
                synthetic_row['NumberOfRepairs'] = max(synthetic_row['NumberOfRepairs'], 
                                                     int(synthetic_row['AssetAgeYears'] * 0.3))
                synthetic_row['FailureRatePerYear'] = min(1.0, 
                                                        synthetic_row['FailureRatePerYear'] + 0.2)
        
        elif target_class == 0:  # Healthy
            # Ensure consistency with healthy state
            if synthetic_row['AssetAgeYears'] < 10:
                synthetic_row['NumberOfRepairs'] = min(synthetic_row['NumberOfRepairs'], 2)
                synthetic_row['FailureRatePerYear'] = min(synthetic_row['FailureRatePerYear'], 0.3)
        
        return synthetic_row
    
    def generate_synthetic_batch(self, batch_size=1000, target_distribution=None):
        """
        Generate a batch of synthetic data with improved target labeling
        """
        if target_distribution is None:
            target_distribution = [0.5, 0.3, 0.2]  # Healthy, At Risk, Critical
        
        synthetic_data = []
        
        for i in range(batch_size):
            # Determine target class based on distribution
            target_class = np.random.choice([0, 1, 2], p=target_distribution)
            
            # Generate base features using learned distributions
            synthetic_row = {}
            
            # Generate features based on target class distribution
            for feature, dist_params in self.feature_distributions[target_class].items():
                if feature in ['MaterialType', 'InstallationType', 'FailureRootCauseMajorCategory', 'FloodZoneRisk']:
                    # Categorical features
                    if dist_params:  # Check if distribution exists
                        values = list(dist_params.keys())
                        probs = list(dist_params.values())
                        synthetic_row[feature] = np.random.choice(values, p=probs)
                    else:
                        # Fallback to uniform distribution
                        if feature == 'MaterialType':
                            synthetic_row[feature] = np.random.choice([0, 1])
                        elif feature == 'InstallationType':
                            synthetic_row[feature] = np.random.choice([0, 1, 2])
                        elif feature == 'FailureRootCauseMajorCategory':
                            synthetic_row[feature] = np.random.choice([0, 1, 2])
                        else:  # FloodZoneRisk
                            synthetic_row[feature] = np.random.choice([0, 1])
                else:
                    # Numerical features - use truncated normal distribution
                    mean = dist_params['mean']
                    std = max(dist_params['std'], 0.01)  # Avoid zero std
                    min_val = dist_params['min']
                    max_val = dist_params['max']
                    
                    # Generate value with some noise for diversity
                    noise_factor = 0.1  # 10% noise
                    value = np.random.normal(mean, std * (1 + noise_factor))
                    value = np.clip(value, min_val, max_val)
                    
                    # Round based on feature type
                    if feature in ['AssetAgeYears', 'RatedLifespanYears', 'YearsRemainingWarranty', 
                                 'NumberOfRepairs', 'LastRepairAgeYears', 'NumberOfFailuresPast3Years',
                                 'AvgRepairDurationHours', 'OverloadEventCountPast3Years', 'SwitchingEventsPerYear',
                                 'LastFCITriggerRecencyDays', 'PDThresholdBreachesPast1Year', 'SensitiveCustomerCount',
                                 'ConnectedLoadKW', 'TimeSinceLastInspectionDays', 'AvgGroundTemperatureCelsius',
                                 'EstimatedDowntimeCostPerFailure', 'TemperatureAnomaliesPast1Year', 'VibrationAnomaliesPast1Year']:
                        synthetic_row[feature] = int(round(value))
                    else:
                        synthetic_row[feature] = round(value, 3)
            
            # Apply correlation adjustments
            synthetic_row = self.synthesize_correlated_features(target_class, synthetic_row)
            
            # Apply domain rules to refine target label
            rule_based_label = self.apply_domain_rules(pd.Series(synthetic_row))
            
            # Combine model prediction and rule-based label (weighted approach)
            if hasattr(self, 'pattern_model') and self.pattern_model is not None:
                feature_cols = [col for col in synthetic_row.keys() if col != 'CableID']
                X_synthetic = pd.DataFrame([synthetic_row])[feature_cols]
                model_prediction = self.pattern_model.predict(X_synthetic)[0]
                
                # Weighted combination: 60% rules, 40% model
                if rule_based_label == model_prediction:
                    final_label = rule_based_label
                else:
                    # Use rule-based label with higher weight
                    final_label = rule_based_label if np.random.random() < 0.6 else model_prediction
            else:
                final_label = rule_based_label
            
            # Add cable ID and final label
            synthetic_row['CableID'] = f"CAB{len(self.seed_data) + len(synthetic_data) + 1:05}"
            synthetic_row['CableHealthScore'] = final_label
            
            synthetic_data.append(synthetic_row)
        
        return pd.DataFrame(synthetic_data)
    
    def iterative_generation(self, total_rows=20000, batch_size=2000, iterations=10):
        """
        Iteratively generate dataset with continuous learning
        """
        print(f"\n🔄 Starting iterative generation: {total_rows} rows in {iterations} iterations")
        
        # Start with seed data
        current_dataset = self.seed_data.copy()
        
        # Learn initial patterns
        self.learn_patterns_from_seed()
        
        rows_per_iteration = total_rows // iterations
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
            
            # Generate synthetic batch
            synthetic_batch = self.generate_synthetic_batch(
                batch_size=min(rows_per_iteration, batch_size),
                target_distribution=[0.5, 0.3, 0.2]  # Adjust based on desired distribution
            )
            
            # Combine with existing data
            current_dataset = pd.concat([current_dataset, synthetic_batch], ignore_index=True)
            
            # Retrain pattern model on expanded dataset (every few iterations)
            if (iteration + 1) % 3 == 0:
                print("🔄 Retraining pattern model...")
                feature_cols = [col for col in current_dataset.columns if col not in ['CableID', 'CableHealthScore']]
                X = current_dataset[feature_cols]
                y = current_dataset['CableHealthScore']
                
                self.pattern_model.fit(X, y)
                
                # Evaluate current model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
                self.pattern_model.fit(X_train, y_train)
                y_pred = self.pattern_model.predict(X_test)
                current_accuracy = accuracy_score(y_test, y_pred)
                
                print(f"✅ Current model accuracy: {current_accuracy:.3f}")
                print(f"✅ Dataset size: {len(current_dataset)} rows")
            
            # Print class distribution
            class_dist = current_dataset['CableHealthScore'].value_counts().sort_index()
            print(f"Class distribution: {dict(class_dist)}")
        
        print(f"\n🎉 Iterative generation completed! Final dataset: {len(current_dataset)} rows")
        
        return current_dataset

# Execute Method 1
def run_method_1():
    """
    Execute the iterative learning and synthesis approach
    """
    print("🚀 Method 1: Iterative Learning and Synthesis")
    print("=" * 60)
    
    # Initialize generator with your seed data
    generator = IterativeCableHealthGenerator('cable_health_sample_ordinal_encoded - cable_health_sample_ordinal_encoded.csv')
    
    # Generate improved dataset
    improved_dataset = generator.iterative_generation(
        total_rows=20000,
        batch_size=2000,
        iterations=10
    )
    
    # Save the generated dataset
    output_file = "cable_health_method1_iterative_20k.csv"
    improved_dataset.to_csv(output_file, index=False)
    print(f"✅ Method 1 dataset saved to {output_file}")
    
    # Evaluate final model performance
    feature_cols = [col for col in improved_dataset.columns if col not in ['CableID', 'CableHealthScore']]
    X = improved_dataset[feature_cols]
    y = improved_dataset['CableHealthScore']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Final Model Performance:")
    print(f"Accuracy: {final_accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'At Risk', 'Critical']))
    
    return improved_dataset, final_accuracy
