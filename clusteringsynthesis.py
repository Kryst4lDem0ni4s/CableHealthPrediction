"""
Method 2: Clustering-Based Target Generation for Cable Health Dataset
====================================================================
This approach generates a new dataset using domain expert knowledge and applies
advanced clustering techniques to derive accurate target labels.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

from iterativesynthesis import run_method_1

class ClusteringBasedCableHealthGenerator:
    """
    Generate cable health dataset using domain knowledge and clustering for target labels
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Domain knowledge parameters
        self.domain_parameters = self._define_domain_parameters()
        self.feature_weights = self._define_feature_weights()
        
    def _define_domain_parameters(self):
        """
        Define realistic parameter ranges based on cable industry knowledge
        """
        return {
            'AssetAgeYears': {'min': 1, 'max': 30, 'critical_threshold': 25},
            'RatedLifespanYears': {'values': [20, 25, 30], 'weights': [0.3, 0.4, 0.3]},
            'NumberOfRepairs': {'healthy_max': 2, 'critical_min': 6},
            'FailureRatePerYear': {'healthy_max': 0.3, 'critical_min': 0.7},
            'PartialDischargeSeverityScore': {'healthy_max': 0.3, 'critical_min': 0.7},
            'InspectionDefectSeverityScore': {'healthy_max': 0.3, 'critical_min': 0.7},
            'AvgVoltageDeviationPercent': {'healthy_max': 4.0, 'critical_min': 6.5},
            'OverloadEventCountPast3Years': {'healthy_max': 3, 'critical_min': 7},
            'TimeSinceLastInspectionDays': {'healthy_max': 180, 'critical_min': 300},
            'SoilCorrosivityIndex': {'healthy_max': 0.4, 'critical_min': 0.7},
            'NumberOfFailuresPast3Years': {'healthy_max': 3, 'critical_min': 8}
        }
    
    def _define_feature_weights(self):
        """
        Define feature importance weights for clustering
        """
        return {
            # Critical health indicators (highest weight)
            'FailureRatePerYear': 3.0,
            'NumberOfFailuresPast3Years': 3.0,
            'PartialDischargeSeverityScore': 2.5,
            'InspectionDefectSeverityScore': 2.5,
            
            # Important risk factors
            'AssetAgeYears': 2.0,
            'NumberOfRepairs': 2.0,
            'AvgVoltageDeviationPercent': 2.0,
            'OverloadEventCountPast3Years': 1.8,
            
            # Moderate importance
            'TimeSinceLastInspectionDays': 1.5,
            'PDThresholdBreachesPast1Year': 1.5,
            'SoilCorrosivityIndex': 1.3,
            'TemperatureAnomaliesPast1Year': 1.2,
            'VibrationAnomaliesPast1Year': 1.2,
            
            # Lower importance but still relevant
            'FCITriggerRatePerYear': 1.0,
            'FloodZoneRisk': 1.0,
            'CriticalityScore': 1.0
        }
    
    def generate_realistic_features(self, num_rows=20000):
        """
        Generate realistic cable features using domain knowledge and correlations
        """
        print(f"🔧 Generating {num_rows} realistic cable feature rows...")
        
        data = []
        
        for i in range(num_rows):
            # Start with basic cable characteristics
            asset_age = np.random.randint(1, 31)
            rated_lifespan = np.random.choice(
                self.domain_parameters['RatedLifespanYears']['values'],
                p=self.domain_parameters['RatedLifespanYears']['weights']
            )
            
            # Calculate remaining warranty (decreases with age)
            max_warranty = max(0, rated_lifespan - asset_age)
            years_remaining_warranty = np.random.randint(0, max_warranty + 1)
            
            # Material and installation type (affects other parameters)
            material_type = np.random.choice([0, 1])  # 0: Copper, 1: Aluminum
            installation_type = np.random.choice([0, 1, 2])  # Underground, Overhead, Submarine
            
            # Age-correlated degradation
            age_factor = asset_age / 30.0  # Normalize age
            
            # Generate correlated failure and repair data
            base_failure_rate = 0.1 + (age_factor * 0.6)  # Increases with age
            failure_rate = np.clip(np.random.normal(base_failure_rate, 0.2), 0, 1)
            
            # Number of repairs correlates with age and failure rate
            expected_repairs = age_factor * 8 + failure_rate * 5
            num_repairs = max(0, int(np.random.poisson(expected_repairs)))
            
            # Failures in past 3 years correlate with failure rate
            expected_failures = failure_rate * 3 + np.random.normal(0, 1)
            failures_past_3_years = max(0, int(expected_failures))
            
            # Last repair age (if repairs exist)
            last_repair_age = np.random.randint(0, min(asset_age + 1, 10)) if num_repairs > 0 else asset_age
            
            # Electrical characteristics
            avg_voltage_deviation = np.random.uniform(1, 8)
            peak_load = np.random.randint(100, 800)
            
            # Overload events correlate with voltage deviation and load
            overload_factor = (avg_voltage_deviation / 8.0) * (peak_load / 800.0)
            overload_events = int(np.random.poisson(overload_factor * 8))
            
            # Partial discharge correlates with age and electrical stress
            pd_base = age_factor * 0.4 + (avg_voltage_deviation / 8.0) * 0.3
            pd_severity = np.clip(np.random.normal(pd_base, 0.2), 0, 1)
            pd_breaches = int(np.random.poisson(pd_severity * 4))
            
            # Inspection and defect data
            inspection_days = np.random.randint(1, 365)
            
            # Defect severity correlates with age and other issues
            defect_base = age_factor * 0.3 + (num_repairs / 10.0) * 0.4 + (failure_rate * 0.3)
            defect_severity = np.clip(np.random.normal(defect_base, 0.2), 0, 1)
            
            # Environmental factors
            ground_temp = np.random.randint(10, 40)
            soil_corrosivity = np.random.uniform(0, 1)
            flood_zone = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% in flood zones
            
            # Environmental stress affects other parameters
            env_stress = (soil_corrosivity * 0.5) + (flood_zone * 0.3) + ((ground_temp - 20) / 20 * 0.2)
            
            # Anomalies correlate with overall cable stress
            stress_level = (age_factor + failure_rate + pd_severity + defect_severity) / 4
            temp_anomalies = int(np.random.poisson(stress_level * 3))
            vibration_anomalies = int(np.random.poisson(stress_level * 1.5))
            
            # Business impact factors
            sensitive_customers = np.random.randint(0, 10)
            connected_load = np.random.randint(1000, 10000)
            criticality_score = np.random.uniform(0, 1)
            downtime_cost = np.random.randint(1000, 50000)
            
            # Additional operational parameters
            switching_events = np.random.randint(5, 25)
            fci_trigger_rate = np.random.uniform(0, 1)
            fci_recency = np.random.randint(1, 365)
            
            # Insulation resistance trend (negative indicates degradation)
            insulation_trend = np.random.uniform(-0.1, 0)
            
            # Repair duration correlates with repair complexity
            avg_repair_duration = np.random.randint(4, 24)
            
            # Failure root cause
            failure_root_cause = np.random.choice([0, 1, 2])  # Mechanical, Insulation, Corrosion
            
            # Create row
            row = {
                'CableID': f"CAB{i+1:05}",
                'AssetAgeYears': asset_age,
                'RatedLifespanYears': rated_lifespan,
                'YearsRemainingWarranty': years_remaining_warranty,
                'MaterialType': material_type,
                'InstallationType': installation_type,
                'InsulationResistanceTrend': round(insulation_trend, 3),
                'NumberOfRepairs': num_repairs,
                'LastRepairAgeYears': last_repair_age,
                'NumberOfFailuresPast3Years': failures_past_3_years,
                'FailureRatePerYear': round(failure_rate, 2),
                'AvgRepairDurationHours': avg_repair_duration,
                'FailureRootCauseMajorCategory': failure_root_cause,
                'AvgVoltageDeviationPercent': round(avg_voltage_deviation, 2),
                'PeakLoadKW': peak_load,
                'OverloadEventCountPast3Years': overload_events,
                'SwitchingEventsPerYear': switching_events,
                'FCITriggerRatePerYear': round(fci_trigger_rate, 2),
                'LastFCITriggerRecencyDays': fci_recency,
                'PartialDischargeSeverityScore': round(pd_severity, 2),
                'PDThresholdBreachesPast1Year': pd_breaches,
                'SensitiveCustomerCount': sensitive_customers,
                'ConnectedLoadKW': connected_load,
                'InspectionDefectSeverityScore': round(defect_severity, 2),
                'TimeSinceLastInspectionDays': inspection_days,
                'AvgGroundTemperatureCelsius': ground_temp,
                'SoilCorrosivityIndex': round(soil_corrosivity, 2),
                'FloodZoneRisk': flood_zone,
                'CriticalityScore': round(criticality_score, 2),
                'EstimatedDowntimeCostPerFailure': downtime_cost,
                'TemperatureAnomaliesPast1Year': temp_anomalies,
                'VibrationAnomaliesPast1Year': vibration_anomalies
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} realistic cable feature rows")
        
        return df
    
    def create_health_score_features(self, df):
        """
        Create composite features specifically for health score clustering
        """
        print("🧮 Creating composite health score features...")
        
        df_features = df.copy()
        
        # Normalize age relative to lifespan
        df_features['AgeLifespanRatio'] = df_features['AssetAgeYears'] / df_features['RatedLifespanYears']
        
        # Failure intensity score
        df_features['FailureIntensity'] = (
            df_features['FailureRatePerYear'] * 0.4 +
            (df_features['NumberOfFailuresPast3Years'] / 15.0) * 0.3 +
            (df_features['NumberOfRepairs'] / 10.0) * 0.3
        )
        
        # Electrical stress composite
        df_features['ElectricalStress'] = (
            (df_features['AvgVoltageDeviationPercent'] / 8.0) * 0.4 +
            (df_features['OverloadEventCountPast3Years'] / 10.0) * 0.3 +
            df_features['PartialDischargeSeverityScore'] * 0.3
        )
        
        # Maintenance responsiveness
        df_features['MaintenanceScore'] = (
            (1 - df_features['TimeSinceLastInspectionDays'] / 365.0) * 0.5 +
            (1 - df_features['InspectionDefectSeverityScore']) * 0.5
        )
        
        # Environmental risk
        df_features['EnvironmentalRisk'] = (
            df_features['SoilCorrosivityIndex'] * 0.4 +
            df_features['FloodZoneRisk'] * 0.3 +
            ((df_features['AvgGroundTemperatureCelsius'] - 20) / 20.0).clip(0, 1) * 0.3
        )
        
        # Anomaly frequency
        df_features['AnomalyFrequency'] = (
            df_features['TemperatureAnomaliesPast1Year'] / 5.0 +
            df_features['VibrationAnomaliesPast1Year'] / 2.0 +
            df_features['PDThresholdBreachesPast1Year'] / 5.0
        ).clip(0, 1)
        
        # Overall degradation score
        df_features['DegradationScore'] = (
            df_features['AgeLifespanRatio'] * 0.2 +
            df_features['FailureIntensity'] * 0.3 +
            df_features['ElectricalStress'] * 0.25 +
            (1 - df_features['MaintenanceScore']) * 0.15 +
            df_features['EnvironmentalRisk'] * 0.1
        )
        
        print("✅ Created 6 composite health score features")
        
        return df_features
    
    def apply_multiple_clustering_algorithms(self, df):
        """
        Apply multiple clustering algorithms and select the best one
        """
        print("🔍 Applying multiple clustering algorithms...")
        
        # Select features for clustering (focus on health-related features)
        clustering_features = [
            'FailureRatePerYear', 'NumberOfFailuresPast3Years', 'NumberOfRepairs',
            'PartialDischargeSeverityScore', 'InspectionDefectSeverityScore',
            'AvgVoltageDeviationPercent', 'OverloadEventCountPast3Years',
            'AssetAgeYears', 'TimeSinceLastInspectionDays', 'SoilCorrosivityIndex',
            'PDThresholdBreachesPast1Year', 'TemperatureAnomaliesPast1Year',
            'VibrationAnomaliesPast1Year', 'AgeLifespanRatio', 'FailureIntensity',
            'ElectricalStress', 'MaintenanceScore', 'EnvironmentalRisk',
            'AnomalyFrequency', 'DegradationScore'
        ]
        
        X_clustering = df[clustering_features].copy()
        
        # Apply feature weights
        for feature in clustering_features:
            if feature in self.feature_weights:
                X_clustering[feature] *= self.feature_weights[feature]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clustering)
        
        # Test multiple clustering algorithms
        clustering_results = {}
        
        # 1. K-Means with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
        
        clustering_results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': kmeans_silhouette,
            'calinski_harabasz': kmeans_calinski,
            'model': kmeans
        }
        
        # 2. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=3, random_state=self.random_state)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
        gmm_calinski = calinski_harabasz_score(X_scaled, gmm_labels)
        
        clustering_results['GMM'] = {
            'labels': gmm_labels,
            'silhouette': gmm_silhouette,
            'calinski_harabasz': gmm_calinski,
            'model': gmm
        }
        
        # 3. Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=3)
        agg_labels = agg_clustering.fit_predict(X_scaled)
        agg_silhouette = silhouette_score(X_scaled, agg_labels)
        agg_calinski = calinski_harabasz_score(X_scaled, agg_labels)
        
        clustering_results['Agglomerative'] = {
            'labels': agg_labels,
            'silhouette': agg_silhouette,
            'calinski_harabasz': agg_calinski,
            'model': agg_clustering
        }
        
        # Select best clustering based on silhouette score
        best_algorithm = max(clustering_results.keys(), 
                           key=lambda x: clustering_results[x]['silhouette'])
        
        best_labels = clustering_results[best_algorithm]['labels']
        best_silhouette = clustering_results[best_algorithm]['silhouette']
        
        print(f"📊 Clustering Results:")
        for alg, results in clustering_results.items():
            print(f"  {alg}: Silhouette={results['silhouette']:.3f}, Calinski-Harabasz={results['calinski_harabasz']:.1f}")
        
        print(f"🏆 Best algorithm: {best_algorithm} (Silhouette: {best_silhouette:.3f})")
        
        return best_labels, best_algorithm, clustering_results, X_scaled, scaler
    
    def map_clusters_to_health_scores(self, df, cluster_labels):
        """
        Map cluster labels to cable health scores (0, 1, 2) using domain knowledge
        """
        print("🎯 Mapping clusters to health scores...")
        
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        # Analyze cluster characteristics
        cluster_stats = {}
        for cluster in range(3):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            
            cluster_stats[cluster] = {
                'count': len(cluster_data),
                'avg_failure_rate': cluster_data['FailureRatePerYear'].mean(),
                'avg_age': cluster_data['AssetAgeYears'].mean(),
                'avg_repairs': cluster_data['NumberOfRepairs'].mean(),
                'avg_pd_severity': cluster_data['PartialDischargeSeverityScore'].mean(),
                'avg_defect_severity': cluster_data['InspectionDefectSeverityScore'].mean(),
                'avg_degradation': cluster_data['DegradationScore'].mean()
            }
        
        # Create composite risk score for each cluster
        for cluster in cluster_stats:
            stats = cluster_stats[cluster]
            risk_score = (
                stats['avg_failure_rate'] * 0.25 +
                (stats['avg_age'] / 30.0) * 0.2 +
                (stats['avg_repairs'] / 10.0) * 0.2 +
                stats['avg_pd_severity'] * 0.15 +
                stats['avg_defect_severity'] * 0.1 +
                stats['avg_degradation'] * 0.1
            )
            cluster_stats[cluster]['risk_score'] = risk_score
        
        # Sort clusters by risk score and map to health scores
        sorted_clusters = sorted(cluster_stats.keys(), key=lambda x: cluster_stats[x]['risk_score'])
        
        # Map: lowest risk -> Healthy (0), medium risk -> At Risk (1), highest risk -> Critical (2)
        cluster_to_health = {
            sorted_clusters[0]: 0,  # Healthy
            sorted_clusters[1]: 1,  # At Risk
            sorted_clusters[2]: 2   # Critical
        }
        
        # Apply mapping
        df_with_clusters['CableHealthScore'] = df_with_clusters['Cluster'].map(cluster_to_health)
        
        print(f"📈 Cluster to Health Score Mapping:")
        for cluster, health_score in cluster_to_health.items():
            stats = cluster_stats[cluster]
            health_name = ['Healthy', 'At Risk', 'Critical'][health_score]
            print(f"  Cluster {cluster} -> {health_score} ({health_name})")
            print(f"    Count: {stats['count']}, Risk Score: {stats['risk_score']:.3f}")
            print(f"    Avg Failure Rate: {stats['avg_failure_rate']:.3f}")
            print(f"    Avg Age: {stats['avg_age']:.1f} years")
        
        # Remove intermediate columns
        final_df = df_with_clusters.drop(['Cluster', 'AgeLifespanRatio', 'FailureIntensity', 
                                        'ElectricalStress', 'MaintenanceScore', 
                                        'EnvironmentalRisk', 'AnomalyFrequency', 
                                        'DegradationScore'], axis=1)
        
        return final_df, cluster_to_health, cluster_stats
    
    def visualize_clustering_results(self, df, cluster_labels, X_scaled):
        """
        Visualize clustering results and health score distribution
        """
        print("📊 Creating clustering visualizations...")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA visualization
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Clustering Results (PCA Visualization)')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Health score distribution
        health_counts = df['CableHealthScore'].value_counts().sort_index()
        axes[0, 1].bar(['Healthy', 'At Risk', 'Critical'], health_counts.values, 
                      color=['green', 'orange', 'red'], alpha=0.7)
        axes[0, 1].set_title('Cable Health Score Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # Feature importance for clustering
        key_features = ['FailureRatePerYear', 'AssetAgeYears', 'NumberOfRepairs', 
                       'PartialDischargeSeverityScore', 'InspectionDefectSeverityScore']
        
        for i, feature in enumerate(key_features[:2]):
            ax = axes[1, i]
            for health_score in [0, 1, 2]:
                data = df[df['CableHealthScore'] == health_score][feature]
                ax.hist(data, alpha=0.6, label=['Healthy', 'At Risk', 'Critical'][health_score],
                       bins=20, color=['green', 'orange', 'red'][health_score])
            ax.set_title(f'{feature} Distribution by Health Score')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_complete_dataset(self, num_rows=20000):
        """
        Generate complete dataset using clustering-based approach
        """
        print(f"🚀 Generating complete dataset with {num_rows} rows using clustering approach")
        
        # Step 1: Generate realistic features
        df = self.generate_realistic_features(num_rows)
        
        # Step 2: Create composite health features
        df_enhanced = self.create_health_score_features(df)
        
        # Step 3: Apply clustering
        cluster_labels, best_algorithm, clustering_results, X_scaled, scaler = self.apply_multiple_clustering_algorithms(df_enhanced)
        
        # Step 4: Map clusters to health scores
        final_df, cluster_mapping, cluster_stats = self.map_clusters_to_health_scores(df_enhanced, cluster_labels)
        
        # Step 5: Visualize results
        self.visualize_clustering_results(final_df, cluster_labels, X_scaled)
        
        return final_df, cluster_mapping, cluster_stats, best_algorithm

# Execute Method 2
def run_method_2():
    """
    Execute the clustering-based target generation approach
    """
    print("🚀 Method 2: Clustering-Based Target Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = ClusteringBasedCableHealthGenerator(random_state=42)
    
    # Generate dataset
    dataset, cluster_mapping, cluster_stats, best_algorithm = generator.generate_complete_dataset(20000)
    
    # Save the generated dataset
    output_file = "cable_health_method2_clustering_20k.csv"
    dataset.to_csv(output_file, index=False)
    print(f"✅ Method 2 dataset saved to {output_file}")
    
    # Evaluate with a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    feature_cols = [col for col in dataset.columns if col not in ['CableID', 'CableHealthScore']]
    X = dataset[feature_cols]
    y = dataset['CableHealthScore']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Method 2 Model Performance:")
    print(f"Best Clustering Algorithm: {best_algorithm}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'At Risk', 'Critical']))
    
    return dataset, accuracy

# Main execution
if __name__ == "__main__":
    print("🔬 Cable Health Dataset Generation - Advanced Methods")
    print("=" * 70)
    
    # Run Method 1
    print("\n" + "="*70)
    dataset_1, accuracy_1 = run_method_1()
    
    # Run Method 2  
    print("\n" + "="*70)
    dataset_2, accuracy_2 = run_method_2()
    
    # Compare results
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    print(f"Method 1 (Iterative Learning): {accuracy_1:.3f} accuracy")
    print(f"Method 2 (Clustering-Based): {accuracy_2:.3f} accuracy")
    
    if accuracy_1 > accuracy_2:
        print("🏆 Method 1 (Iterative Learning) performs better!")
        recommended_file = "cable_health_method1_iterative_20k.csv"
    else:
        print("🏆 Method 2 (Clustering-Based) performs better!")
        recommended_file = "cable_health_method2_clustering_20k.csv"
    
    print(f"📁 Recommended dataset: {recommended_file}")
