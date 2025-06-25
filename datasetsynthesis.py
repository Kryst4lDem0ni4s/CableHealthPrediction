import pandas as pd
import numpy as np

# Number of rows you want to generate
num_rows = 20000

# Fixed mappings for categorical fields with Ordinal Encoding
material_type_mapping = {"Copper": 0, "Aluminum": 1}
installation_type_mapping = {"Underground": 0, "Overhead": 1, "Submarine": 2}
failure_root_cause_mapping = {"Mechanical Damage": 0, "Insulation Failure": 1, "Corrosion": 2}
health_score_mapping = {"Healthy": 0, "At Risk": 1, "Critical": 2}

# Function to generate one row of data
def generate_sample_row(cable_id):
    return {
        "CableID": f"CAB{cable_id:05}",
        "AssetAgeYears": np.random.randint(1, 30),
        "RatedLifespanYears": np.random.choice([20, 25, 30]),
        "YearsRemainingWarranty": np.random.randint(0, 15),
        "MaterialType": np.random.choice(list(material_type_mapping.values())),
        "InstallationType": np.random.choice(list(installation_type_mapping.values())),
        "InsulationResistanceTrend": np.round(np.random.uniform(-0.1, 0), 3),
        "NumberOfRepairs": np.random.randint(0, 10),
        "LastRepairAgeYears": np.random.randint(0, 10),
        "NumberOfFailuresPast3Years": np.random.randint(0, 15),
        "FailureRatePerYear": np.round(np.random.uniform(0, 1), 2),
        "AvgRepairDurationHours": np.random.randint(4, 24),
        "FailureRootCauseMajorCategory": np.random.choice(list(failure_root_cause_mapping.values())),
        "AvgVoltageDeviationPercent": np.round(np.random.uniform(1, 8), 2),
        "PeakLoadKW": np.random.randint(100, 800),
        "OverloadEventCountPast3Years": np.random.randint(0, 10),
        "SwitchingEventsPerYear": np.random.randint(5, 25),
        "FCITriggerRatePerYear": np.round(np.random.uniform(0, 1), 2),
        "LastFCITriggerRecencyDays": np.random.randint(1, 365),
        "PartialDischargeSeverityScore": np.round(np.random.uniform(0, 1), 2),
        "PDThresholdBreachesPast1Year": np.random.randint(0, 5),
        "SensitiveCustomerCount": np.random.randint(0, 10),
        "ConnectedLoadKW": np.random.randint(1000, 10000),
        "InspectionDefectSeverityScore": np.round(np.random.uniform(0, 1), 2),
        "TimeSinceLastInspectionDays": np.random.randint(1, 365),
        "AvgGroundTemperatureCelsius": np.random.randint(10, 40),
        "SoilCorrosivityIndex": np.round(np.random.uniform(0, 1), 2),
        "FloodZoneRisk": np.random.choice([0, 1]),
        "CriticalityScore": np.round(np.random.uniform(0, 1), 2),
        "EstimatedDowntimeCostPerFailure": np.random.randint(1000, 50000),
        "TemperatureAnomaliesPast1Year": np.random.randint(0, 5),
        "VibrationAnomaliesPast1Year": np.random.randint(0, 2),
        "CableHealthScore": np.random.choice(list(health_score_mapping.values()))
    }

# Generate the dataset
data = [generate_sample_row(i + 1) for i in range(num_rows)]
df = pd.DataFrame(data)

# Save to CSV
output_file = "cable_health_sample_ordinal_encoded_20000.csv"
df.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")
