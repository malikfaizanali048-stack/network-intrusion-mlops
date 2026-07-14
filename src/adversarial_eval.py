import pandas as pd
import numpy as np
import joblib

model = joblib.load('../models/random_forest_binary.pkl')

print("Loading attack samples for perturbation testing...")
data = pd.read_csv('../data/processed_combined_final.csv')
attack_samples = data[data['Label_Binary'] == 1].drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

test_sample = attack_samples.sample(n=2000, random_state=42)

baseline_pred = model.predict(test_sample)
baseline_detection_rate = baseline_pred.mean()
print(f"\nBaseline detection rate (unperturbed attacks): {baseline_detection_rate:.2%}")

PERTURBABLE_FEATURES = {
    'Flow Duration': 0.15,
    'Total Fwd Packets': 0.15,
    'Total Backward Packets': 0.15,
    'Total Length of Fwd Packets': 0.15,
    'Total Length of Bwd Packets': 0.15,
    'Flow Bytes/s': 0.15,
    'Flow Packets/s': 0.15,
    'Fwd IAT Mean': 0.20,
    'Bwd IAT Mean': 0.20,
}

def perturb_features(df, perturbation_map, direction='random'):
    perturbed = df.copy()
    for feature, pct in perturbation_map.items():
        if feature not in perturbed.columns:
            continue
        if direction == 'random':
            noise = np.random.uniform(-pct, pct, size=len(perturbed))
        elif direction == 'reduce':
            noise = np.random.uniform(-pct, 0, size=len(perturbed))
        elif direction == 'increase':
            noise = np.random.uniform(0, pct, size=len(perturbed))
        perturbed[feature] = perturbed[feature] * (1 + noise)
        perturbed[feature] = perturbed[feature].clip(lower=0)
    return perturbed

print("\nRunning perturbation scenarios...\n")

results = {}

for direction in ['random', 'reduce', 'increase']:
    perturbed_sample = perturb_features(test_sample, PERTURBABLE_FEATURES, direction=direction)
    perturbed_pred = model.predict(perturbed_sample)
    detection_rate = perturbed_pred.mean()
    evasion_rate = 1 - detection_rate
    results[direction] = evasion_rate
    print(f"Perturbation direction: {direction:10s} | Detection: {detection_rate:.2%} | Evasion: {evasion_rate:.2%}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline (no perturbation) detection: {baseline_detection_rate:.2%}")
for direction, evasion in results.items():
    print(f"After '{direction}' perturbation (±15-20% on 9 features): {evasion:.2%} evaded detection")

    print("\n" + "="*60)
print("ESCALATING PERTURBATION INTENSITY")
print("="*60)

for intensity in [0.30, 0.50, 0.75, 1.0]:
    scaled_map = {k: intensity for k in PERTURBABLE_FEATURES}
    perturbed_sample = perturb_features(test_sample, scaled_map, direction='reduce')
    perturbed_pred = model.predict(perturbed_sample)
    detection_rate = perturbed_pred.mean()
    evasion_rate = 1 - detection_rate
    print(f"Intensity ±{intensity:.0%} (reduce direction) | Detection: {detection_rate:.2%} | Evasion: {evasion_rate:.2%}")

print("\n" + "="*60)
print("TARGETED ATTACK: Perturbing top SHAP features specifically")
print("="*60)

top_shap_features = {
    'Destination Port': 0.5,
    'Total Length of Bwd Packets': 0.5,
    'Avg Bwd Segment Size': 0.5,
    'Average Packet Size': 0.5,
    'Max Packet Length': 0.5,
    'Init_Win_bytes_backward': 0.5,
    'Subflow Bwd Bytes': 0.5,
    'Bwd Packet Length Std': 0.5,
    'Fwd Packet Length Max': 0.5,
    'Packet Length Variance': 0.5,
}

targeted_perturbed = perturb_features(test_sample, top_shap_features, direction='reduce')
targeted_pred = model.predict(targeted_perturbed)
targeted_detection = targeted_pred.mean()
targeted_evasion = 1 - targeted_detection

print(f"Targeted perturbation (top 10 SHAP features, ±50%, reduce): Detection: {targeted_detection:.2%} | Evasion: {targeted_evasion:.2%}")    