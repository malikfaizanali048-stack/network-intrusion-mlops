import pandas as pd
import joblib
import numpy as np

classifier = joblib.load('../models/holdout_classifier.pkl')
iso_forest = joblib.load('../models/holdout_iso_forest.pkl')

held_out = pd.read_csv('../data/held_out_portscan.csv')
X_held_out = held_out.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

print(f"Testing on {len(X_held_out)} PortScan samples the models have NEVER seen...\n")

classifier_pred = classifier.predict(X_held_out)
classifier_detection_rate = classifier_pred.mean()

print(f"SUPERVISED CLASSIFIER (never trained on PortScan):")
print(f"  Detected as ATTACK: {classifier_pred.sum()} / {len(classifier_pred)} ({classifier_detection_rate:.2%})")

iso_pred = iso_forest.predict(X_held_out)
iso_pred_binary = np.where(iso_pred == -1, 1, 0)
iso_detection_rate = iso_pred_binary.mean()

print(f"\nISOLATION FOREST SAFETY NET (unsupervised, benign-only training):")
print(f"  Detected as ANOMALY: {iso_pred_binary.sum()} / {len(iso_pred_binary)} ({iso_detection_rate:.2%})")

either_caught = np.maximum(classifier_pred, iso_pred_binary)
combined_rate = either_caught.mean()

print(f"\nCOMBINED (classifier OR anomaly detector flags it):")
print(f"  Caught by at least one system: {either_caught.sum()} / {len(either_caught)} ({combined_rate:.2%})")

missed_by_both = ((classifier_pred == 0) & (iso_pred_binary == 0)).sum()
print(f"\n  Missed by BOTH systems entirely: {missed_by_both} ({missed_by_both/len(X_held_out):.2%})")