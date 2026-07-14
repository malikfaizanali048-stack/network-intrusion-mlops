import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

HELD_OUT_ATTACK = "PortScan"

print("Loading full dataset...")
all_data = pd.read_csv('../data/processed_combined_final.csv')

held_out_data = all_data[all_data['Label'] == HELD_OUT_ATTACK].copy()
training_pool = all_data[all_data['Label'] != HELD_OUT_ATTACK].copy()

print(f"Held out {len(held_out_data)} '{HELD_OUT_ATTACK}' samples entirely from training.")
print(f"Training pool size: {len(training_pool)}")

X = training_pool.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
y = training_pool['Label_Binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training classifier WITHOUT PortScan knowledge...")
holdout_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
holdout_classifier.fit(X_train, y_train)

y_pred = holdout_classifier.predict(X_test)
print("\nPerformance on normal held-out test set (no PortScan involved):")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))

print("\nTraining Isolation Forest safety net (same as before, benign-only)...")
benign_only = X_train[y_train == 0]
holdout_iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
holdout_iso_forest.fit(benign_only)

joblib.dump(holdout_classifier, '../models/holdout_classifier.pkl')
joblib.dump(holdout_iso_forest, '../models/holdout_iso_forest.pkl')
held_out_data.to_csv('../data/held_out_portscan.csv', index=False)

print("\nSaved: holdout_classifier.pkl, holdout_iso_forest.pkl, held_out_portscan.csv")