import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "network-intrusion-classifier"

print("Loading held-out validation data...")
all_data = pd.read_csv('../data/processed_combined_final.csv')
X = all_data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
y = all_data['Label_Binary']

_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading current PRODUCTION model...")
client = mlflow.MlflowClient()
production_version = client.get_model_version_by_alias(MODEL_NAME, "production")
production_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")

print("Loading LATEST (candidate) model version...")
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(all_versions, key=lambda v: int(v.version))
candidate_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_version.version}")

print(f"\nComparing Production (v{production_version.version}) vs Candidate (v{latest_version.version})...")

prod_pred = production_model.predict(X_val)
prod_precision = precision_score(y_val, prod_pred)
prod_recall = recall_score(y_val, prod_pred)

cand_pred = candidate_model.predict(X_val)
cand_precision = precision_score(y_val, cand_pred)
cand_recall = recall_score(y_val, cand_pred)

print(f"\nProduction (v{production_version.version}): Precision={prod_precision:.4f}, Recall={prod_recall:.4f}")
print(f"Candidate  (v{latest_version.version}): Precision={cand_precision:.4f}, Recall={cand_recall:.4f}")

passes_gate = (cand_precision >= prod_precision) and (cand_recall >= prod_recall)

print(f"\n{'='*50}")
if passes_gate:
    print(f"✅ VALIDATION PASSED: Candidate v{latest_version.version} meets or exceeds production performance.")
    print(f"Candidate is ELIGIBLE for promotion to production (requires manual approval).")
else:
    print(f"❌ VALIDATION FAILED: Candidate v{latest_version.version} does not meet production performance.")
    print(f"Candidate will NOT be promoted.")
print(f"{'='*50}")