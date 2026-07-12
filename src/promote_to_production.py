import mlflow
import sys

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()

MODEL_NAME = "network-intrusion-classifier"

if len(sys.argv) != 2:
    print("Usage: python promote_to_production.py <version_number>")
    sys.exit(1)

version_to_promote = sys.argv[1]

print(f"You are about to promote version {version_to_promote} of '{MODEL_NAME}' to PRODUCTION.")
confirm = input("Type 'yes' to confirm: ")

if confirm.strip().lower() == "yes":
    client.set_registered_model_alias(MODEL_NAME, "production", version_to_promote)
    print(f"✅ Version {version_to_promote} is now PRODUCTION.")
else:
    print("Promotion cancelled.")