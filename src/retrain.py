import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("network-intrusion-detection")

print("Loading data for retraining...")
all_data = pd.read_csv('../data/processed_combined_final.csv')
X = all_data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
y = all_data['Label_Binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print("Training new candidate model...")
new_model = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
new_model.fit(X_train, y_train)

y_pred = new_model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"New model - Precision: {precision:.4f}, Recall: {recall:.4f}")

with mlflow.start_run(run_name="random_forest_retrain_candidate") as run:
    mlflow.log_param("random_state", 123)
    mlflow.log_param("trigger", "manual_retrain_test")
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    mlflow.sklearn.log_model(
        new_model,
        "model",
        registered_model_name="network-intrusion-classifier"
    )
    
    run_id = run.info.run_id

print(f"New candidate model registered. Run ID: {run_id}")