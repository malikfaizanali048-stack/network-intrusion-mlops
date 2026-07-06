import requests
import pandas as pd
import time
import random

API_URL = "http://127.0.0.1:8000/predict"

test_data = pd.read_csv('../data/processed_combined_final.csv')
sample_data = test_data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass']).sample(n=200, random_state=1)
true_labels = test_data.loc[sample_data.index, 'Label_Binary']

for i, (idx, row) in enumerate(sample_data.iterrows()):
    features = row.to_dict()
    payload = {"features": features}
    
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        true_label = "ATTACK" if true_labels.loc[idx] == 1 else "BENIGN"
        
        print(f"[{i+1}] Predicted: {result['prediction']} (confidence: {result['confidence']:.2f}) | Actual: {true_label}")
    except Exception as e:
        print(f"Error on request {i+1}: {e}")
    
    time.sleep(random.uniform(0.5, 2))