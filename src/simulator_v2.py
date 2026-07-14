import requests
import pandas as pd
import numpy as np
import time
import random

API_URL = "http://127.0.0.1:8000/predict"

print("Loading traffic pools...")
main_data = pd.read_csv('../data/processed_combined_final.csv')
held_out_portscan = pd.read_csv('../data/held_out_portscan.csv')

benign_pool = main_data[main_data['Label_Binary'] == 0].drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
known_attack_pool = main_data[main_data['Label_Binary'] == 1].drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
novel_attack_pool = held_out_portscan.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

print(f"Benign pool: {len(benign_pool)} | Known attacks: {len(known_attack_pool)} | Novel (PortScan): {len(novel_attack_pool)}")

def generate_burst_traffic(total_requests=150):
    """
    Generates a realistic mixed traffic sequence:
    - Variable attack ratio per 'session' (not fixed)
    - Occasional bursts of rapid requests (simulating an attack wave)
    - Occasional novel/unseen attack traffic mixed in
    """
    traffic_sequence = []

    while len(traffic_sequence) < total_requests:
        session_type = random.choices(
            ['calm', 'attack_burst', 'novel_burst'],
            weights=[0.6, 0.3, 0.1]
        )[0]

        if session_type == 'calm':
            session_size = random.randint(5, 15)
            attack_ratio = random.uniform(0.0, 0.15)
            for _ in range(session_size):
                is_attack = random.random() < attack_ratio
                row = known_attack_pool.sample(1) if is_attack else benign_pool.sample(1)
                traffic_sequence.append(('KNOWN_ATTACK' if is_attack else 'BENIGN', row, random.uniform(0.5, 2.0)))

        elif session_type == 'attack_burst':
            session_size = random.randint(10, 30)
            for _ in range(session_size):
                row = known_attack_pool.sample(1)
                traffic_sequence.append(('KNOWN_ATTACK', row, random.uniform(0.02, 0.15)))

        elif session_type == 'novel_burst':
            session_size = random.randint(5, 20)
            for _ in range(session_size):
                row = novel_attack_pool.sample(1)
                traffic_sequence.append(('NOVEL_ATTACK', row, random.uniform(0.05, 0.2)))

    return traffic_sequence[:total_requests]

print("\nGenerating realistic mixed traffic sequence...")
sequence = generate_burst_traffic(total_requests=150)

print(f"Sending {len(sequence)} requests with variable ratios and burst timing...\n")

results_log = []

for i, (true_category, row, delay) in enumerate(sequence):
    features = row.iloc[0].to_dict()
    payload = {"features": features}

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        predicted = result.get('prediction', 'ERROR')
        confidence = result.get('confidence', 0)

        flag = ""
        if true_category in ('KNOWN_ATTACK', 'NOVEL_ATTACK') and predicted == 'BENIGN':
            flag = " ⚠️ MISSED"

        print(f"[{i+1}/{len(sequence)}] True: {true_category:14s} | Predicted: {predicted:8s} | Confidence: {confidence:.2f}{flag}")

        results_log.append({
            'true_category': true_category,
            'predicted': predicted,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Error on request {i+1}: {e}")

    time.sleep(delay)

print("\n" + "="*60)
results_df = pd.DataFrame(results_log)
summary = results_df.groupby('true_category')['predicted'].apply(
    lambda x: (x == 'ATTACK').mean() if x.name != 'BENIGN' else (x == 'BENIGN').mean()
)
print("Detection summary by traffic type:")
print(summary)
print("="*60)

results_df.to_csv('../simulation_results.csv', index=False)
print("\nFull results saved to simulation_results.csv")