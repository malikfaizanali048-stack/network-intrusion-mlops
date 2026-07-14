import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime

DRIFT_THRESHOLD = 0.3
ATTACK_RATE_THRESHOLD = 0.5

def check_drift_alert():
    data = pd.read_csv('../data/processed_combined_final.csv', nrows=20000)
    features = data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
    
    current = features.sample(n=3000, random_state=99)
    reference = features.sample(n=3000, random_state=1)
    
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=reference, current_data=current)
    result = my_eval.dict()
    
    drift_share = result['metrics'][0]['value']['share']
    return drift_share

def check_attack_rate_alert():
    data = pd.read_csv('../data/processed_combined_final.csv', nrows=20000)
    attack_rate = data['Label_Binary'].mean()
    return attack_rate

def send_alert(alert_type, message, severity="WARNING"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚨 ALERT [{severity}] - {timestamp}")
    print(f"Type: {alert_type}")
    print(f"Message: {message}")
    print(f"{'='*60}\n")
    
    with open('../alerts.log', 'a') as f:
        f.write(f"{timestamp} | {severity} | {alert_type} | {message}\n")

def check_attack_rate_alert_test():
    data = pd.read_csv('../data/processed_combined_final.csv', nrows=20000)
    attack_only = data[data['Label_Binary'] == 1]
    benign_sample = data[data['Label_Binary'] == 0].sample(n=len(attack_only)//2, random_state=1)
    skewed = pd.concat([attack_only, benign_sample])
    return skewed['Label_Binary'].mean()        

if __name__ == "__main__":
    print("Running alert checks...")
    
    drift_share = check_drift_alert()
    print(f"Current drift share: {drift_share:.2%}")
    if drift_share > DRIFT_THRESHOLD:
        send_alert("DATA_DRIFT", f"Drift share ({drift_share:.2%}) exceeded threshold ({DRIFT_THRESHOLD:.0%}).", severity="HIGH")
    else:
        print("Drift within normal range. No alert triggered.")
    
    attack_rate = check_attack_rate_alert()
    print(f"Current attack rate: {attack_rate:.2%}")
    if attack_rate > ATTACK_RATE_THRESHOLD:
        send_alert(
            "ATTACK_RATE_SPIKE",
            f"Attack prediction rate ({attack_rate:.2%}) exceeded threshold ({ATTACK_RATE_THRESHOLD:.0%}). Possible active attack in progress.",
            severity="CRITICAL"
        )
    else:
        print("Attack rate within normal range. No alert triggered.")
    
    print("\nAlert check complete.")