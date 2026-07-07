import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

data = pd.read_csv('../data/processed_combined_final.csv', nrows=50000)

benign_data = data[data['Label_Binary'] == 0].drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])
attack_data = data[data['Label_Binary'] == 1].drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

print("Benign samples:", benign_data.shape[0])
print("Attack samples:", attack_data.shape[0])

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_data=benign_data.sample(n=min(3000, len(benign_data)), random_state=1), 
                      current_data=attack_data.sample(n=min(3000, len(attack_data)), random_state=1))

my_eval.save_html('../drift_report_real_attack.html')

result = my_eval.dict()
drift_summary = result['metrics'][0]
drifted_count = drift_summary['value']['count']
drifted_share = drift_summary['value']['share']

print(f"Number of drifted columns: {drifted_count}")
print(f"Share of drifted columns: {drifted_share:.2%}")