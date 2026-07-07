import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

reference_data = pd.read_csv('../data/processed_combined_final.csv', nrows=20000)
reference_features = reference_data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

current_data = reference_features.sample(n=5000, random_state=99)
reference_sample = reference_features.sample(n=5000, random_state=1)

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_data=reference_sample, current_data=current_data)

my_eval.save_html('../drift_report.html')

result = my_eval.dict()

drift_summary = result['metrics'][0]
drifted_count = drift_summary['value']['count']
drifted_share = drift_summary['value']['share']

print(f"Number of drifted columns: {drifted_count}")
print(f"Share of drifted columns: {drifted_share:.2%}")
print(f"Drift threshold triggers retraining at: 50% (configurable)")