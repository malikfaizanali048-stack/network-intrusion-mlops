import pandas as pd
import joblib
import shap
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = joblib.load('../models/random_forest_binary.pkl')

data = pd.read_csv('../data/processed_combined_final.csv', nrows=5000)
X = data.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

explainer = shap.TreeExplainer(model)

sample = X.sample(n=100, random_state=42)
shap_values = explainer.shap_values(sample)

shap_values_attack = shap_values[:, :, 1]

shap.summary_plot(shap_values_attack, sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig('../shap_summary.png', dpi=150, bbox_inches='tight')
print("SHAP summary plot saved to shap_summary.png")

mean_abs_shap = pd.DataFrame({
    'feature': sample.columns,
    'mean_abs_shap': abs(shap_values_attack).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 features driving ATTACK predictions:")
print(mean_abs_shap.head(10).to_string(index=False))

mean_abs_shap.to_csv('../shap_feature_importance.csv', index=False)

top_10_json = mean_abs_shap.head(10).to_dict(orient='records')
with open('../src/shap_feature_importance.json', 'w') as f:
    json.dump(top_10_json, f, indent=2)

print("Saved shap_feature_importance.json for API use")