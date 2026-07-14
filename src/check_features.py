import pandas as pd
import joblib

classifier = joblib.load('../models/holdout_classifier.pkl')

held_out = pd.read_csv('../data/held_out_portscan.csv')
X_held_out = held_out.drop(columns=['Label', 'Label_Binary', 'Label_Multiclass'])

print("Classifier expects features:", classifier.n_features_in_)
print("Held-out data has features:", X_held_out.shape[1])

if hasattr(classifier, 'feature_names_in_'):
    match = list(X_held_out.columns) == list(classifier.feature_names_in_)
    print("Column order match:", match)
    if not match:
        print("\nMismatch details:")
        print("Classifier's first 5 features:", list(classifier.feature_names_in_)[:5])
        print("Data's first 5 features:", list(X_held_out.columns)[:5])
else:
    print("Classifier doesn't store feature names - checking manually")
    print("Held-out columns:", list(X_held_out.columns)[:5])