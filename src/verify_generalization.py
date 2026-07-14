import pandas as pd

all_data = pd.read_csv('../data/processed_combined_final.csv')

portscan = all_data[all_data['Label'] == 'PortScan']
benign = all_data[all_data['Label'] == 'BENIGN']
ddos = all_data[all_data['Label'] == 'DDoS']

top_features = ['Destination Port', 'Total Length of Bwd Packets', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']

comparison = pd.DataFrame({
    'PortScan_mean': portscan[top_features].mean(),
    'BENIGN_mean': benign[top_features].mean(),
    'DDoS_mean': ddos[top_features].mean()
})

print(comparison)