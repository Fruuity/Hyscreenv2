import lightgbm as lgb
import numpy as np

# Load your trained model
MODEL_PATH = "HyScreen_LSA_SBERT_v2.model";
ranker = lgb.Booster(model_file=MODEL_PATH);

# Define your feature names (same order used during training)
feature_names = ['LSA_Similarity', 'SBERT_Role', 'SBERT_Description', 'SBERT_Education'];

# Get feature importances
importances = ranker.feature_importance(importance_type='gain');

# Convert to percentages
total = np.sum(importances);
percentages = (importances / total) * 100;

print("\n>Feature Importance (from saved model):");
for name, imp, pct in zip(feature_names, importances, percentages):
	print(f"{name}: {imp:.4f} gain ({pct:.2f}%)");

