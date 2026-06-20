# diagnose_v2.py
# Train a simple random forest — if RF can't do it, neither can BiLSTM
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report

data = np.load(r'D:\trustgate_pcaps\A12_windowed.npz', allow_pickle=True)

# Flatten windows to single feature vector (mean across time)
X_s_tr = data['X_s_train'].mean(axis=1)   # (N, 71)
X_n_tr = data['X_n_train'].mean(axis=1)   # (N, 19)
X_tr   = np.concatenate([X_s_tr, X_n_tr], axis=1)
y_tr   = data['y_b_train']

X_s_va = data['X_s_val'].mean(axis=1)
X_n_va = data['X_n_val'].mean(axis=1)
X_va   = np.concatenate([X_s_va, X_n_va], axis=1)
y_va   = data['y_b_val']

print("="*60)
print("DIAGNOSIS: Can a simple Random Forest separate attacks?")
print("="*60)

print(f"\nTrain shape: {X_tr.shape}  Val shape: {X_va.shape}")
print(f"Train attacks: {int(y_tr.sum())}  Val attacks: {int(y_va.sum())}")

print(f"\nTraining RF (100 trees, class_weight=balanced)...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_va)
y_prob = rf.predict_proba(X_va)[:, 1]

print(f"\nRF Validation Results:")
print(f"  F1       : {f1_score(y_va, y_pred):.4f}")
print(f"  AUC      : {roc_auc_score(y_va, y_prob):.4f}")
print(f"\n{classification_report(y_va, y_pred, target_names=['Normal','Attack'])}")

print(f"\nFEATURE IMPORTANCES (top 15):")
feature_names = list(data['sensor_cols']) + list(data['network_cols'])
importances = rf.feature_importances_
ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
for name, imp in ranked[:15]:
    print(f"  {name:30s}: {imp:.4f}")

print(f"\nINTERPRETATION:")
print(f"  If RF F1 > 0.70: Data IS separable, BiLSTM needs fixing")
print(f"  If RF F1 < 0.50: Data is NOT separable, need different features")