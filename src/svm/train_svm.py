import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Parameters
FEAT_DIM = 128 * 8 * 8
SELECTED_LABELS = [0, 1, 2, 3, 4, 5]  
TRAIN_FEAT_FILE = './extracted_feature/train_features.bin'
TRAIN_LABEL_FILE = './extracted_feature/train_labels.bin'
TEST_FEAT_FILE = './extracted_feature/test_features.bin'
TEST_LABEL_FILE = './extracted_feature/test_labels.bin'
MODEL_OUT = 'svm_model_filtered.joblib'

# Load binary features & labels
train_feats = np.fromfile(TRAIN_FEAT_FILE, dtype=np.float32)
train_labels = np.fromfile(TRAIN_LABEL_FILE, dtype=np.int32)
test_feats = np.fromfile(TEST_FEAT_FILE, dtype=np.float32)
test_labels = np.fromfile(TEST_LABEL_FILE, dtype=np.int32)

# Reshape features
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]
train_feats = train_feats.reshape((n_train, FEAT_DIM))
test_feats = test_feats.reshape((n_test, FEAT_DIM))

print("Sample feature (first 10 values):")
print(train_feats[0][:10])

print("Before filtering:")
print(f"Train: {train_feats.shape}  Test: {test_feats.shape}")

# ============================
#  Lọc train theo 3 nhãn
# ============================
train_mask = np.isin(train_labels, SELECTED_LABELS)
train_feats = train_feats[train_mask]
train_labels = train_labels[train_mask]

# ============================
#  Lọc test theo 3 nhãn
# ============================
test_mask = np.isin(test_labels, SELECTED_LABELS)
test_feats = test_feats[test_mask]
test_labels = test_labels[test_mask]

print("After filtering:")
print(f"Train: {train_feats.shape}  Test: {test_feats.shape}")
print(f"Labels kept: {SELECTED_LABELS}")

# ============================
# Scale features
# ============================
scaler = StandardScaler()
train_feats = scaler.fit_transform(train_feats)
test_feats = scaler.transform(test_feats)

# ============================
# Train SVM
# ============================
print("Training SVM on filtered labels...")
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(train_feats, train_labels)

# Save model + scaler
joblib.dump({'model': clf, 'scaler': scaler}, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# ============================
#  Evaluation
# ============================
pred = clf.predict(test_feats)
acc = accuracy_score(test_labels, pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(test_labels, pred))
