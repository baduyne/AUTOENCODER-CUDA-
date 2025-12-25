import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Parameters - update paths if needed
FEAT_DIM = 128 * 8 * 8
TRAIN_FEAT_FILE = 'train_features.bin'
TRAIN_LABEL_FILE = 'train_labels.bin'
TEST_FEAT_FILE = 'test_features.bin'
TEST_LABEL_FILE = 'test_labels.bin'
MODEL_OUT = 'svm_model.joblib'

# Load binary features (float32) and labels (int32)
train_feats = np.fromfile(TRAIN_FEAT_FILE, dtype=np.float32)
train_labels = np.fromfile(TRAIN_LABEL_FILE, dtype=np.int32)
test_feats = np.fromfile(TEST_FEAT_FILE, dtype=np.float32)
test_labels = np.fromfile(TEST_LABEL_FILE, dtype=np.int32)

# Reshape
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]
train_feats = train_feats.reshape((n_train, FEAT_DIM))
test_feats = test_feats.reshape((n_test, FEAT_DIM))

print(f'Train shape: {train_feats.shape}, Test shape: {test_feats.shape}')

# Scale features
scaler = StandardScaler()
train_feats = scaler.fit_transform(train_feats)
test_feats = scaler.transform(test_feats)

# Train SVM (RBF) - may be slow for large feature dim; consider LinearSVC or PCA first
print('Training SVM (this may take time)...')
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(train_feats, train_labels)

# Save model and scaler
joblib.dump({'model': clf, 'scaler': scaler}, MODEL_OUT)
print(f'Model saved to {MODEL_OUT}')

# Predict and report
pred = clf.predict(test_feats)
acc = accuracy_score(test_labels, pred)
print(f'Test accuracy: {acc:.4f}')
print(classification_report(test_labels, pred))
