import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Import SVM Classifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

# 1. Load the dataset
data = pd.read_csv('dataset_full.csv')

# 2. Data Inspection
print(f"Dataset shape: {data.shape}")
print(data.head())
print(data.isnull().sum())

# 3. Handle missing values
data.dropna(inplace=True)  # Alternatively, impute if necessary

# 4. Separate features and target
X = data.drop('phishing', axis=1)
y = data['phishing']

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Feature Selection
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support(indices=True)]
print("Selected Features:", selected_features)

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 9. Initialize and train the SVM model
svm_classifier = SVC(
    kernel='rbf',          # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    probability=True,      # Enable probability estimates
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
svm_classifier.fit(X_train_res, y_train_res)

# 10. Predictions
y_pred = svm_classifier.predict(X_test)
y_prob = svm_classifier.predict_proba(X_test)[:,1]

# 11. Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 12. Save the model and scaler
joblib.dump(svm_classifier, 'phishing_detector_svm.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
