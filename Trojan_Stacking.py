import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_curve,roc_auc_score
import lightgbm as lgb


data = pd.read_csv("Trojan_Detection.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print("Missing values: ", data.isnull().sum())

data["Class"] = data["Class"].map({'Benign': 0, 'Trojan': 1})
data['Class'] = data['Class'].astype(int)
data.drop(['Unnamed: 0', 'Flow ID'],axis=1,inplace=True)
print(data.dtypes)

encoder = LabelEncoder()
data[' Source IP'] = encoder.fit_transform(data[' Source IP'])
print(data[' Source IP'].nunique())
data[' Destination IP'] = encoder.fit_transform(data[' Destination IP'])
print(data[' Destination IP'].nunique())

data[' Timestamp'] = pd.to_datetime(data[' Timestamp'], format='%d/%m/%Y %H:%M:%S')

data['year'] = data[' Timestamp'].dt.year
data['month'] = data[' Timestamp'].dt.month
data['day'] = data[' Timestamp'].dt.day
data['hour'] = data[' Timestamp'].dt.hour
data['minute'] = data[' Timestamp'].dt.minute
data['weekday'] = data[' Timestamp'].dt.weekday  # 0 = Monday, 6 = Sunday
data['dayofyear'] = data[' Timestamp'].dt.dayofyear

data.drop(' Timestamp',axis=1,inplace=True)

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

# 1. Mutual Information
mi_scores = mutual_info_classif(X_scaled_df, y)
mi_selected = pd.Series(mi_scores, index=X.columns)
mi_selected = mi_selected.sort_values(ascending=False)
mi_top = mi_selected[mi_selected > 0.01].index.tolist()
print("Mutual Info Top Features:", mi_top)

cor_matrix = X_scaled_df.corrwith(y).abs()
cor_top = cor_matrix[cor_matrix > 0.1].index.tolist()
print("Correlation Top Features:", cor_top)

selected_features = list(set(mi_top).intersection(set(cor_top)))
print("Final selected features:", selected_features)

X_selected = X_scaled_df[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
base_learners = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(random_state=42))
]

# Define meta-learner
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# Train the model
stacking_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacking_clf.predict(X_test)
y_proba = stacking_clf.predict_proba(X_test)[:, 1]

print("\nStacking Classifier Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='Stacking Classifier (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()