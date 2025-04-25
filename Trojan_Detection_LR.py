import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

data = pd.read_csv("Trojan_Detection.csv")
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print("Missing values: ", data.isnull().sum())

data.drop(data.columns[[0]],axis=1,inplace=True)
data.drop(['Flow ID',' Source IP',' Destination IP',' Timestamp'],axis=1,inplace=True)
data["Class"] = data["Class"].map({'Benign': 0, 'Trojan': 1})
data['Class'] = data['Class'].astype(int)
print(data.dtypes)
sns.countplot(data=data, x='Class', legend=False)
plt.xticks([0, 1], ['Benign', 'Trojan'])
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 10))
correlation = data.corr()
sns.heatmap(correlation, cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
corr_with_target = correlation['Class'].drop('Class').sort_values(ascending=False)
print("\nTop features positively correlated with Trojan presence:\n", corr_with_target.head(5))
print("\nTop features negatively correlated with Trojan presence:\n", corr_with_target.tail(5))

top_features = corr_with_target.abs().sort_values(ascending=False).head(5).index.tolist()

for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x='Class', y=feature, legend=False)
    plt.xticks([0, 1], ['Benign', 'Trojan'])
    plt.title(f'{feature} vs Trojan Label')
    plt.show()
for feature in top_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q1 + 1.5 * IQR

    data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
for feature in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x='Class', y=feature, legend=False)
    plt.xticks([0, 1], ['Benign', 'Trojan'])
    plt.title(f'{feature} vs Trojan Label')
    plt.show()


X = data[top_features]
y = data["Class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(
    pd.crosstab(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Benign', 'Trojan'],
    yticklabels=['Benign', 'Trojan']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

