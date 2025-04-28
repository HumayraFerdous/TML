import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import xgboost as xgb

df = pd.read_csv('Trojan_Detection.csv')  # Replace with your actual file name

print("Dataset Shape:", df.shape)
df.head()

df.drop(df.columns[[0]],axis=1,inplace=True)
df.drop(['Flow ID',' Source IP',' Destination IP',' Timestamp'],axis=1,inplace=True)
df["Class"] = df["Class"].map({'Benign': 0, 'Trojan': 1})
df['Class'] = df['Class'].astype(int)


X = df.drop('Class', axis=1)
y = df['Class']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}


results = {}

for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])


    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]


    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    conf = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': acc,
        'ROC AUC': roc,
        'Confusion Matrix': conf
    }


    print(f"==== {name} ====")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("Confusion Matrix:\n", conf)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")


summary_df = pd.DataFrame(results).T[['Accuracy', 'ROC AUC']]

summary_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison: Accuracy and ROC AUC')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid()
plt.xticks(rotation=45)
plt.show()