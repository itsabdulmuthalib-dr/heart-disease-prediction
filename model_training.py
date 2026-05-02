import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("dataset.csv")

# Drop unwanted columns
for col in ["id", "dataset"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Convert target
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# ============================
# ENCODING
# ============================
categorical_cols = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "thal"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ============================
# HANDLE MISSING VALUES
# ============================
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ============================
# SPLIT DATA
# ============================
X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# SCALING
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# MODELS
# ============================
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

best_model = None
best_acc = 0
best_name = ""

acc_results = {}

# ============================
# TRAIN MODELS
# ============================
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    acc_results[name] = acc

    print(f"{name}: {round(acc*100,2)}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print("\nBest Model:", best_name)

# ============================
# SAVE FINAL MODEL (IMPORTANT)
# ============================
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as model.pkl")
print("Scaler saved as scaler.pkl")

# ============================
# FINAL EVALUATION
# ============================
y_pred = best_model.predict(X_test)

# Accuracy graph
plt.figure()
plt.bar(list(acc_results.keys()), list(acc_results.values()))
plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))