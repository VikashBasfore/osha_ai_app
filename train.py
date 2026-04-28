import pandas as pd
import joblib
import os

from preprocessing import FullPipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(
    r"C:\Users\vikas\Downloads\ITA Case Detail Data 2024 through 08-31-2025.csv",
    low_memory=False
)

# -------------------------
# SPLIT FIRST (NO LEAKAGE 🔥)
# -------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['incident_outcome']   # better balance
)

# -------------------------
# PIPELINE
# -------------------------
pipeline = FullPipeline()

pipeline.fit(train_df)

train_df = pipeline.transform(train_df)
test_df = pipeline.transform(test_df)

# -------------------------
# SPLIT X, y
# -------------------------
X_train = train_df.drop(columns=['incident_outcome'])
y_train = train_df['incident_outcome']

X_test = test_df.drop(columns=['incident_outcome'])
y_test = test_df['incident_outcome']

# -------------------------
# ALIGN FEATURES
# -------------------------
for col in X_train.columns:
    if col not in X_test:
        X_test[col] = 0

X_test = X_test[X_train.columns]

# -------------------------
# MODEL
# -------------------------
model = LGBMClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42
    )

model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -------------------------
# SAVE
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump({
    "pipeline": pipeline,
    "model": model,
    "columns": X_train.columns
}, "models/final_model.pkl")

print("✅ Model saved successfully")