import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Configuration
data_path = "/Users/karinpitlik/Desktop/DataScience/Thesis/Case1_Nov_2022_23_25/Ens/Graphs/UNITED/4by4/3_hours/training_data_24km.csv"

# 2. Read CSV
df = pd.read_csv(data_path)

# drop NANs
print(f"Original size: {len(df)}")
df = df.dropna()
print(f"Size after dropping NaNs: {len(df)}")

# 3. Define Features (X) and Target (y)
X = df[["LPI", "KI", "CAPE2D", "PREC_RATE"]]
y = df["Target_Lightning"]

# random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Random Forest Training
# n_estimators=100
clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
print("Training the model...")
clf.fit(X_train, y_train)

# 5. Evaluation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
custom_threshold = 0.25
y_pred_new = (y_prob >= custom_threshold).astype(int)

print(f"\n--- Results with Threshold {custom_threshold} ---")
# שימי לב: ה-Accuracy אולי ירד, אבל אנחנו מחפשים Recall > 0
print(confusion_matrix(y_test, y_pred_new))

print("\nDetailed Report:")
print(classification_report(y_test, y_pred_new))
