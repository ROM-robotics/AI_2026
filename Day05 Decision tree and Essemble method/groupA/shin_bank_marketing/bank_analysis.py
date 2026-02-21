import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
df = pd.read_csv('bank.csv')

# 2. Data Preprocessing
# Categorical columns များကို numeric ပြောင်းခြင်း
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature နှင့် Target ခွဲထုတ်ခြင်း (Target သည် 'deposit' ဖြစ်သည်)
X = df.drop('deposit', axis=1)
y = df['deposit']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 5. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

#For Graph Output 

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature Importance တွက်ချက်ခြင်း
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Importance အလိုက် အကြီးကနေ အငယ် စီခြင်း
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 2. Graph ဆွဲခြင်း
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')

plt.title('Top 10 Important Features (Bank Marketing Project)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

#For Tree Photo

from sklearn.tree import plot_tree

# Decision Tree ပုံဆွဲရန် Size သတ်မှတ်ခြင်း
plt.figure(figsize=(20, 10))

# plot_tree သုံးပြီး Visualization လုပ်ခြင်း
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=['No', 'Yes'], 
          filled=True, 
          rounded=True, 
          max_depth=3) # ပုံ အရမ်းမရှုပ်သွားအောင် depth ကို ၃ ဆင့်ပဲ ပြထားပါတယ်

plt.title("Decision Tree Visualization for Bank Marketing")
plt.show()