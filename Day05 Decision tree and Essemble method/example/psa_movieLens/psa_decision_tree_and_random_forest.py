#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:05:15 2026

@author: mr_robot
"""
import pandas as pd

DATA_PATH = r'/home/mr_robot/Desktop/AI_2026/Day05 Decision tree and Essemble method/example/psa_movieLens'
# CSV ဖိုင်များကို Pandas သုံးပြီး ဖတ်ယူခြင်း
ratings = pd.read_csv(f'{DATA_PATH}/ratings.csv') # Rating ပေးထားချက်များကို ဖတ်ခြင်း
movies = pd.read_csv(f'{DATA_PATH}/movies.csv')

df = ratings.merge(movies, on='movieId')

genres = df['genres'].str.get_dummies(sep='|')

user_stats = df.groupby('userId').agg({
    'rating': ['mean', 'count', 'std'] # ပျမ်းမျှအမှတ်၊ အကြိမ်ရေ နှင့် သွေဖည်ကိန်းတို့ကို တွက်ခြင်း
}).reset_index()

# Column နာမည်များကို အလွယ်တကူ ခေါ်သုံးနိုင်ရန် ပြောင်းလဲသတ်မှတ်ခြင်း
user_stats.columns = ['userId', 'user_avg_rating', 'user_rating_count', 'user_rating_std']

# Standard Deviation (std) တန်ဖိုး မရှိသည့် (NaN) နေရာများတွင် 0 ဖြည့်သွင်းခြင်း
user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

# ရုပ်ရှင်တစ်ကားချင်းစီရဲ့ Rating အခြေအနေများကို တွက်ချက်ခြင်း (Movie Statistics)
movie_stats = df.groupby('movieId').agg({
    'rating': ['mean', 'count'] # ရုပ်ရှင်တစ်ခုချင်းစီရဲ့ ပျမ်းမျှအမှတ် နှင့် လူဘယ်နှစ်ယောက် ပေးထားလဲဆိုသည်ကို တွက်ခြင်း
}).reset_index()

# Movie statistics အတွက် column နာမည်များ သတ်မှတ်ခြင်း
movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_rating_count']

# Final Dataset ပြင်ဆင်ခြင်း
# မူလ Data ထဲသို့ User stats နှင့် Movie stats များကို ပေါင်းထည့်ခြင်း
df_final = df.merge(user_stats, on='userId').merge(movie_stats, on='movieId')

# One-hot encoding လုပ်ထားသော Genres column များကို ဘေးတိုက် (Axis=1) ပေါင်းထည့်ခြင်း
df_final = pd.concat([df_final, genres], axis=1)

df_final['high_rating'] = (df_final['rating'] >= 4).astype(int)

# Features and Target ခွဲခြားခြင်း
# Model ထဲသို့ ထည့်သွင်းတွက်ချက်မည့် Column နာမည်များကို စုစည်းခြင်း
feature_cols = ['user_avg_rating', 'user_rating_count', 'user_rating_std',
                'movie_avg_rating', 'movie_rating_count'] + genres.columns.tolist()

X = df_final[feature_cols]      # သင်ကြားပေးမည့် အချက်အလက်များ (Input Data)
y = df_final['high_rating']    # ခန့်မှန်းခိုင်းမည့် အဖြေ (Target Label)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,           # Tree ၏ အလွှာအနက်ကို ၅ ထပ်အထိပဲ ကန့်သတ်ခြင်း (Overfitting မဖြစ်စေရန်)
    min_samples_split=10,  # အချက်အလက် အနည်းဆုံး ၁၀ ခုရှိမှသာ အကိုင်းအခက် ဆက်ခွဲရန် သတ်မှတ်ခြင်း
    random_state=42        # ရလဒ် တည်ငြိမ်စေရန်အတွက် random state သတ်မှတ်ခြင်း
)

# Training Data (X_train, y_train) ကို သုံးပြီး Model ကို သင်ကြားပေးခြင်း
dt_model.fit(X_train, y_train)

# စမ်းသပ်ရန်ဖယ်ထားသော X_test ကို သုံးပြီး Rating များကို ခန့်မှန်းခြင်း
y_pred = dt_model.predict(X_test)

# အဖြေမှန် (y_test) နှင့် ခန့်မှန်းချက် (y_pred) မည်မျှ ကိုက်ညီသလဲ (Accuracy) ကို တွက်ချက်ခြင်း
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

plt.figure(figsize=(20, 10))

# Decision Tree ကို ပုံဖော်ရန် plot_tree function ကို သုံးခြင်း
plot_tree(dt_model, 
          feature_names=feature_cols,      # အသုံးပြုထားသည့် feature အမည်များ ထည့်သွင်းခြင်း
          class_names=['Low', 'High'],     # ခွဲခြားမည့် အတန်းအစား အမည်များ သတ်မှတ်ခြင်း
          filled=True,                     # ရလဒ်အလိုက် အရောင်များ ဖြည့်သွင်းခြင်း (ဥပမာ- High ဆိုလျှင် အပြာ)
          rounded=True,                    # အကွက်လေးများကို ထောင့်ဝိုင်းပုံစံ ပြုလုပ်ခြင်း
        #   impurity=False,
        #   label='none',
          fontsize=8)                      # စာလုံးအရွယ်အစား သတ်မှတ်ခြင်း

plt.title('Decision Tree Visualization') # ပုံ၏ ခေါင်းစဉ်ကို သတ်မှတ်ခြင်း
plt.tight_layout() # ပုံကို နေရာလွတ်မကျန် သေသေသပ်သပ် ဖြစ်စေခြင်း
plt.show() # ပုံကို ထုတ်ပြခြင်း


# 1. Random Forest Library ကို ခေါ်ယူခြင်း
from sklearn.ensemble import RandomForestClassifier

# 2. Model တည်ဆောက်ခြင်း
# n_estimators=100 ဆိုသည်မှာ Decision Tree ပေါင်း ၁၀၀ ကို အသုံးပြုမည်ဟု ဆိုလိုသည်
# max_features='sqrt' သည် သစ်ပင်တစ်ပင်စီအတွက် feature များကို ကျပန်းရွေးချယ်ရာတွင် သုံးသောနည်းလမ်းဖြစ်သည်
rf_model = RandomForestClassifier(n_estimators=100, 
                                  max_depth=10, 
                                  random_state=42) 

# 3. Model ကို စာသင်ပေးခြင်း (Training)
rf_model.fit(X_train, y_train)

# 4. ခန့်မှန်းခိုင်းခြင်း (Prediction)
y_pred_rf = rf_model.predict(X_test)

# 5. ရလဒ်ကို စစ်ဆေးခြင်း
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
