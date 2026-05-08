# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:46:54 2026

@author: AI
"""
import pandas as pd

df = pd.read_csv("ecommerce_customer_churn_dataset.csv")

print(df.head())
df.info()
df.describe()
df.head()
df['Churned'].value_counts()
df['Churned'].value_counts(normalize=True)


print(df['Churned'].value_counts())
print(df['Churned'].value_counts(normalize=True))

result = df.groupby('Churned').mean(numeric_only=True)
print(result.T)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df.boxplot(column='Cart_Abandonment_Rate', by='Churned')
plt.title("Cart_Abandonment_Rate vs Churn")
plt.suptitle('') 
plt.savefig("Cart_Abandonment_Rate_boxplot.png")
plt.show()

diff = df.groupby('Churned').mean(numeric_only=True)
diff = (diff.loc[1] - diff.loc[0]).abs().sort_values(ascending=False)
print(diff.head(8))

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df.boxplot(column='Credit_Balance', by='Churned')
plt.title("Cart_Abandonment_Rate vs Churn")
plt.suptitle('') 
plt.savefig("Credit_Balance_boxplot.png")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df.boxplot(column='Lifetime_Value', by='Churned')
plt.title("Lifetime_Value vs Churn")
plt.suptitle('') 
plt.savefig("Lifetime_Value_boxplot.png")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df.boxplot(column='Average_Order_Value', by='Churned')
plt.title("Average_Order_Valuee vs Churn")
plt.suptitle('') 
plt.savefig("Average_Order_Value_boxplot.png")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
df.boxplot(column='Days_Since_Last_Purchase', by='Churned')
plt.title("Days_Since_Last_Purchase vs Churn")
plt.suptitle('') 
plt.savefig("Days_Since_Last_Purchase_boxplot.png")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. 选变量
X = df[['Credit_Balance', 'Lifetime_Value', 
        'Cart_Abandonment_Rate', 'Average_Order_Value', 
        'Days_Since_Last_Purchase']]
y = df['Churned']

# 2. 缺失值处理
X = X.fillna(X.mean())

# 3. 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. 建模
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train, y_train)

# 6. 评估
print("Accuracy:", model.score(X_test, y_test))


# ====== 使用全部数值特征建模 + 输出系数表 ======
# 1. 准备数据
X_all = df.select_dtypes(include=['number']).drop('Churned', axis=1)
feature_names = X_all.columns.tolist()   # 这就是20个特征的名字
y = df['Churned']

# 2. 缺失值填充
X_all = X_all.fillna(X_all.mean())

# 3. 标准化
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# 4. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_all_scaled, y, test_size=0.2, random_state=42)

# 5. 建模
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train, y_train)

# 6. 评估
print("Accuracy (all features):", model.score(X_test, y_test))

# 7. 输出系数表
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print("\n===== 各特征对流失概率的影响 =====")
print("正系数 → 特征值越大，流失概率越高（增加流失风险）")
print("负系数 → 特征值越大，流失概率越低（降低流失风险）\n")
print(coef_df.to_string())


# ⭐ 单独给RF用的X（不要标准化）
X_rf = df.select_dtypes(include=['number']).drop('Churned', axis=1)

X_rf = X_rf.fillna(X_rf.mean())

# 保存列名
X_columns = X_rf.columns

# 划分数据
from sklearn.model_selection import train_test_split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)

# 训练RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,class_weight='balanced',
                            random_state=42)
rf.fit(X_train_rf, y_train_rf)

# Feature importance ⭐
import pandas as pd
importance = pd.DataFrame({
    'Feature': X_columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance)

import matplotlib.pyplot as plt

# 取前10个最重要变量
top10 = importance.head(10)

# 画图
plt.figure(figsize=(8,6))
plt.barh(top10['Feature'], top10['Importance'])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ===== RF 分类报告 =====

from sklearn.metrics import classification_report

y_pred_rf = rf.predict(X_test_rf)

print("\n===== Random Forest 分类报告 =====")
print(classification_report(y_test_rf, y_pred_rf))

import os
print(os.getcwd())
