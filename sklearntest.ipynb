# 캘리포니아 주택 다중회귀분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드 및 데이터프레임 생성
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# 데이터 확인
print(df.head())
print("\n기술 통계량:")
print(df.describe())

# 상관관계 분석
correlation = df.corr()
print("\n상관관계:")
print(correlation['Price'].sort_values(ascending=False))

# 데이터 분할
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean squared error: {mse:.4f}")
print(f"R-squared score: {r2:.4f}")

# 특성 중요도 (베타 값) 시각화
importances = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh')
plt.title("Feature Importances in California Housing Price Prediction")
plt.xlabel("Beta Coefficient")
plt.tight_layout()
plt.show()

# 실제 가격 vs 예측 가격 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()
plt.show()
