import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.font_manager as fm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import platform
import os

system = platform.system()

if system == 'Windows':
    font_path = r'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    # 또는
    # font_path = r'C:\Windows\Fonts\gulim.ttc'  # 굴림
elif system == 'Darwin':  # macOS
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # Apple SD Gothic Neo
    # 또는
    # font_path = '/Library/Fonts/NanumGothic.ttf'  # 나눔고딕 (설치된 경우)
elif system == 'Linux':
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 우분투의 나눔고딕

font_prop = fm.FontProperties(fname=font_path, size=12)
# matplotlib 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

warnings.filterwarnings('ignore')


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
print("\n가격과의 상관관계:")
print(correlation['Price'].sort_values(ascending=False))

# 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("특성 간 상관관계 히트맵")
plt.tight_layout()
plt.show()

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

# 훈련 데이터에 대한 예측 및 평가
y_train_pred = model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\n훈련 데이터 성능:")
print(f"Mean squared error: {train_mse:.4f}")
print(f"R-squared score: {train_r2:.4f}")

# 테스트 데이터에 대한 예측 및 평가
y_test_pred = model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n테스트 데이터 성능:")
print(f"Mean squared error: {test_mse:.4f}")
print(f"R-squared score: {test_r2:.4f}")

# 특성 중요도 (베타 값) 시각화
importances = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh')
plt.title("특성 중요도 (베타 계수)")
plt.xlabel("베타 계수")
plt.tight_layout()
plt.show()

# 실제 가격 vs 예측 가격 산점도 (테스트 데이터)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("실제 가격")
plt.ylabel("예측 가격")
plt.title("실제 가격 vs 예측 가격 (테스트 데이터)")
plt.tight_layout()
plt.show()

# 잔차 플롯
residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='--')
plt.xlabel("예측 가격")
plt.ylabel("잔차")
plt.title("잔차 플롯")
plt.tight_layout()
plt.show()
