import joblib
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# 設置隨機種子，確保模型訓練的可重複性
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)

# 1. 讀取數據集
# 從 CSV 文件載入數據
data = pd.read_csv("final_train.csv")

# 數據清理：確保目標變量為數值型態
# errors='coerce' 參數：如果轉換失敗，將無效值設為 NaN
data['Power(mW)'] = pd.to_numeric(data['Power(mW)'], errors='coerce')
data['HourlyTemperature(°C)'] = pd.to_numeric(data['HourlyTemperature(°C)'], errors='coerce')
data['merged_gobalred'] = pd.to_numeric(data['merged_gobalred'], errors='coerce')

# 2. 特徵工程：創建週期性特徵
# 使用正弦和餘弦函數捕捉時間週期性
# 小時週期（0-24）和月份週期（0-12）
data['Hour_sin'] = np.sin(2 * np.pi * data['Hours'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hours'] / 24)
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# 3. 特徵選擇
# 選擇用於模型訓練的特徵和目標變量
X = data[['LocationCode', 'Year', 'Month', 'Day', 'Hours', 'Minute', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos','HourlyTemperature(°C)','merged_gobalred','MinutesOfDay']]
y = data['Power(mW)']

# 4. 數據縮放
# 使用 MinMaxScaler 將特徵和目標變量縮放到 [0, 1] 範圍
# 有助於模型收斂和防止某些特徵主導學習過程
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 5. LightGBM 模型參數配置
# 根據數據特性和模型性能調整超參數
lgb_params = {
   'objective': 'tweedie',  # 回歸任務
    'metric': 'rmse',            # 均方誤差作為評估指標
    'boosting_type': 'gbdt',    # 梯度提升決策樹
    'bagging_freq': 5,          # 數據採樣頻率
    'verbose': -1,               # 不輸出詳細訓練日誌    
    'random_state': 42 ,         # 隨機種子，保證可重複性

'learning_rate': 0.06913626692756031,
 'num_leaves': 254, 
 'max_depth': 15, 
 'feature_fraction': 0.6048364776496044, 
 'bagging_fraction': 0.9982650520404789, 
 'lambda_l1': 0.3461856637051388,
'lambda_l2': 0.22625418661138497

}


# 6. 創建 LightGBM 數據集
# 將縮放後的數據轉換為 LightGBM 可識別的格式
train_data = lgb.Dataset(X_scaled, label=y_scaled)

# 7. 模型訓練
# 使用全部數據進行訓練
num_round = 2373  # 訓練輪數
model = lgb.train(
    lgb_params, 
    train_data, 
    num_boost_round=num_round
)

# 8. 模型預測和性能評估
# 使用訓練好的模型進行預測並反縮放
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

# 計算評估指標
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"均方誤差 (MSE): {mse}")
print(f"平均絕對誤差 (MAE): {mae}")
print(f"R平方分數 (R2): {r2}")

# 9. 結果可視化
# 繪製預測值 vs 真實值散點圖
# plt.figure(figsize=(12, 8), dpi=300)
# plt.scatter(y_true, y_true, c='blue', alpha=0.6, edgecolors='none', s=30, label='真實值')
# plt.scatter(y_true, y_pred, c='red', alpha=0.6, edgecolors='none', s=30, label='預測值')
# plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'g--', lw=3, label='理想預測線')
# plt.xlabel("功率 (mW)", fontsize=12)
# plt.ylabel("功率 (mW)", fontsize=12)
# plt.title("LightGBM: 真實值 vs 預測值", fontsize=15)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
# plt.tight_layout()

# 10. 模型和結果保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(model_dir, f"lightgbm_model_{timestamp}_last.txt")
model.save_model(model_path)

# 保存性能評估圖
# plt.savefig(os.path.join(model_dir, f"lightgbm_predictions_{timestamp}.png"))

# 11. 特徵重要性分析
feature_importance = model.feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# 繪製特徵重要性圖
plt.figure(figsize=(10, 6))
importance_df.plot(x='feature', y='importance', kind='bar')
plt.title("LightGBM 模型特徵重要性")
plt.xlabel("特徵")
plt.ylabel("重要性")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, f"lightgbm_feature_importance_{timestamp}.png"))

print(f"模型已保存到: {model_path}")
print("特徵重要性:")
print(importance_df)

# 9. 保存縮放器
scaler_X_path = os.path.join(model_dir, f"scaler_X_{timestamp}.pkl")
scaler_y_path = os.path.join(model_dir, f"scaler_y_{timestamp}.pkl")

joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)