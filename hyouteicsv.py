 # CSVを読み込み用
import pandas as pd
# Mean Absolute Error(MAE)用
from sklearn.metrics import mean_absolute_error
# Root Mean Squared Error(RMSE)用
from sklearn.metrics import mean_squared_error
import numpy as np

# CSV読み込み
data = pd.read_csv('/Users/sakumasin/Documents/vscode/zemi/MAEpy/valo.csv')

# 正規化（Min-Maxスケーリング）
label = (data['NLP'] - data['NLP'].min()) / (data['NLP'].max() - data['NLP'].min())
pred = (data['SJT'] - data['SJT'].min()) / (data['SJT'].max() - data['SJT'].min())

# MAE計算
mae = mean_absolute_error(label, pred)
print('MAE : {:.3f}'.format(mae))  # 小数点以下3桁で表示

# RMSE計算
rmse = np.sqrt(mean_squared_error(label, pred))
print('RMSE : {:.3f}'.format(rmse))

# CSVに出力　
df = pd.DataFrame([{
    'MAE': mae,
    'RMSE': rmse
}])
df.to_csv('/Users/sakumasin/Documents/vscode/zemi/MAEpy/valoMAE.csv', index=False, encoding='UTF-8')
