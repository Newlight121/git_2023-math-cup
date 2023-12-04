import pandas as pd
import numpy as np

# 生成一个日期范围
# dates = pd.date_range(start='2023-01-01', periods=2048, freq='D')
#
# # 创建一个带有趋势和季节性的模拟时间序列数据
# data = pd.DataFrame({
#     'ds': dates,
#     'y': np.sin(1.5 * np.linspace(0, 20, num=2048)) + np.random.normal(0, 0.4, 2048)
# })
#
# data['y'] = (data['y'] - data['y'].mean()) / (data['y'].std())
# data['ds'] = pd.to_datetime(data['ds'], unit='s')
# # 展示生成的数据
# print(data.head())


from prophet import Prophet

# from neuralprophet import NeuralProphet

import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv("extend1.csv", index_col=0)
category_data = df[df['fenlei2'] == 5]
# 可视化生成的数据
data = category_data

data_time_value = data.loc[:, '2022/11/1':'2023/5/30']
prophet_input = pd.DataFrame()
prophet_input['ds'] = data_time_value.columns
# for i in range(data_time_value.shape[0]):
prophet_input['y'] = data_time_value.iloc[0].values

data = prophet_input
plt.figure(figsize=(12, 6))
plt.plot(data['ds'], data['y'])
plt.title('Generated Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

from prophet import Prophet

# 假设的节假日日期
double11 = pd.DataFrame({
    'holiday': 'double11',
    'ds': pd.to_datetime(['2022/11/1', '2022/11/2', '2022/11/3', '2022/11/4', '2022/11/5',
                          '2022/11/6', '2022/11/7', '2022/11/8', '2022/11/9', '2022/11/10', '2022/11/11', '2023/6/1',
                          '2023/6/2', '2023/6/3', '2023/6/4', '2023/6/5',
                          '2023/6/6', '2023/6/7', '2023/6/8', '2023/6/9', '2023/6/10', '2023/6/11', '2023/6/12',
                          '2023/6/13', '2023/6/14', '2023/6/15',
                          '2023/6/16', '2023/6/17', '2023/6/18', '2023/6/19', '2023/6/20', '2023/6/21', '2023/6/22',
                          '2023/6/23', '2023/6/24', '2023/6/25', '2023/6/26', '2023/6/27',
                          '2023/6/28', '2023/6/29', '2023/6/30'])  # 假设国庆节和圣诞节
})

yuandan = pd.DataFrame({
    'holiday': 'yuandan',
    'ds': pd.to_datetime(['2023/1/1', '2023/1/2', '2023/1/3', '2023/1/4', '2023/1/5'])  # 假设国庆节和圣诞节
})

holidays = pd.concat([double11, yuandan])

# 初始化Prophet模型并添加假日效应
model = Prophet(changepoint_range=0.5, yearly_seasonality=True, seasonality_prior_scale=10.0,
model = Prophet(changepoint_range=0.5, yearly_seasonality=True, seasonality_prior_scale=10.0,
                holidays_prior_scale=50.0, changepoint_prior_scale=0.05, weekly_seasonality=True,
                daily_seasonality=True, n_changepoints=25, holidays=holidays)

# model = Prophet(holidays=holidays)

# model.add_seasonality(name="2yearly", period=730, fourier_order=1)
# model.add_seasonality(name="yearly", period=365, fourier_order=1)
# model.add_seasonality(name='quarterly', period=91.5, fourier_order=2)
# model.add_seasonality(name='monthly', period=30.5, fourier_order=1)
# model.add_seasonality(name='weekly', period=7, fourier_order=2)

# model = Prophet()
# model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# model = Prophet(weekly_seasonality=True)
# model.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)

# 拟合模型
model.fit(data)

# 创建一个用于预测未来数据的数据框
future = model.make_future_dataframe(periods=60)  # 预测未来30天
future.tail()
# 进行预测
forecast = model.predict(future)

fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
plt.title('Prophet Forecast with Holidays')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

model.plot_components(forecast)
plt.title('Prophet Forecast with Holidays')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
