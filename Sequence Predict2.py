from datetime import datetime
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from keras.models import load_model

warnings.filterwarnings(action='ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    
    # Memory growth must be set before GPUs have been initialized
    print(e)
tf.debugging.set_log_device_placement(True)

table = pd.read_csv("/workspace/附件5编码_fianl2.csv", encoding="utf-8", index_col=False)
table["data_len"] = np.nan
table_data = table["data"] # 获取data数据
for i in range(len(table)):
  table["data_len"].iloc[i] = len(eval(table["data"].iloc[i]))
for fenlei_index in range(0,22):
    category_data = table[table['fenlei'] == fenlei_index]
    if not category_data.empty:
        print(fenlei_index,min(category_data["data_len"]))

# 定义滑动窗口的宽度与预测长度
window_size = 23
# 预测接下来的y_pre_len个时间点长度
next_predict = 30

y_pre_len = 5

for index, line in enumerate(table_data):
    category_class = table['fenlei'].iloc[index]  #获取类别
    category_ylist = line

    category_ylist = np.expand_dims(np.array(eval(category_ylist)), axis=0)
    model = load_model("/workspace/model_save2/"+"model_class_"+str(category_class) + "_choosen" +".h5")
    
    # 预测未来时间步的值
    future_input = category_ylist[:, -window_size:]  # 选取最后数据作为初始预测未来的输入
    future_input = future_input.reshape(-1, window_size, 1)
    future_predictions = []
    
    for _ in range(next_predict):
        future_prediction = model.predict(future_input)
        future_predictions.append(future_prediction)
        # 更新输入窗口，添加新的预测值，删除第一个时间步的数据
        # future_input = np.concatenate((future_input[:, y_pre_len:, :], future_prediction.reshape(-1,y_pre_len,1)), axis=1)
        future_input = np.concatenate((future_input[:, 1:, :], np.round(future_prediction[:,0:1]).reshape(-1,1,1)), axis=1) # 步进1进行输入
        # print(future_input[1, -8:-1].reshape(1, -1))

    # 将预测结果转换为NumPy数组
    future_predictions = np.array(future_predictions).transpose(1,0,2).reshape(-1, next_predict, y_pre_len)

    # 创建一个长度为 next_predict+y_pre_len 的数组来存储输出
    final_output = np.zeros((future_predictions.shape[0],next_predict+y_pre_len-1))
    weights = np.zeros(next_predict+y_pre_len-1)

    # 对预测结果进行滑动窗口整合
    for i in range(future_predictions.shape[1]):
        final_output[:,i:i+5] += future_predictions[:,i]
        weights[i:i+5] += 1

    # 计算最终输出的平均值
    final_output = final_output /weights
    future_predictions = final_output.reshape(-1, next_predict + y_pre_len - 1, 1)

    future_predictions = np.round(future_predictions)

    select_index = 0
    plot_series_data = category_ylist[select_index]
    plot_future_prediction = future_predictions[select_index]
    # 绘制时间序列数据和预测值，使用不同颜色区分
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(plot_series_data)), plot_series_data, label='Original Data', color='blue')
    plt.plot(np.arange(len(plot_series_data)-1, len(plot_series_data) + next_predict+y_pre_len-1), np.insert(plot_future_prediction,0,plot_series_data[-1])
            label='Predicted Data', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

pass