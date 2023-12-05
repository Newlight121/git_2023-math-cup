from datetime import datetime
from re import T
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

table_review = pd.read_csv("/workspace/1111.csv", encoding="gb18030", index_col=0)

table = pd.read_csv("/workspace/q1_result.csv", encoding="utf-8", index_col=0)

merged_table = pd.merge(pd.concat([table.loc[:,['seller', 'product', 'house',"fenlei2"]],table.loc[:,'2022/12/1':'2023/5/30']],axis=1) ,table_review.loc[:,['seller', 'product', 'house','data']],  on=['seller', 'product', 'house',])
table_save = merged_table.copy()

for i in range(11):
    table_save["extend_data" + str(i)] = np.nan  # 初始化nan

for i,line in enumerate(merged_table["data"]):
    line = eval(line)
    for j in range(len(line)):
        table_save.loc[i,"extend_data" + str(10-j)] = line[-(j+1)]

table_save_data = table_save["data"]
table_save = table_save.drop("data",axis = 1)

table_save = table_save.interpolate(method='linear', axis=0)

table_save = pd.concat([table_save, table_save_data],axis=1)
table_save.to_csv("extend.csv") #保存插值扩展后的结果

table_save = table_save.drop("data",axis = 1)


table = table_save.copy()
table_save = table.copy()

# 定义滑动窗口的宽度与预测长度
window_size = 30
# 预测接下来的y_pre_len个时间点长度
next_predict = 16
y_pre_len = 5

for i in range(next_predict+y_pre_len-1):
    table_save["extend_extend_data"+str(i+1)] = np.nan #初始化nan

for fenlei_index in range(0,22):
    fenlei = fenlei_index
    category_data = table[table['fenlei2'] == fenlei]
    start_index = 155
    len_index = 41
    category_t = category_data.columns[start_index:start_index+len_index]
    # datetime_list = [datetime.timestamp(datetime.strptime(ts, '%Y/%m/%d')) for ts in category_t]

    category_ylist = category_data.iloc[:, start_index:start_index+len_index]

    time_series_data = np.array(category_ylist)

        # 准备数据，将时间序列数据分割成输入（X）和输出（y）
    X = []
    y = []

    for i in range(time_series_data.shape[1] - window_size - y_pre_len+1):
        X.append(time_series_data[:, i:i + window_size])
        y.append(time_series_data[:, i + window_size:i + window_size + y_pre_len])

    X = np.array(X).transpose(1, 0, 2).reshape(-1, len_index - window_size- y_pre_len+1, window_size, 1)
    print(X.shape)
    y = np.array(y).transpose(1, 0, 2)

    loss_min = 999999999
    final_choosen = None

    model = load_model("/workspace/model_save1/"+"model_class_"+str(fenlei_index)+"_choosen" +".h5")
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    tscv = TimeSeriesSplit(n_splits=3)
    loss_tmp = 0
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 将输入数据形状调整为适合RNN模型的形状
        X_train = X_train.reshape(-1, window_size, 1)
        X_test = X_test.reshape(-1, window_size, 1)
        y_train = y_train.reshape(-1, y_pre_len, 1)
        y_test = y_test.reshape(-1, y_pre_len, 1)

        X_train, X_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test)
        y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
        # 训练模型并添加动态进度条显示
        epochs = 20
        batch_size = 32  # 训练模型
        for epoch in tqdm(range(epochs), desc="Training"):
            model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, workers=8)
        
        # 在测试集上进行预测
        test_loss = model.evaluate(X_test, y_test)
        train_loss = model.evaluate(X_train, y_train)
        print("Train loss:", train_loss)
        print("Test loss:", test_loss)
        
        loss_tmp = loss_tmp + test_loss + train_loss


    # 预测历史时间步的值
    history_input = X
    history_input = history_input.reshape(-1, window_size, 1)
    history_predictions = []
    history_prediction = model.predict(history_input)
    history_predictions.append(history_prediction)

    history_predictions = np.array(history_predictions).reshape(-1, len_index - window_size - y_pre_len + 1, y_pre_len)[:,
                            :, -1]


    # 预测未来时间步的值
    # future_input = X[:, -1:]  # 使用最后一个窗口的数据作为输入
    future_input = time_series_data[:, -window_size:]  # 选取最后数据作为初始预测未来的输入
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
    history_predictions = np.round(history_predictions)

    for i in range(future_predictions.shape[1]):
        table_save["extend_extend_data"+str(i+1)].loc[category_data.index] = future_predictions[:,i].flatten()

    accuracy_cal =  1-np.sum(np.abs(history_predictions - time_series_data[:,-history_predictions.shape[1]:]),axis = 1)/np.sum(time_series_data,axis =1)

    select_index = 1
    plot_series_data = time_series_data[select_index]
    plot_future_prediction = future_predictions[select_index]
    plot_history_prediction = history_predictions[select_index]
    # 绘制时间序列数据和预测值，使用不同颜色区分
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(plot_series_data)-len(plot_history_prediction)), plot_series_data[:-len(plot_history_prediction)], label='Original Data', color='blue')
    # plt.plot(np.arange(window_size + y_pre_len - 1, len(plot_series_data)), plot_history_prediction,
    #             label='history prediction Data', color='red')
    plt.plot(np.arange(len(plot_series_data) - plot_history_prediction.shape[0] - 1, len(plot_series_data) + next_predict + y_pre_len -1),
                np.insert(np.insert(plot_future_prediction, 0, plot_history_prediction), 0 , plot_series_data[-len(plot_history_prediction)-1]),
                label='Predicted Data', color='green', ls="--")
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("/workspace/save_pic2/"+str(select_index)+str(fenlei_index)+".svg")
    plt.show()

    # table_save.to_csv("/workspace/table_save4/table_save_"+str(fenlei_index)+"_.csv")