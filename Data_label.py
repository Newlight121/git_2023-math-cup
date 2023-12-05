from matplotlib import pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#test
#dev_test
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings(action='ignore')

table1 = pd.read_csv("数据处理.csv", encoding="ANSI", index_col=0)

encoding_name = ["product_ifo_0", "product_ifo_1", "product_ifo_2"]
category_name = ["encoded_category_0", "encoded_category_1", "encoded_category_2"]

label_encoder_0 = LabelEncoder()
index_class_0 = table1["product_ifo_0"]

for i in range(len(encoding_name)):
    table1[category_name[i]] = label_encoder_0.fit_transform(table1[encoding_name[i]])  # 编码

table_tmp = table1.copy()
unique_categories = table_tmp['encoded_category_0'].unique()  # 获取类别标号
for category in unique_categories:
    category_rows = table_tmp[table_tmp['encoded_category_0'] == category]
    category_rows['encoded_category_1'] = 'Updated'
    label_encoder_1 = LabelEncoder()
    category_rows['encoded_category_1'] = label_encoder_1.fit_transform(category_rows[encoding_name[1]])  # 1级编码
    table_tmp.update(category_rows)

unique_categories = table_tmp['encoded_category_0'].unique()  # 获取类别标号
for category in unique_categories:
    category_rows = table_tmp[table_tmp['encoded_category_0'] == category]
    unique_categories1 = category_rows['encoded_category_1'].unique()  # 获取类别标号
    category_rows['encoded_category_2'] = 'Updated'  # 更新设定
    for category1 in unique_categories1:
        category_rows1 = category_rows[category_rows['encoded_category_1'] == category1]
        category_rows1['encoded_category_2'] = 'Updated'
        label_encoder_1 = LabelEncoder()
        category_rows1['encoded_category_2'] = label_encoder_1.fit_transform(category_rows1[encoding_name[2]])  # 二级编码
        category_rows.update(category_rows1)
        table_tmp.update(category_rows)

table_tmp['encoded_category_cal'] = table_tmp['encoded_category_0'] * 100 + table_tmp['encoded_category_1'] * 10 + \
                                    table_tmp['encoded_category_2']

single_endconding_name = ["seller_ifo_0", "seller_ifo_1", "seller_ifo_2", "house_ifo_0", "house_ifo_1"]

for name in single_endconding_name:
    label_encoder_tmp = LabelEncoder()
    table_tmp[name] = label_encoder_tmp.fit_transform(table_tmp[name])  # 编码

for i in range(len(encoding_name)):
    table_tmp[encoding_name[i]] = table_tmp[category_name[i]]

table_tmp.to_csv("数据处理编码.csv", encoding="ANSI")



