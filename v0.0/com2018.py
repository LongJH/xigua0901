print("开始..........")

'''
    1. 导入相关模块
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import time

'''
    2. 导入数据
'''
st = time.time()
df_train = pd.read_csv('../input/train_set.csv')
df_test = pd.read_csv('../input/test_set.csv')
df_train.drop(['article, id'], inplace=True)
df_test.drop(['article, id'], inplace=True)
et = time.time()
import_time = et - st
print("数据导入完成, 耗时 {:.3f}s".format(import_time))
'''
    3. 特征工程
'''
st = time.time()
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class'] - 1
et = time.time()
feature_time = et - st
print("特征处理完成, 耗时 {:.3f}s".format(feature_time))
'''
    4. 构建模型并预测
'''
st = time.time()
lg = LogisticRegression(C=4, dual=True)
lg.fit(x_train, y_train)
y_test = lg.predict(x_test)
model_time = et - st
print("建模预测完成, 耗时 {:.3f}s".format(model_time))
'''
    5. 按要求输出预测文件
'''
st = time.time()
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('result.csv', index=False)
export_time = et - st
print("文件导出完成, 耗时 {:.3f}s".format(export_time))
print("完成..........共耗时 {:.3f}s".format(import_time + feature_time + model_time + export_time))
