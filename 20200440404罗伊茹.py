#导包
import numpy as np              
import pandas as pd             
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from pandas import Series , DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from snownlp import SnowNLP

#观察数据前5行
review_data = pd.read_csv(r'C:\Users\luoyiru\OneDrive\文档\Musical_instruments_reviews.csv')
review_data.head()

#查看缺失值
review_data.info()

#format要与时间格式一致
review_data['reviewTime']=pd.to_datetime(review_data['reviewTime'],format='%m %d, %Y')

#查看缺失的数据情况,发现27条数据没有用户姓名信息，但是有评论信息,考虑到真实性，这部分评论数据需要去掉，
review_data.drop(review_data[review_data['reviewerName'].isna()].index,inplace=True)

#使用isna定位到reviewerName列缺失的索引,对源数据使用drop删除，删除后剩余10234条数据
review_data.info()

#查看缺失的数据情况,发现7条数据没有具体评论内容，但是有评论摘要以及用户姓名，这部分数据给予保留
review_data[review_data['reviewText'].isna()].info()

#根据评分，使用lambda匿名函数，把评分>3的，取值1，当作正向情感，评分<3的，取值0，当作负向情感
def make_label(df):
    df["sentiment"] = df["overall"].apply(lambda x: 1 if x > 3 else 0)

make_label(review_data)
#查看数据集形状
print(review_data)

#特征、标签分开赋值
X = review_data['summary']
y = review_data.sentiment

#将数据拆分成训练数据集、测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#查看数据集形状
print(X_train.shape)

#默认参数向量化
vect = CountVectorizer()
term_matrix = DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
#查看向量化后的数据集形状
print(term_matrix.shape)

#过滤一部分特征向量
#在超过这一比例的文档中出现的关键词（过于平凡），去除掉
max_df = 0.8
#在低于这一数量的文档中出现的关键词（过于独特），去除掉
min_df = 3

vect = CountVectorizer(max_df = max_df,
                       min_df = min_df,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')

term_matrix = DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
#查看向量化后的数据集形状
print(term_matrix.shape)

#导入 朴素贝叶斯函数，建立分类模型
nb = MultinomialNB()
#将顺序工作连接起来
pipe = make_pipeline(vect, nb)
#将未特征向量化的数据输入，验证模型的准确率
print(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())

#用训练集拟合数据
pipe.fit(X_train, y_train)
#测试集预测结果
y_pred = pipe.predict(X_test)
#查看评分
print(metrics.accuracy_score(y_test, y_pred))

#用 snowNLP 来做对比
def get_sentiment(text):
    return SnowNLP(text).sentiments
#用测试集跑一遍
y_pred_snow = X_test.apply(get_sentiment)
#将数据转换一下，大于0.5为1，小于0.5为0
y_pred_snow_norm = y_pred_snow.apply(lambda x: 1 if x>0.5 else 0)
#查看评分
print(metrics.accuracy_score(y_test, y_pred_snow_norm))
