#导包
import pandas as pd
from datetime import datetime
from textblob import TextBlob

#读取文件
musical_instruments_reviews_path = open(r'Musical_instruments_reviews.csv')

mir_data = pd.read_csv(musical_instruments_reviews_path)

print(mir_data.head())

#查看缺失值
mir_data.info()

#日期类型为object，一致化处理，将日期类型转换为date类型
mir_data['reviewTime'] = pd.to_datetime(mir_data['reviewTime'],
                                        format = '%m %d, %Y')
mir_data.info()

#将没有用户姓名的数据删除
mir_data.drop(mir_data[mir_data['reviewerName'].isna()].index,inplace=True)
mir_data.info()

#查看缺失值，发现没有具体评论但有评论摘要和用户姓名，这部分数据保存
mir_data[mir_data['reviewText'].isna()].info()

#将评论单独存取
summary_data = mir_data['summary']

#对每条评论进行分类
for i in range(0,len(summary_data)):
    text = summary_data.iloc[i]

    blob = TextBlob(text)
    print(text,blob.sentiment)
