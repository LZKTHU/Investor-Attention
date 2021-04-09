#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import pickle as pk
from os import path, listdir, mkdir
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import time
import math
from autobase import dl
from util import tradingday

##@package util.nlp.vectorize 文本向量化。

##
# @brief 过滤掉涉及多支股票的新闻，'A'开头代码的，以及内容为空的
#
# @param df DataFrame对象，必须包含'code'和'wordlist'列
#
# @return 只和单只股票相关的dataframe
def get_single_stock_news(df):
    positions = [i for i, code in enumerate(df['code']) if len(code.split()) == 1
                 and 'A' not in code and not pd.isnull(df.iloc[i].wordlist)]
    return df.iloc[positions]


##
# @brief 以当天收益率作为文档的label
#
# @param df DataFrame对象，必须包含'date'和'code'列
#
# @return 当天收益率的list
def get_close_label(df):
    close = dl.get_data('close')
    close_label = []
    for index, row in df.iterrows():
        di = dl.get_di(row['date'])
        ii = dl.get_ii(row['code'])
        close_1 = close[di-1][ii]
        r = (close[di][ii] - close_1) / close_1 * 100
        if math.isinf(r) or math.isnan(r):
            r = 0
        r = round(r, 4)  # 保留4位小数
        close_label.append(r)
    return close_label


##
# @brief 以当天收益率的正负作为文档的binary label
#
# @param df DataFrame对象，必须包含'date'和'code'列
#
# @return 当天收益正负的list
def get_close_binary_label(df):
    binary_label = [r > 0 for r in get_close_label(df)]
    return binary_label


##
# @brief 为分词后的文档集构建词汇表，使用sklearn的TfidfVectorizer
#
# @param docs 字符串的一维列表，其中每个字符串是由空格分隔的分好词的文档
# @param filename 词汇表的存储路径
# @param vocab_size 词汇表的大小, 默认为3000
# @param min_df 最小文档频率，词汇表中的词必须在超过该数量的文档中出现过，默认为10
#
# @return
def build_vocabulary(docs, filename,
                     vocab_size=30000, min_df=10):
    print('start building vocabulary...')
    vectorizer = TfidfVectorizer(analyzer=str.split, min_df=min_df,
                                 max_df=0.9, max_features=vocab_size)
    X = vectorizer.fit_transform(docs)
    print('build vocabulary size:', len(vectorizer.vocabulary_))
    with  open(filename, 'wb') as f:
        pk.dump(vectorizer.vocabulary_, f)


##
# @brief 根据词汇表将文档tf-idf向量化
#
# @param docs 字符串的一维列表，其中每个字符串是由空格分隔的分好词的文档
# @param vocab_path 词汇表的路径
#
# @return 向量化后的文档
def tfidf(docs, vocab_path):
    if path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pk.load(f)
        print('vocabulary already exist: ',len(vocab))
        vectorizer = TfidfVectorizer(analyzer=str.split, vocabulary=vocab)
        X = vectorizer.fit_transform(docs)
    else:
        raise ValueError("vocabulary path not exist! Please build_vocabulary first. ")
    return X


##
# @brief 从目录读取数据集并过滤得到只和单只股票相关的新闻
#
# @param basedir 数据基目录，该目录下都是预处理后的csv文件
# @param only_single_stock 是否只保留和单只股票相关的新闻，默认为True
#
# @return DataFrame形式的数据集
def get_dataset(basedir, only_single_stock=True):
    files = [path.join(basedir, f) for f in listdir(basedir)]
    dataset = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f)
        # filter news that is not in a trading day
        if not tradingday.is_tradingday(str(df['date'][0])):
            continue
        if only_sinle_stock:
            df = get_single_stock_news(df)
        dataset = pd.concat([dataset, df], ignore_index=True)
    print('dataset size: ', len(dataset))
    return dataset


##
# @brief 保存模型预测后的结果到csv文件
#
# @param dataset DataFrame至少包含三列，code, date, y_hat（即预测值）
# @param modelname 模型名称，默认存储在./result/res_modelname.csv
#
# @return 无
def save_result(dataset, modelname):
    # 对于同一日期的同一股票，多条新闻进行平均
    df = dataset.groupby(['date', 'code'])['y_hat'].mean()
    # reset index to convert series to dataframe
    df = df.reset_index()
    if not path.exists('./result'):
        os.mkdir('./result')
    df.to_csv('./result/res_' + modelname + '.csv', index=False, columns=['date', 'code','y_hat'])

