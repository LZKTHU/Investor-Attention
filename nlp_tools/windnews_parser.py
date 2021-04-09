#!/usr/bin/python
import pandas as pd
from collections import defaultdict, Counter
from os import path, listdir, mkdir
import jieba
from bs4 import BeautifulSoup
import pickle
from .tokenize import Tokenizer

##@package util.nlp.windnews_parser wind新闻预处理。

##
# @brief wind新闻数据的预处理，提取了code，date，publishdate，content列.
# code列只保留SZ，SH的代码，多个代码以空格分隔，content列做HTML解析和分词
# 后以空格连接，每天的数据为一个csv文件存储
#
# @param news_basedir 新闻的基目录，如/data/news/wind_company_news
# @param output_dir 处理后csv文件的输出目录
#
# @return 无
def parse_raw_files(news_basedir, output_dir):
    if not path.exists(output_dir):
        mkdir(output_dir)
    # get al files
    basedir = news_basedir
    yeardirs = [path.join(basedir, year) for year in listdir(basedir)]
    monthdirs = []
    for year in yeardirs:
        monthdirs += [path.join(year, month) for month in listdir(year)]
    filepaths = []
    for month in monthdirs:
        filepaths += [path.join(month, file) for file in listdir(month)]

    tokenizer = Tokenizer(parallel_workers=4)
    word_counts = Counter()
    cnt_files = 0

    for filepath in filepaths:
        df = pd.read_csv(filepath, sep='\{,\}', encoding='utf-8', engine='python')
        filename = path.split(filepath)[-1]
        date = filename.split('.')[0].split('_')[-1]
        wind_codes = df['WINDCODES:8']
        contents = df['CONTENT:3']
        publish_dates = df['PUBLISHDATE:1']

        no_code_count = 0
        word_data = []

        for i, content in enumerate(contents):
            code_line = wind_codes[i]
            if pd.isnull(code_line):  # some lines has no code
                no_code_count += 1
                continue
            # split code field and filter invalid codes
            codes = [code for ele in code_line.split('|')
                     for code in ele.split(':') if '.' in code]
            codes = [code for code in codes if code.split('.')[-1] in ['SZ', 'SH']]
            if len(codes) == 0:
                continue
            # print(content)
            if pd.isnull(content):
                continue

            # parse html text
            soup = BeautifulSoup(content, 'lxml')
            text = soup.get_text().strip().split()
            text = ' '.join(text)
            # cut words
            word_list = tokenizer.tokenize(text, rm_stopwords=True, filter_num=True)
            if len(word_list) == 0:
                continue
            word_counts.update(word_list)
            word_data.append([date, publish_dates[i], ' '.join(codes), ' '.join(word_list)])

        if len(word_data) == 0: # have no valid data this day
            continue
        # save new file
        cols = ['date', 'publishdate', 'code', 'wordlist']
        word_df = pd.DataFrame(word_data, columns=cols)
        save_file = path.join(output_dir, 'word_list_' + date + '.csv')
        word_df.to_csv(save_file, index=False, columns=cols)
        cnt_files += 1
        if cnt_files % 50 == 0:
            print('processed files', cnt_files)

    print('total different words:', len(word_counts))
    print('most common 500 words:', word_counts.most_common(500))


def main():
    parse_raw_files('/data/news/stock_positive_news', '/home/lingchunyang/data/positive_news')

if __name__ == '__main__':
    main()
