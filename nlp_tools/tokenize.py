#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import jieba
import zhconv

## @package util.nlp.tokenize 中文分词工具类。


##
# @brief 中文分词工具类
#
# @details 加载自定义的词典和停用词表，使用jieba分词器进行分词
class Tokenizer():
    
    ## 本地路径。数据都在该路径下。
    __LOCAL_PATH = os.path.realpath(__file__ + "/../")

    ##
    # @brief 初始化函数
    #
    # @param user_dict_path 词典路径，默认为./data/user_dict.txt
    # @param stopwords_path 停用词路径，默认为./data/stopwords.txt
    # @param parallel_workers 使用jiaba分词的并行度，默认为1
    #
    # @return 无
    def __init__(self,
                 user_dict_path=__LOCAL_PATH + '/data/user_dict.txt',
                 stopwords_path=__LOCAL_PATH + '/data/stopwords.txt',
                 parallel_workers=1):
        self.stopwords = self.__load_stopwords(stopwords_path)
        jieba.load_userdict(user_dict_path)
        self.workers = parallel_workers
        if self.workers > 1:
            jieba.enable_parallel(self.workers)

    def __del__(self):
        if self.workers > 1:
            jieba.disable_parallel()

    ##
    # @brief 私有方法，加载停用词表
    #
    # @param path 停用词表路径
    # @return 停用词表的集合
    def __load_stopwords(self, path):
        with  open(path, encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines() if line.strip()]
        print('loaded stopword ', len(stopwords))
        return set(stopwords)

    ##
    # @brief 将文本分词，返回分词后的列表，英文字母都转为小写，繁体字都转为简体
    #
    # @param text 待分词的字符串文本
    # @param rm_stopwords 是否去除停用词，默认为False
    # @param filter_num 是否过滤掉数字（整数，小数，百分数），默认为False
    #
    # @return 分词后的列表
    def tokenize(self, text, rm_stopwords=False, filter_num=False):
        cut_result = jieba.cut(text)

        # lower English words and simplify Chinese words
        cut_result = [word.strip().lower() for word in cut_result if word.strip()]
        cut_result = [zhconv.convert(word, 'zh-cn') for word in cut_result]
        # filter numbers, percentage and stopwords
        if filter_num:
            cut_result = self.filter_numbers(cut_result)
        if rm_stopwords:
            cut_result = self.remove_stopwords(cut_result)
        return cut_result

    ##
    # @brief 过滤掉数字
    #
    # @param word_list 分好词的列表
    #
    # @return 去掉数字后的列表
    def filter_numbers(self, word_list):
        result = [word for word in word_list if not word.isnumeric()]
        result = [word for word in result if '.' not in word and '%' not in word]
        return result

    ##
    # @brief 过滤掉停用词
    #
    # @param word_list 分好词的列表
    #
    # @return 去掉停用词后的列表
    def remove_stopwords(self, word_list):
        word_list = [word for word in word_list if word not in self.stopwords]
        return word_list


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        usage="分词",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=50))

    parser.add_argument("-t", "--text", help="要切分的文本。")
    parser.add_argument("-f", "--file", help="从文件读入要切分的文本。")

    options = parser.parse_args()

    if options.file is not None:
        options.text = "\n".join(open(options.file, "w").readlines())

    return options


def _main():
    options = _parse_args()

    tokenize = Tokenizer()
    print(tokenize.tokenize(options.text))


if __name__ == "__main__":
    _main()
