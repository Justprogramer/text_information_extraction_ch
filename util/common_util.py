# -*-coding:utf-8-*-
import platform
import re
import os
import pickle
import numpy as np


# 将中文括号替换成英文括号
def format_brackets(str):
    if str is None or is_nan(str):
        return ""
    replace = str.replace('（', '(').replace('）', ')')
    return clean_text(replace)


# 判断文本是否是发言起始段落
def check_paragraph(str):
    pattern_str = r"^([原|被|代|审|书|时|地|案|公][\S]{0,4}|^[\S]{0,10})[：|:]"
    return re.search(pattern_str, str)


# 检测是否是windows平台
def is_windows():
    return platform.system() == "Windows"


def split_paragraph(str):
    return re.split(r'[\n|\r]{0,2}', str)


# 替换所有的空白分隔符
def clean_text(str):
    if str is None or is_nan(str):
        return ""
    return re.sub("[\r\n\\s]+", "", str)


# 判断文件是否存在
def is_file_exist(path):
    return os.path.exists(path)


# 判断是否是NaN值
def is_nan(num):
    return num != num


def tokens2id_array(items, voc, oov_id=1):
    """
    将词序列映射为id序列
    Args:
        items: list, 词序列
        voc: item -> id的映射表
        oov_id: int, 未登录词的编号, default is 1
    Returns:
        arr: np.array, shape=[max_len,]
    """
    arr = np.zeros((len(items),), dtype='int32')
    for i, item in enumerate(items):
        if item in voc:
            arr[i] = voc[item]
        else:
            arr[i] = oov_id
    return arr


# 抽取主要的举证质证段落
def check_evidence_paragraph(document):
    evidence_start_pattern = r"(?:被告|原告){0,1}.*(?:提供|举示|出示){0,1}.*(?:质证|证据|举证)"
    anti_evidence_start_patter = r"审.*[《|》]?.*(?:规定|责任|义务)"
    evidence_end_pattern = r"(?:质证|证据|举证|调查).*(?:结束|完毕|不再进行)"
    start, end = 0, None
    for index, paragraph in enumerate(document):
        paragraph = "。".join(paragraph)
        if start == 0 \
                and re.search(evidence_start_pattern, paragraph) \
                and not re.search(anti_evidence_start_patter, paragraph):
            start = index
            continue
        if re.search(evidence_end_pattern, paragraph):
            end = index
            if end <= start:
                print("提取证据段落出错，起始位置：%s,结束位置：%s" % (start, end))
    if end is None:
        end = len(document) - 1
    return start, end


# 替换数字
def normalize_word(word):
    new_word = ''
    for c in word:
        if c.isdigit():
            new_word += '0'
        else:
            new_word += c
    return new_word


def read_bin(path):
    """读取二进制文件
    Args:
        path: str, 二进制文件路径
    Returns:
        pkl_ob: pkl对象
    """
    file = open(path, 'rb')
    return pickle.load(file)


def dump_pkl_data(ob, path):
    """将python对象写入pkl文件
        Args:
            path: str, pkl文件路径
            ob: python的list, dict, ...
        """
    with open(path, 'wb') as file_pkl:
        pickle.dump(ob, file_pkl)


if __name__ == '__main__':
    str = "ab\nc\r  \td    "
    print(clean_text(str))
