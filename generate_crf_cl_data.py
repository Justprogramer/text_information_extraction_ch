# -*-coding:utf-8-*-
import codecs
import csv

import os
import pandas as pd

import util.common_util as my_util
from collections import Counter

# 证据名称列表
evidence_list = list()
# 笔录正文字典 文件名:内容
content_dict = dict()
# 笔录中举证质证文本 文件名：内容
train_evidence_paragraph_dict = dict()
dev_evidence_paragraph_dict = dict()
test_evidence_paragraph_dict = dict()
# 笔录中存在的证据对应关系 文件名:[举证方 Evidence(E) ,证据名称 Trigger(T) ,证实内容 Content(C), 质证意见 View(V),质证方 Anti-Evidence(A)]
evidence_dict = dict()
opinion_dict = dict()

other_count, evidence_count, view_count = 0, 0, 0

symbol_list = [",", "，", ".", "。", "?", "？", ":", "：", "（", "）", "(", ")", "*", "、", ";", "；", "!", "！"]


def analyse_cl_data():
    global other_count, evidence_count, view_count
    analyse_data_excel_content()

    length = len(content_dict.values())
    train_content_keys = sorted(content_dict)[:int(length * 0.8)]
    dev_content_keys = sorted(content_dict)[int(length * 0.8):int(length * 0.9)]
    test_content_keys = sorted(content_dict)[int(length * 0.9):]
    train_content, dev_content, test_content = {}, {}, {}
    for key in train_content_keys:
        train_content[key] = content_dict[key]
    for key in dev_content_keys:
        dev_content[key] = content_dict[key]
    for key in test_content_keys:
        test_content[key] = content_dict[key]

    analyse_data_excel_tags()
    extract_evidence_paragraph(train_content, "train")
    extract_evidence_paragraph(dev_content, "dev")
    extract_evidence_paragraph(test_content, "test")

    # analyse_dir_document()
    create_cl_data(train_evidence_paragraph_dict, "train")
    create_cl_data(dev_evidence_paragraph_dict, "train")
    print("\ntrain other_count:%s,evidence_count:%s,view_count:%s," % (other_count, evidence_count, view_count))
    other_count, evidence_count, view_count = 0, 0, 0
    create_cl_data(test_evidence_paragraph_dict, "test")
    print("\ntest other_count:%s,evidence_count:%s,view_count:%s," % (other_count, evidence_count, view_count))


# 从excel中加载数据
def analyse_data_excel_content(title=None, content=None):
    if title is None and content is None:
        rows = pd.read_excel("./raw_data/文书内容.xls", sheet_name=0, header=0)
        for title, content in rows.values:
            title = my_util.format_brackets(title.strip())
            # print(title)
            analyse_data_excel_content(title, content)
    else:
        old_paragraphs = [paragraph for paragraph in my_util.split_paragraph(content)
                          if paragraph is not None and len(paragraph.strip()) > 0]
        new_paragraphs = list()
        new_paragraph = ""
        # 合并发言人段落
        for index, paragraph in enumerate(old_paragraphs):
            # print("%s:%s" % (title, index))
            # 最后一个是中文的话，加上句号
            if '\u4e00' <= paragraph[-1] <= '\u9fff' and paragraph[-1] not in symbol_list:
                paragraph += "。"
            if my_util.check_paragraph(paragraph):
                if new_paragraph is not None and len(new_paragraph) > 0:
                    new_paragraphs.append(new_paragraph)
                new_paragraph = paragraph
            else:
                new_paragraph = new_paragraph + paragraph
        content_dict[title] = [
            [sentence for sentence in paragraph.split("。")
             if sentence is not None and len(sentence.strip()) > 0]
            for paragraph in new_paragraphs]
    return content_dict[title]


# 举证方 Evidence(E) 证据名称 Trigger(T) 证实内容 Content(C) 质证意见 Opinion(O) 质证方 Anti-Evidence(A)
def analyse_data_excel_tags():
    rows = pd.read_excel("./raw_data/证据关系对应.xls", sheet_name=0, header=0)
    for title, E, T, C, O, A in rows.values:
        title = my_util.clean_text(title)
        title = my_util.format_brackets(title)
        E = my_util.clean_text(E)
        T = my_util.clean_text(T)
        C = my_util.clean_text(C)
        O = my_util.clean_text(O)
        A = my_util.clean_text(A)
        if title not in evidence_dict:
            evidence_dict[title] = list()
        if len(T) != 0:
            tag_index = str(T).find("]")
            if tag_index != -1:
                T = T[tag_index + 1:]
            evidence_dict[title].append([E, T, C])
        if title not in opinion_dict:
            opinion_dict[title] = list()
        if len(O) != 0 or len(A) != 0:
            opinion_dict[title].append([A, O])


# 抽取主要举证质证段落
def extract_evidence_paragraph(content, type=None):
    for d in content:
        if d not in evidence_dict:
            continue
        start, end = my_util.check_evidence_paragraph(content[d])
        if type == "train":
            train_evidence_paragraph_dict[d] = content[d][start:end]
        elif type == "dev":
            dev_evidence_paragraph_dict[d] = content[d][start:end]
        else:
            test_evidence_paragraph_dict[d] = content[d][start:end]


def create_cl_data(evidence_paragraph_dict, type=None):
    global other_count, evidence_count, view_count
    for d in evidence_paragraph_dict:
        text = []
        if d not in evidence_dict or len(evidence_dict[d]) == 0:
            # print("文档《%s》没有对应的数据标签\n" % d)
            continue
        evidence_content = evidence_paragraph_dict[d]
        for paragraph in evidence_content:
            paragraph = "。".join(paragraph)
            paragraph = my_util.clean_text(paragraph)
            if paragraph[-1] not in symbol_list:
                paragraph += "。"
            paragraph_type = "O"
            if len(paragraph) <= 4:
                continue
            for [A, O] in opinion_dict[d]:
                find_o, find_a = -1, -1
                if len(A) > 0:
                    find_a = str(paragraph).find(A)
                if len(O) > 1:
                    find_o = str(paragraph).find(O)
                if find_a == 0 and find_o > -1:
                    paragraph_type = "V"
                    break
            if paragraph_type == "O":
                for [E, T, C] in evidence_dict[d]:
                    find_e, find_t, find_c = -1, -1, -1
                    if len(E) > 0:
                        find_e = str(paragraph).find(E)
                    if len(T) > 1:
                        find_t = str(paragraph).find(T)
                    if len(C) > 1:
                        find_c = str(paragraph).find(C)
                    if find_e == 0 and find_t > -1 or find_c > -1:
                        paragraph_type = "E"
                        break
            if paragraph_type == "E":
                text.append(["E", paragraph])
                evidence_count += 1
            elif paragraph_type == "V":
                text.append(["V", paragraph])
                view_count += 1
            else:
                if type == "train" and other_count % 10 == 0:
                    text.append(["O", paragraph])
                else:
                    text.append(["O", paragraph])
                other_count += 1
        with codecs.open('./data/crf_cl_%s.tsv' % type, "a", "utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for line in text:
                tsv_writer.writerow([line[0], line[1]])
            tsv_writer.writerow([])


if __name__ == '__main__':
    analyse_cl_data()
