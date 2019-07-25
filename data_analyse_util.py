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
tag_dic = dict()

other_count, evidence_count, view_count = 0, 0, 0


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


def analyse_ner_data():
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
    create_ner_data(train_evidence_paragraph_dict, "ner_train")
    create_ner_data(dev_evidence_paragraph_dict, "ner_dev")
    create_ner_data(test_evidence_paragraph_dict, "ner_test")


def analyse_joint_data():
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
    create_joint_data(train_evidence_paragraph_dict, "train")
    create_joint_data(dev_evidence_paragraph_dict, "train")
    print("\ntrain other_count:%s,evidence_count:%s,view_count:%s," % (other_count, evidence_count, view_count))
    other_count, evidence_count, view_count = 0, 0, 0
    create_joint_data(test_evidence_paragraph_dict, "test")
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
            if my_util.check_paragraph(paragraph):
                if new_paragraph is not None and len(new_paragraph) > 0:
                    if '\u4e00' <= paragraph[-1] <= '\u9fff':
                        paragraph += "。"
                    new_paragraphs.append(new_paragraph)
                new_paragraph = paragraph
            else:
                if '\u4e00' <= paragraph[-1] <= '\u9fff':
                    paragraph += "。"
                new_paragraph = new_paragraph + paragraph
        content_dict[title] = [
            [my_util.clean_text(sentence) for sentence in paragraph.split("。")
             if sentence is not None and len(sentence.strip()) > 0]
            for paragraph in new_paragraphs]
    return content_dict[title]


def extract_single_sentence_from_paragraph(paragraph):
    sentences = []
    if my_util.is_nan(paragraph):
        return sentences
    for sentence in paragraph.split("。"):
        if sentence is not None and len(sentence.strip()) > 0:
            for sen in sentence.split("；"):
                if sen is not None and len(sen.strip()) > 0:
                    sentences.append(my_util.clean_text(sen))
    return sentences


# 举证方 Evidence(E) 证据名称 Trigger(T) 证实内容 Content(C) 质证意见 Opinion(O) 质证方 Anti-Evidence(A)
def analyse_data_excel_tags():
    rows = pd.read_excel("./raw_data/证据关系对应.xls", sheet_name=0, header=0)
    for title, E, T, C, V, A in rows.values:
        title = my_util.clean_text(title)
        E = my_util.clean_text(E)
        A = my_util.clean_text(A)
        title = my_util.format_brackets(title)
        # print("tag_title:%s" % title)
        T = extract_single_sentence_from_paragraph(T)
        C = extract_single_sentence_from_paragraph(C)
        V = extract_single_sentence_from_paragraph(V)
        if title not in tag_dic:
            tag_list = list()
            for t in T:
                tag_list.append([E, t, C, V, A])
            tag_dic[title] = tag_list
        else:
            for t in T:
                tag_dic[title].append([E, t, C, V, A])
                if t not in evidence_list:
                    evidence_list.append(t)


# 抽取主要举证质证段落
def extract_evidence_paragraph(content, type=None):
    for d in content:
        if d not in tag_dic:
            continue
        start, end = my_util.check_evidence_paragraph(content[d])
        # print(
        #     "提取证据段落完成《%s》(%s)，起始位置：%s,结束位置：%s\n%s\n%s" % (
        #         d, len(content_dict[d]), start, end, content_dict[d][start],
        #         content_dict[d][end - 1]))
        if type == "train":
            train_evidence_paragraph_dict[d] = content[d][start:end]
        elif type == "dev":
            dev_evidence_paragraph_dict[d] = content[d][start:end]
        else:
            test_evidence_paragraph_dict[d] = content[d][start:end]


def create_cl_data(evidence_paragraph_dict, type=None):
    global other_count, evidence_count, view_count
    text = []
    for d in evidence_paragraph_dict:
        if d not in tag_dic:
            # print("文档《%s》没有对应的数据标签\n" % d)
            continue
        evidence_content = evidence_paragraph_dict[d]
        last_paragraph = None
        for paragrah in evidence_content:
            paragrah = "。".join(paragrah)
            tag = ["O"] * len(paragrah)
            if len(paragrah) <= 0:
                continue
            for [_, t, C, V, _] in tag_dic[d]:
                find_t = str(paragrah).find(t)
                while find_t != -1 and tag[find_t] == "O":
                    tag = tag[:find_t] + ["E"] * len(t) + tag[find_t + len(t):]
                    find_t = str(paragrah).find(t, find_t)
                for c in C:
                    if len(c) <= 1:
                        continue
                    find_c = str(paragrah).find(c)
                    while find_c != -1 and tag[find_c] == "O":
                        tag = tag[:find_c] + ["E"] * len(c) + tag[find_c + len(c):]
                        find_c = str(paragrah).find(c, find_c)
                for v in V:
                    if len(v) <= 1:
                        continue
                    find_v = str(paragrah).find(v)
                    while find_v != -1 and tag[find_v] == "O":
                        tag = tag[:find_v] + ["V"] * len(v) + tag[find_v + len(v):]
                        find_v = str(paragrah).find(v, find_v)
            context = paragrah + "\t" + ("" if last_paragraph is None else last_paragraph)
            last_paragraph = paragrah
            counter = Counter(tag)
            if counter["E"] > counter["V"] and counter["O"] - counter["E"] < 5:
                text.append(["E", context])
                evidence_count += 1
            elif counter["E"] < counter["V"] and counter["O"] - counter["V"] < 5:
                text.append(["V", context])
                view_count += 1
            else:
                if type == "train" and other_count % 10 == 0:
                    text.append(["O", context])
                else:
                    text.append(["O", context])
                other_count += 1
    with codecs.open('./data/%s.tsv' % type, "a", "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in text:
            if len(line[1]) <= 0:
                continue
            tsv_writer.writerow([line[0], line[1].split("\t")[0], line[1].split("\t")[1]])


def create_ner_data(evidence_paragraph_dict, type=None):
    evidence_count = 0
    opinion_count = 0
    for d in evidence_paragraph_dict:
        if d not in tag_dic:
            continue
        evidence_content = evidence_paragraph_dict[d]
        for paragrah in evidence_content:
            paragrah = "。".join(paragrah)
            tag = ["O"] * len(paragrah)
            evidence_paragraph, opinion_paragraph = False, False
            for [E, t, C, O, A] in tag_dic[d]:
                has_t, has_c, has_o = False, False, False
                find_t = str(paragrah).find(t)
                while find_t != -1 and tag[find_t] == "O":
                    has_t = True
                    tag = tag[:find_t] + ["B-T"] + ["I-T"] * (len(t) - 1) + tag[find_t + len(t):]
                    find_t = str(paragrah).find(t)
                for c in C:
                    if len(c) <= 1:
                        continue
                    find_c = str(paragrah).find(c)
                    while find_c != -1 and tag[find_c] == "O":
                        has_c = True
                        tag = tag[:find_c] + ["B-C"] + ["I-C"] * (len(c) - 1) + tag[find_c + len(c):]
                        find_c = str(paragrah).find(c)
                for o in O:
                    if len(o) <= 1:
                        continue
                    find_o = str(paragrah).find(o)
                    while find_o != -1 and tag[find_o] == "O":
                        has_o = True
                        tag = tag[:find_o] + ["B-O"] + ["I-O"] * (len(o) - 1) + tag[find_o + len(o):]
                        find_o = str(paragrah).find(o)
                if len(A.strip()) > 1:
                    find_a = str(paragrah).find(A + "：")
                    if find_a != -1 and has_o and tag[find_a] == "O":
                        tag = tag[:find_a] + ["B-A"] + ["I-A"] * (len(A) - 1) + tag[find_a + len(A):]
                        opinion_paragraph = True
                if len(E.strip()) > 1:
                    find_e = str(paragrah).find(E + "：")
                    if find_e != -1 and (has_t or has_c) and tag[find_e] == "O":
                        tag = tag[:find_e] + ["B-E"] + ["I-E"] * (len(E) - 1) + tag[find_e + len(E):]
                        evidence_paragraph = True
            assert len(paragrah) == len(tag)
            if opinion_paragraph:
                for i, label in enumerate(tag):
                    if label not in ["O", "B-A", "I-A", "B-O", "I-O"]:
                        tag[i] = tag[i - 1]
                opinion_count += 1
                with codecs.open('./data/%s_opinion.txt' % type, "a", "utf-8") as f:
                    for i, word in enumerate(paragrah):
                        f.write("%s %s\n" % (word, tag[i]))
                    f.write("\n")
            if evidence_paragraph:
                for i, label in enumerate(tag):
                    if label not in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"]:
                        tag[i] = tag[i - 1]
                evidence_count += 1
                with codecs.open('./data/%s_evidence.txt' % type, "a", "utf-8") as f:
                    for i, word in enumerate(paragrah):
                        f.write("%s %s\n" % (word, tag[i]))
                    f.write("\n")
    print("\ncount:evidence[%d],opinion[%d]" % (evidence_count, opinion_count))


def create_joint_data(evidence_paragraph_dict, type=None):
    global other_count, evidence_count, view_count
    text = []
    for d in evidence_paragraph_dict:
        if d not in tag_dic:
            continue
        evidence_content = evidence_paragraph_dict[d]
        for paragrah in evidence_content:
            paragrah = "。".join(paragrah)
            tag = ["O"] * len(paragrah)
            if len(paragrah) <= 0:
                continue
            for [E, t, C, V, A] in tag_dic[d]:
                has_t, has_c, has_v = False, False, False
                find_t = str(paragrah).find(t)
                while find_t != -1 and tag[find_t] == "O":
                    has_t = True
                    tag = tag[:find_t] + ["B-T"] + ["I-T"] * (len(t) - 1) + tag[find_t + len(t):]
                    find_t = str(paragrah).find(t, find_t)
                for c in C:
                    if len(c) <= 1:
                        continue
                    find_c = str(paragrah).find(c)
                    while find_c != -1 and tag[find_c] == "O":
                        has_c = True
                        tag = tag[:find_c] + ["B-C"] + ["I-C"] * (len(c) - 1) + tag[find_c + len(c):]
                        find_c = str(paragrah).find(c, find_c)
                for v in V:
                    if len(v) <= 1:
                        continue
                    find_v = str(paragrah).find(v)
                    while find_v != -1 and tag[find_v] == "O":
                        has_v = True
                        tag = tag[:find_v] + ["B-O"] + ["I-O"] * (len(v) - 1) + tag[find_v + len(v):]
                        find_v = str(paragrah).find(v, find_v)
                if len(A.strip()) > 1:
                    find_a = str(paragrah).find(A + "：")
                    if find_a != -1 and has_v and tag[find_a] == "O":
                        tag = tag[:find_a] + ["B-A"] + ["I-A"] * (len(A) - 1) + tag[find_a + len(A):]
                if len(E.strip()) > 1:
                    find_e = str(paragrah).find(E + "：")
                    if find_e != -1 and (has_t or has_c) and tag[find_e] == "O":
                        tag = tag[:find_e] + ["B-E"] + ["I-E"] * (len(E) - 1) + tag[find_e + len(E):]
            assert len(paragrah) == len(tag)
            counter = Counter(tag)
            counter_e = counter["B-E"] + counter["I-E"] + counter["B-C"] + counter["I-C"] + counter["B-T"] + counter[
                "I-T"]
            counter_a = counter["B-A"] + counter["I-A"] + counter["B-O"] + counter["I-O"]
            if counter_e > counter_a and counter["O"] - counter_e < 5:
                for i, label in enumerate(tag):
                    if label not in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"]:
                        tag[i] = tag[i - 1] if tag[i - 1] in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"] else "O"
                text.append(["E", paragrah, tag, ["O"] * len(paragrah)])
                evidence_count += 1
            elif counter_e < counter_a and counter["O"] - counter_a < 5:
                for i, label in enumerate(tag):
                    if label not in ["O", "B-A", "I-A", "B-O", "I-O"]:
                        tag[i] = tag[i - 1] if tag[i - 1] in ["O", "B-A", "I-A", "B-O", "I-O"] else "O"
                text.append(["V", paragrah, ["O"] * len(paragrah), tag])
                view_count += 1
            else:
                if type == "train" and other_count % 10 == 0:
                    text.append(["O", paragrah, ["O"] * len(paragrah), ["O"] * len(paragrah)])
                else:
                    text.append(["O", paragrah, ["O"] * len(paragrah), ["O"] * len(paragrah)])
                other_count += 1
    with codecs.open('./joint_data/%s.tsv' % type, "a", "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in text:
            if len(line[1]) <= 0:
                continue
            tsv_writer.writerow([line[0], line[1], " ".join(line[2]), " ".join(line[3])])


def generate_data():
    test_all_data_path = '.\\test_all_data'
    if not os.path.exists(test_all_data_path):
        os.makedirs(test_all_data_path)
    analyse_data_excel_content()
    length = len(content_dict.values())
    test_content_keys = sorted(content_dict)[int(length * 0.9):]
    for key in test_content_keys:
        file_path = os.path.join(test_all_data_path, key + ".tsv")
        # 提取完整的文档
        with codecs.open(file_path, "w", "utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for line in content_dict[key]:
                if len(line) <= 0:
                    continue
                tsv_writer.writerow(["。".join(line)])
        start, end = my_util.check_evidence_paragraph(content_dict[key])
        evidence_file_path = os.path.join(test_all_data_path, key + "_evidence.tsv")
        # 提取重要段落
        with codecs.open(evidence_file_path, "w", "utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            last_line = None
            for line in content_dict[key][start:end]:
                if len(line) <= 0:
                    continue
                last_line = "" if last_line is None else last_line
                tsv_writer.writerow(["。".join(line), last_line])
                last_line = "。".join(line)


if __name__ == '__main__':
    # analyse_cl_data()
    # analyse_joint_data()
    # generate_data()
    analyse_data_excel_content()
    analyse_data_excel_tags()
    for title in content_dict:
        if title not in tag_dic:
            print("%s\n" % title)
