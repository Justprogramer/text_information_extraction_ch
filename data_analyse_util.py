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
    text = []
    for d in evidence_paragraph_dict:
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
    with codecs.open('./data/%s.tsv' % type, "a", "utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for line in text:
            tsv_writer.writerow([line[0], line[1]])


def create_ner_data(evidence_paragraph_dict, type=None):
    evidence_count = 0
    opinion_count = 0
    for d in evidence_paragraph_dict:
        if d not in evidence_dict or len(evidence_dict[d]) == 0:
            continue
        evidence_content = evidence_paragraph_dict[d]
        for paragraph in evidence_content:
            paragraph = "。".join(paragraph)
            paragraph = my_util.clean_text(paragraph)
            if paragraph[-1] not in symbol_list:
                paragraph += "。"
            tag = ["O"] * len(paragraph)
            paragraph_type = "O"
            for [A, O] in opinion_dict[d]:
                find_o, find_a = -1, -1
                if len(A) > 0:
                    find_a = str(paragraph).find(A)
                    if find_a == 0 and tag[find_a] == "O":
                        tag = tag[:find_a] + ["B-A"] + ["I-A"] * (len(A) - 1) + tag[find_a + len(A):]
                if len(O) > 0:
                    find_o = str(paragraph).find(O)
                    if find_o != -1 and tag[find_o] == "O":
                        tag = tag[:find_o] + ["B-O"] + ["I-O"] * (len(O) - 1) + tag[find_o + len(O):]
                if find_a == 0 and find_o > -1:
                    paragraph_type = "V"
            if paragraph_type == "O":
                tag = ["O"] * len(paragraph)
                for [E, T, C] in evidence_dict[d]:
                    find_e, find_t, find_c = -1, -1, -1
                    if len(E) > 0:
                        find_e = str(paragraph).find(E)
                        if find_e == 0 and tag[find_e] == "O":
                            tag = tag[:find_e] + ["B-E"] + ["I-E"] * (len(E) - 1) + tag[find_e + len(E):]
                    if len(T) > 0:
                        find_t = str(paragraph).find(T)
                        if find_t != -1 and tag[find_t] == "O":
                            tag = tag[:find_t] + ["B-T"] + ["I-T"] * (len(T) - 1) + tag[find_t + len(T):]
                    if len(C) > 0:
                        find_c = str(paragraph).find(C)
                        if find_c != -1 and tag[find_c] == "O":
                            tag = tag[:find_c] + ["B-C"] + ["I-C"] * (len(C) - 1) + tag[find_c + len(C):]
                    if find_e == 0 and find_t > -1 or find_c > -1:
                        paragraph_type = "E"
            assert len(paragraph) == len(tag)
            if paragraph_type == "V":
                for i, label in enumerate(tag):
                    if label not in ["O", "B-A", "I-A", "B-O", "I-O"]:
                        tag[i] = tag[i - 1]
                opinion_count += 1
                with codecs.open('./data/%s_opinion.txt' % type, "a", "utf-8") as f:
                    for i, word in enumerate(paragraph):
                        f.write("%s %s\n" % (word, tag[i]))
                    f.write("\n")
            if paragraph_type == "E":
                for i, label in enumerate(tag):
                    if label not in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"]:
                        tag[i] = tag[i - 1]
                evidence_count += 1
                with codecs.open('./data/%s_evidence.txt' % type, "a", "utf-8") as f:
                    for i, word in enumerate(paragraph):
                        f.write("%s %s\n" % (word, tag[i]))
                    f.write("\n")
    print("\ncount:evidence[%d],opinion[%d]" % (evidence_count, opinion_count))


def create_joint_data(evidence_paragraph_dict, type=None):
    global other_count, evidence_count, view_count
    text = []
    for d in evidence_paragraph_dict:
        if d not in evidence_dict or len(evidence_dict[d]) == 0:
            continue
        evidence_content = evidence_paragraph_dict[d]
        for paragraph in evidence_content:
            paragraph = "。".join(paragraph)
            paragraph = my_util.clean_text(paragraph)
            if paragraph[-1] not in symbol_list:
                paragraph += "。"
            tag = ["O"] * len(paragraph)
            paragraph_type = "O"
            if len(paragraph) <= 4:
                continue
            for [A, O] in opinion_dict[d]:
                find_o, find_a = -1, -1
                if len(A) > 0:
                    find_a = str(paragraph).find(A)
                    if find_a == 0 and tag[find_a] == "O":
                        tag = tag[:find_a] + ["B-A"] + ["I-A"] * (len(A) - 1) + tag[find_a + len(A):]
                if len(O) > 0:
                    find_o = str(paragraph).find(O)
                    if find_o != -1 and tag[find_o] == "O":
                        tag = tag[:find_o] + ["B-O"] + ["I-O"] * (len(O) - 1) + tag[find_o + len(O):]
                if find_a == 0 and find_o > -1:
                    paragraph_type = "V"
            if paragraph_type == "O":
                tag = ["O"] * len(paragraph)
                for [E, T, C] in evidence_dict[d]:
                    find_e, find_t, find_c = -1, -1, -1
                    if len(E) > 0:
                        find_e = str(paragraph).find(E)
                        if find_e == 0 and tag[find_e] == "O":
                            tag = tag[:find_e] + ["B-E"] + ["I-E"] * (len(E) - 1) + tag[find_e + len(E):]
                    if len(T) > 0:
                        find_t = str(paragraph).find(T)
                        if find_t != -1 and tag[find_t] == "O":
                            tag = tag[:find_t] + ["B-T"] + ["I-T"] * (len(T) - 1) + tag[find_t + len(T):]
                    if len(C) > 0:
                        find_c = str(paragraph).find(C)
                        if find_c != -1 and tag[find_c] == "O":
                            tag = tag[:find_c] + ["B-C"] + ["I-C"] * (len(C) - 1) + tag[find_c + len(C):]
                    if find_e == 0 and find_t > -1 or find_c > -1:
                        paragraph_type = "E"
            assert len(paragraph) == len(tag)
            if paragraph_type == "E":
                for i, label in enumerate(tag):
                    if label not in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"]:
                        tag[i] = tag[i - 1] if tag[i - 1] in ["O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"] else "O"
                text.append(["E", paragraph, tag, ["O"] * len(paragraph)])
                evidence_count += 1
            elif paragraph_type == "V":
                for i, label in enumerate(tag):
                    if label not in ["O", "B-A", "I-A", "B-O", "I-O"]:
                        tag[i] = tag[i - 1] if tag[i - 1] in ["O", "B-A", "I-A", "B-O", "I-O"] else "O"
                text.append(["V", paragraph, ["O"] * len(paragraph), tag])
                view_count += 1
            else:
                if type == "train" and other_count % 10 == 0:
                    text.append(["O", paragraph, ["O"] * len(paragraph), ["O"] * len(paragraph)])
                else:
                    text.append(["O", paragraph, ["O"] * len(paragraph), ["O"] * len(paragraph)])
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


def title_with_no_tag():
    analyse_data_excel_content()
    analyse_data_excel_tags()
    for title in content_dict:
        if title not in evidence_dict:
            print("%s\n" % title)


if __name__ == '__main__':
    analyse_cl_data()
    # analyse_ner_data()
    # analyse_joint_data()
    # generate_data()
