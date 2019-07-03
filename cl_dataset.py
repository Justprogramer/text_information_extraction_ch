# -*-coding:utf-8-*-
import re
from torchtext import data
import jieba
import logging

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=False,
        train='train.tsv', validation='test.tsv',
        fields=[
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev


def generate_test_dataset(path, text_field, file_name):
    text_field.tokenize = word_cut
    test = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=False,
        train=file_name,
        fields=[
            ('text', text_field),
        ]
    )
    return test[0]
