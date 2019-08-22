# -*-coding:utf-8-*-
import codecs

import re
import numpy as np

from log import logger

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z]')


def read_data(path, separator="\t"):
    batchs = []
    batch = []
    count = 0
    with codecs.open(path, 'r', 'utf-8') as r:
        for line in r:
            line = line.strip()
            if len(line) == 0:
                if batch:
                    batchs.append(batch)
                batch = []
            else:
                split = line.split(separator)
                split[1] = regex.sub('', split[1])
                if len(split[1]) > 800:
                    split[1] = split[1][:800]
                split.append(len(split[1]))
                for i, column in enumerate(split):
                    if len(batch) < i + 1:
                        batch.append([])
                    batch[i].append(column)
        if batch:
            batchs.append(batch)
    return batchs


def get_crf_dataset(text_vocab, label_vocab, path):
    batchs = read_data(path, separator="\t")
    new_batchs = []
    for [labels, sentences, lengths] in batchs:
        max_length = max(lengths)
        labels = [label_vocab[label] for label in labels]
        new_sentences = []
        for sentence in sentences:
            new_sentences.append(
                [text_vocab[word] if word in text_vocab else text_vocab['<unk>'] for word in sentence] + [
                    text_vocab['<pad>']] * (max_length - len(sentence)))
        new_batchs.append([labels, np.array(new_sentences), lengths])
    return new_batchs
