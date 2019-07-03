# -*-coding:utf-8-*-
import codecs

import torch
from torchtext import data
from torchtext.data import Dataset, Field, BucketIterator, ReversibleField
from torchtext.vocab import Vectors
from torchtext.datasets import SequenceTaggingDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from util.log import logger
from ner_config import DEVICE, DEFAULT_CONFIG

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def light_tokenize(sequence: str):
    return [sequence]


TEXT = Field(sequential=True, tokenize=light_tokenize, include_lengths=True)
TAG = ReversibleField(sequential=True, tokenize=light_tokenize, is_target=True, unk_token=None)
Fields = [('text', TEXT), ('tag', TAG)]


class CustomSequenceTaggingDataset(SequenceTaggingDataset):
    def __init__(self, path, fields, separator="\t", **kwargs):
        examples = []
        columns = []
        with codecs.open(path, "r", "utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     **kwargs)


class NERTool(object):
    def get_dataset(self, path: str, fields=Fields, separator=' '):
        logger.info('loading dataset from {}'.format(path))
        st_dataset = CustomSequenceTaggingDataset(path, fields=fields, separator=separator)
        logger.info('successed loading dataset')
        return st_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        TEXT.build_vocab(*dataset)
        logger.info('successed building word vocab')
        logger.info('building tag vocab...')
        TAG.build_vocab(*dataset)
        logger.info('successed building tag vocab')
        return TEXT.vocab, TAG.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                     sort_key=lambda x: len(x.text), sort_within_batch=True):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,
                              sort_within_batch=sort_within_batch)

    def get_score(self, model, x, y, field_x, field_y, score_type='f1'):
        metrics_map = {
            'f1': f1_score,
            'p': precision_score,
            'r': recall_score,
            'acc': accuracy_score
        }
        metric_func = metrics_map[score_type] if score_type in metrics_map else metrics_map['f1']
        vec_x = torch.tensor([field_x.stoi[i] for i in x])
        len_vec_x = torch.tensor([len(vec_x)]).to(DEVICE)
        predict_y = model(vec_x.view(-1, 1).to(DEVICE), len_vec_x)[0]
        true_y = [field_y.stoi[i] for i in y]
        assert len(true_y) == len(predict_y)
        return metric_func(predict_y, true_y, average='micro')


ner_tool = NERTool()
