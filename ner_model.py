import os
import pickle

import torch
import torch.nn as nn
from TorchCRF import CRF
from torchtext.vocab import Vectors
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util.log import logger
from ner_config import DEVICE, DEFAULT_CONFIG


class Config(object):
    def __init__(self, word_vocab, tag_vocab, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.tag_num = len(self.tag_vocab)
        self.vocabulary_size = len(self.word_vocab)
        for name, value in kwargs.items():
            setattr(self, name, value)

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path'], type="evidence"):
        config_path = os.path.join(path, '%s_config.pkl' % type)
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logger.info('loading config from {}'.format(config_path))
        return config

    def save(self, path=None, type="evidence"):
        if not hasattr(self, 'save_path'):
            raise AttributeError('config object must init save_path attr in init method!')
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, '%s_config.pkl' % type)
        with open(os.path.join(path, '%s_config.pkl' % type), 'wb') as f:
            pickle.dump(self, f)
        logger.info('saved config to {}'.format(config_path))


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None, type="evidence"):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, '%s_model.pkl' % type)
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loadding model from {}'.format(model_path))

    def save(self, path=None, type="evidence"):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, '%s_model.pkl' % type)
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))


class BiLstmCrf(BaseModel):
    def __init__(self, args):
        super(BiLstmCrf, self).__init__(args)
        self.args = args
        self.hidden_dim = 300
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        if args.static:
            logger.info('logging word vectors from {}/{}'.format(args.pretrained_path, args.pretrained_name))
            vectors = Vectors(args.pretrained_name, args.pretrained_path).vectors
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.non_static).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        self.hidden2label = nn.Linear(self.hidden_dim, self.tag_num).to(DEVICE)
        self.crflayer = CRF(self.tag_num).to(DEVICE)

        # self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2).to(DEVICE)

        return h0, c0

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        loss = self.crflayer(emissions, y, mask=mask)
        return loss

    def forward(self, x, sent_lengths):
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        return self.crflayer.decode(emissions, mask=mask)

    def lstm_forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(DEVICE)).to(DEVICE)
        x = pack_padded_sequence(x, sent_lengths)
        self.hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)
        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE))
        y = self.hidden2label(lstm_out.to(DEVICE))
        return y.to(DEVICE)
