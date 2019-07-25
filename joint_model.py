# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.log import logger

from TorchCRF import CRF

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LabelingLayer(nn.Module):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        if args.multichannel:
            linear_input = self.args.hidden_dim * 2
        else:
            linear_input = self.args.hidden_dim
        assert self.name in ["evidence", "opinion"]
        if self.name == 'evidence':
            self.tag_num = self.args.evidence_tag_num
        else:
            self.tag_num = self.args.opinion_tag_num
        self.hidden2label = nn.Linear(linear_input, self.tag_num)
        self.crflayer = CRF(self.tag_num, batch_first=True)

    def loss(self, encoder_output, mask, y):
        emissions = self.hidden2label(encoder_output)
        loss = self.crflayer(emissions, y, mask=mask)
        return loss

    def forward(self, encoder_output, mask):
        emissions = self.hidden2label(encoder_output)
        return self.crflayer.decode(emissions, mask=mask)


class JointModel(nn.Module):
    def __init__(self, args):
        super(JointModel, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.bidirectional = args.bidirectional
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout
        self.save_path = args.save_path
        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        # embedding layer
        embedding_dimension = args.embedding_dim
        # position_embedding_dimension = args.position_embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(2, args.position_embedding_dim)
        if args.static:
            logger.info('logging word vectors from {}/{}'.format(args.pretrained_path, args.pretrained_name))
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        # encoder layer：bi-LSTM
        self.encoder = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                               num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        if args.multichannel:
            self.encoder2 = nn.LSTM(embedding_dimension, self.hidden_dim // 2, bidirectional=self.bidirectional,
                                    num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        # classifier layer
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in
             filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)
        #     sequence labeling layer
        self.evidence_layer = LabelingLayer('evidence', args)
        self.opinion_layer = LabelingLayer('opinion', args)

    def init_weight(self):
        nn.init.xavier_normal_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.hidden2label.weight)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2)
        if self.args.cuda:
            return h0.cuda(), c0.cuda()
        return h0, c0

    def encoder_forward(self, x, sent_lengths):
        if self.embedding2:
            x1 = self.embedding(x)
            x1 = pack_padded_sequence(x1, sent_lengths, batch_first=True)
            self.hidden = self.init_hidden(batch_size=len(sent_lengths))
            lstm_out, self.hidden = self.encoder(x1, self.hidden)
            lstm_out, new_batch_size = pad_packed_sequence(lstm_out, batch_first=True)
            x2 = self.embedding2(x)
            x2 = pack_padded_sequence(x2, sent_lengths, batch_first=True)
            self.hidden = self.init_hidden(batch_size=len(sent_lengths))
            lstm_out2, self.hidden = self.encoder(x2, self.hidden)
            lstm_out2, new_batch_size = pad_packed_sequence(lstm_out2, batch_first=True)
            assert torch.equal(sent_lengths, new_batch_size)
            x_classifier = torch.stack([lstm_out, lstm_out2], dim=1)
            x_sequence = torch.cat([lstm_out, lstm_out2], dim=-1)
        else:
            x = self.embedding(x)
            x = pack_padded_sequence(x, sent_lengths, batch_first=True)
            self.hidden = self.init_hidden(batch_size=len(sent_lengths))
            lstm_out, self.hidden = self.encoder(x, self.hidden)
            lstm_out, new_batch_size = pad_packed_sequence(lstm_out, batch_first=True)
            x_classifier = lstm_out.unsqueeze(1)
            x_sequence = lstm_out
            assert torch.equal(sent_lengths, new_batch_size)
        return x_classifier, x_sequence

    def classifier_forward(self, lstm_out):
        classifier = [F.relu(conv(lstm_out)).squeeze(3) for conv in self.classifiers]
        classifier = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in classifier]
        classifier = torch.cat(classifier, 1)
        classifier = self.dropout(classifier)
        logits = self.fc(classifier)
        return logits

    def sequence_loss(self, lstm_features, mask, y, type="evidence"):
        if type == 'evidence':
            return self.evidence_layer.loss(lstm_features, mask, y)
        else:
            return self.opinion_layer.loss(lstm_features, mask, y)

    def sequence_predict(self, lstm_features, mask, type="evidence"):
        if type == 'evidence':
            return self.evidence_layer(lstm_features, mask)
        else:
            return self.opinion_layer(lstm_features, mask)

    def forward(self, x, sent_lengths, tags, labels):
        evidence_loss, opinion_loss = None, None
        x_classifier, x_sequence = self.encoder_forward(x, sent_lengths)
        classifier_logit = self.classifier_forward(x_classifier)

        mask = torch.ne(x, self.args.pad_index)
        evidence_index = [index for index, label_ix in enumerate(labels) if "E" == self.args.label[label_ix]]
        evidence_mask = mask[evidence_index]
        evidence_tags = tags[evidence_index]
        evidence_features = x_sequence[evidence_index]
        # evidence_sent_lengths = sent_lengths[evidence_index]
        if len(evidence_index) > 0:
            evidence_loss = (-self.sequence_loss(evidence_features, evidence_mask, evidence_tags,
                                                 'evidence')) / len(
                evidence_index)

        opinion_index = [index for index, label_ix in enumerate(labels) if "V" == self.args.label[label_ix]]
        opinion_mask = mask[opinion_index]
        opinion_tags = tags[opinion_index]
        opinion_features = x_sequence[opinion_index]
        # opinion_sent_lengths = sent_lengths[opinion_index]
        if len(opinion_index) > 0:
            opinion_loss = (-self.sequence_loss(opinion_features, opinion_mask, opinion_tags,
                                                'opinion')) / len(
                opinion_index)
        return classifier_logit, evidence_loss, opinion_loss

    def eval_forward(self, x, sent_lengths, tags, labels):
        x_classifier, x_sequence = self.encoder_forward(x, sent_lengths)
        classifier_logit = self.classifier_forward(x_classifier)
        mask = torch.ne(x, self.args.pad_index)
        evidence_index = [index for index, label_ix in enumerate(labels) if "E" == self.args.label[label_ix]]
        evidence_mask = mask[evidence_index]
        evidence_tags = tags[evidence_index]
        gold_evidence = list(torch.masked_select(evidence_tags, evidence_mask).view(-1).data.cpu().numpy())
        evidence_features = x_sequence[evidence_index]
        # evidence_sent_lengths = sent_lengths[evidence_index]
        predict_evidence = []
        if len(evidence_index) > 0:
            evidence = self.sequence_predict(evidence_features, evidence_mask, 'evidence')
            for e in evidence:
                predict_evidence.extend(e)
        assert len(predict_evidence) == len(gold_evidence)
        opinion_index = [index for index, label_ix in enumerate(labels) if "V" == self.args.label[label_ix]]
        opinion_mask = mask[opinion_index]
        opinion_tags = tags[opinion_index]
        gold_opinion = list(torch.masked_select(opinion_tags, opinion_mask).view(-1).data.cpu().numpy())
        opinion_features = x_sequence[opinion_index]
        # opinion_sent_lengths = sent_lengths[opinion_index]
        predict_opinion = []
        if len(opinion_index) > 0:
            opinion = self.sequence_predict(opinion_features, opinion_mask, 'opinion')
            for o in opinion:
                predict_opinion.extend(o)
        assert len(predict_opinion) == len(gold_opinion)
        return classifier_logit, predict_evidence, gold_evidence, predict_opinion, gold_opinion

    def predict(self, x, sent_lengths):
        x_classifier, x_sequence = self.encoder_forward(x, sent_lengths)
        classifier_logit = self.classifier_forward(x_classifier)
        labels = torch.max(classifier_logit, 1)[1].view(-1).data.cpu().numpy()
        mask = torch.ne(x, self.args.pad_index)
        evidence_index = [index for index, label_ix in enumerate(labels) if "E" == self.args.label[label_ix]]
        evidence_mask = mask[evidence_index]
        evidence_features = x_sequence[evidence_index]
        evidence = None
        if len(evidence_index) > 0:
            evidence = self.sequence_predict(evidence_features, evidence_mask, 'evidence')
        opinion_index = [index for index, label_ix in enumerate(labels) if "V" == self.args.label[label_ix]]
        opinion_mask = mask[opinion_index]
        opinion_features = x_sequence[opinion_index]
        opinion = None
        if len(opinion_index) > 0:
            opinion = self.sequence_predict(opinion_features, opinion_mask, 'opinion')
        return labels, evidence_index, evidence, opinion_index, opinion
