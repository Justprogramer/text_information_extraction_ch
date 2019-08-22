# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(inputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = inputs * weights.unsqueeze(-1)
        return outputs, weights


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.use_crf = args.crf
        self.pad_index = args.pad_index

        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.embedding_dim = embedding_dimension
        self.attention = SelfAttention(self.embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=chanel_num, out_channels=filter_num, kernel_size=(size, self.embedding_dim),
                       padding=0) for size in
             filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * filter_num, class_num),
        )
        if self.use_crf:
            self.crf = CRF(class_num)

    def forward(self, x):
        emissions = self.cnn_forward(x)
        if self.use_crf:
            emissions = emissions.unsqueeze(0)
            return torch.LongTensor(self.crf.decode(emissions))
        return emissions

    def loss(self, x, y):
        emissions = self.cnn_forward(x)
        emissions = emissions.unsqueeze(0)
        loss = self.crf(emissions, y)
        return loss

    def cnn_forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x, attn_weights = self.attention(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
