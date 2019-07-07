# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        position_embedding_dimension = args.position_embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(2, args.position_embedding_dim)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension + position_embedding_dimension)) for size in
             filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x1, x2):
        zeros = torch.zeros([x2.size(0), x2.size(1)]).long()
        ones = torch.ones([x1.size(0), x1.size(1)]).long()
        if self.args.cuda:
            zeros = zeros.cuda()
            ones = ones.cuda()
        ones_embedding = self.position_embedding(ones)
        zeros_embedding = self.position_embedding(zeros)
        if self.embedding2:
            x1_1 = torch.cat([self.embedding(x1), ones_embedding], dim=-1)
            x1_2 = torch.cat([self.embedding2(x1), ones_embedding], dim=-1)
            x2_1 = torch.cat([self.embedding(x2), zeros_embedding], dim=-1)
            x2_2 = torch.cat([self.embedding2(x2), zeros_embedding], dim=-1)
            x1 = torch.stack([x1_1, x1_2], dim=1)
            x2 = torch.stack([x2_1, x2_2], dim=1)
            x = torch.cat([x1, x2], dim=-2)
        else:
            x1 = torch.cat([self.embedding(x1), ones_embedding], dim=-1)
            x2 = torch.cat([self.embedding(x2), zeros_embedding], dim=-1)
            x = torch.cat([x1, x2], dim=-2)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
