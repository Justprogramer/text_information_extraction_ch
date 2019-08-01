# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.embedding_dim = embedding_dimension + position_embedding_dimension
        self.attention = SelfAttention(self.embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, self.embedding_dim)) for size in
             filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * filter_num, class_num),
        )

    def forward(self, x, position_tensor):
        position_embedding = self.position_embedding(position_tensor)
        if self.embedding2:
            x1 = torch.cat([self.embedding(x), position_embedding], dim=-1)
            x2 = torch.cat([self.embedding2(x), position_embedding], dim=-1)
            x = torch.stack([x1, x2], dim=1)
        else:
            x = torch.cat([self.embedding(x), position_embedding], dim=-1)
            x = x.unsqueeze(1)
        x, attn_weights = self.attention(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
