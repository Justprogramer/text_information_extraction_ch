# -*-coding:utf-8-*-
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_CONFIG = {
    'lr': 0.015,
    'epoch': 1000,
    'lr_decay': 0.05,
    'l2_rate': 1.0e-8,
    'momentum': 0.,
    'batch_size': 128,
    'dropout': 0.5,
    'static': True,
    'non_static': False,
    'embedding_dim': 300,
    'num_layers': 2,
    'pad_index': 1,
    'vector_path': '',
    'tag_num': 0,
    'vocabulary_size': 0,
    'word_vocab': None,
    'tag_vocab': None,
    'save_path': '../saves',
    'pretrained_path': '../pretrained',
    'pretrained_name': 'sgns.zhihu.word',
    'max_patience': 20,
}
