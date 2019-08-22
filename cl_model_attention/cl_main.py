# -*-coding:utf-8-*-
import argparse

import torch
import torchtext.data as data
from torchtext.vocab import Vectors

from cl_model_attention import cl_train, cl_model, cl_tool
import dataset

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.015, help='initial learning rate [default: 0.015]')
parser.add_argument('-momentum', type=float, default=0., help='initial momentum [default: 0.]')
parser.add_argument('-l2_rate', type=float, default=1.0e-8, help='initial l2_rate [default: 1.0e-8]')
parser.add_argument('-lr_decay', type=float, default=0.05, help='initial learning rate ecay [default: 0.05]')
parser.add_argument('-epochs', type=int, default=10000, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-save-dir', type=str, default='../cl_snapshot', help='where to save the snapshot')
parser.add_argument('-max_patience', type=int, default=10,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-position_embedding_dim', type=int, default=20,
                    help='number of position embedding dimension [default: 5]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('-non-static', type=bool, default=True,
                    help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=True, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.renmin.bigram-char',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='../pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default='../cl_snapshot/cl_model.pkl',
                    help='filename of model snapshot [default: None]')
# option
parser.add_argument('-crf', type=bool, default=True, help='whether use crf')
args = parser.parse_args()

torch.manual_seed(args.seed)


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset = dataset.get_dataset('../data', text_field, label_field)
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    args.text_field = text_field
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter = data.Iterator.splits(
        (train_dataset,),
        batch_sizes=(args.batch_size,),
        sort_key=lambda x: len(x.text),
        **kwargs)
    dev_iter = data.Iterator.splits(
        (dev_dataset,),
        batch_sizes=(args.batch_size,),
        sort_key=lambda x: len(x.text),
        shuffle=False)
    return train_iter[0], dev_iter[0]


print('Loading data...')
text_field = data.Field(include_lengths=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
if args.crf:
    train_iter = cl_tool.get_crf_dataset(text_field.vocab.stoi, label_field.vocab.stoi, '../data/crf_cl_train.tsv')
    dev_iter = cl_tool.get_crf_dataset(text_field.vocab.stoi, label_field.vocab.stoi, '../data/crf_cl_test.tsv')

args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.pad_index = args.text_field.vocab.stoi['<pad>']
args.class_num = len(label_field.vocab) - 1
args.label = label_field.vocab.itos
args.label.remove('<unk>')
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in str(args.filter_sizes).split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))
text_cnn = cl_model.TextCNN(args)

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()
try:
    cl_train.train(train_iter, dev_iter, text_cnn, args)
    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        text_cnn.load_state_dict(torch.load(args.snapshot))
    cl_train.eval(dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')