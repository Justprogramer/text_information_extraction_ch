# -*-coding:utf-8-*-
import torchtext.data as data
from torchtext.vocab import Vectors

import dataset
from joint_model import joint_config
import joint_model
from joint_model.joint_train import *
import dill as pickle


class Config:
    def __init__(self, config):
        for name, value in config.items():
            setattr(self, name, value)


def load_word_vectors(vector_name, vector_path):
    vectors = Vectors(name=vector_name, cache=vector_path)
    return vectors


def load_dataset(text_field, label_field, evidence_tag_field, opinion_tag_field, args, **kwargs):
    train_dataset, dev_dataset = dataset.get_joint_dataset('joint_data', text_field, label_field, evidence_tag_field,
                                                           opinion_tag_field, )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    evidence_tag_field.build_vocab(train_dataset, dev_dataset)
    opinion_tag_field.build_vocab(train_dataset, dev_dataset)
    args.text_field = text_field
    args.evidence_tag_filed = evidence_tag_field
    args.opinion_tag_filed = opinion_tag_field
    args.label_field = label_field
    train_iter = data.Iterator.splits(
        (train_dataset,),
        batch_sizes=(args.batch_size,),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        **kwargs)
    dev_iter = data.Iterator.splits(
        (dev_dataset,),
        batch_sizes=(args.batch_size / args.batch_size,),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=False)
    return train_iter[0], dev_iter[0]


logger.info("Loading data...")
label_field = data.Field(sequential=False, unk_token=None)
text_field = data.Field(lower=True, include_lengths=True)
evidence_tag_field = data.Field(sequential=True, unk_token=None)
opinion_tag_field = data.Field(sequential=True, unk_token=None)
args = Config(joint_config.DEFAULT_CONFIG)

train_iter, dev_iter = load_dataset(text_field, label_field, evidence_tag_field, opinion_tag_field, args, device=-1,
                                    repeat=False, shuffle=True)

args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.label = label_field.vocab.itos
args.evidence_tag = evidence_tag_field.vocab.itos
args.opinion_tag = opinion_tag_field.vocab.itos
# args.evidence_tag.remove("<pad>")
# args.opinion_tag.remove("<pad>")
args.evidence_tag_num = len(args.evidence_tag)
args.opinion_tag_num = len(args.opinion_tag)
logger.info("Parameters:")
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    logger.info('{}={}'.format(attr.upper(), value))
joint_model = joint_model.JointModel(args)
logger.info("{}".format(joint_model))

if args.cuda:
    torch.cuda.set_device(args.device)
    joint_model = joint_model.cuda()
try:
    train(train_iter, dev_iter, joint_model, args)
    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        joint_model.load_state_dict(torch.load(args.snapshot))
        with open(args.snapshot_args, 'rb') as f:
            args = pickle.load(f)
    eval(dev_iter, joint_model, args)
except KeyboardInterrupt:
    print('Exiting from training early')
