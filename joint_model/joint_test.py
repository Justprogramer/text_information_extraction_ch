# -*-coding:utf-8-*-
import codecs
import json
import os
import pickle

import torch
from torchtext import data
from torchtext.vocab import Vectors

import dataset as cl_dataset
from cl_model_attention import cl_train, cl_model
from ner_model.ner_module import NER


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, file_name, args, **kwargs):
    test_dataset = cl_dataset.generate_test_dataset('.\\test_all_data', text_field, file_name)
    test_iter = data.Iterator.splits(
        (test_dataset,),
        batch_sizes=(args.batch_size,),
        sort_key=lambda x: len(x.text),
        shuffle=False)
    return test_iter[0]


def load_cl_model():
    args_path = os.path.join(".\\snapshot", "cl_args.pkl")
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    text_cnn = cl_model.TextCNN(args)
    print('\nLoading model from {}...\n'.format("snapshot\\cl_model.pkl"))
    text_cnn.load_state_dict(torch.load(".\\snapshot\\cl_model.pkl"))
    return text_cnn, args


text_cnn, args = load_cl_model()
evidence_ner_model = NER()
opinion_ner_model = NER()
evidence_ner_model.load('./ner_saves', type="evidence")
opinion_ner_model.load('./ner_saves', type="opinion")
if args.cuda:
    text_cnn.cuda()

test_all_data = ".\\test_all_data"
files = [f for f in os.listdir(test_all_data) if not f.startswith(".") and f.endswith("_evidence.tsv")]


def type2chinese(type):
    if type == "E":
        return "举证句"
    if type == "O":
        return "无关句"
    if type == "V":
        return "质证句"


for f in files:
    text_field = args.text_field
    args.batch_size = 108
    dataset = load_dataset(text_field, f, args, device=-1, repeat=False)
    preds = cl_train.predict(dataset, text_cnn, args)
    file_path = os.path.join(test_all_data, f)
    result_path = file_path.replace("evidence", "result")
    with codecs.open(file_path, "r", "utf-8") as c:
        lines = c.readlines()
    assert len(preds) == len(lines)
    result = []
    for type, content in zip(preds, lines):
        content = content.split("\t")[0]
        if args.label[type] == "V":
            result.append(("V", content.strip(), opinion_ner_model.predict(content.strip() + "。")))
        elif args.label[type] == "E":
            result.append(("E", content.strip(), evidence_ner_model.predict(content.strip() + "。")))
        else:
            result.append(("O", content.strip()))
    with codecs.open(result_path, "w", "utf-8") as c:
        for r in result:
            c.write("%s\t%s\n" % (r[0], r[1]))
            # c.write("%s\t%s\n" % (type2chinese(r[0]), r[1]))
            if r[0] != "O" and len(r[2]) > 0:
                json.dump(r[2], c, ensure_ascii=False)
                c.write("\n")
