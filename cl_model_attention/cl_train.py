# -*-coding:utf-8-*-
import codecs
import os
import pickle
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score


# def construct_features(batch, args):
#     (feature1, lenght1), (feature2, lenght2), (feature3, lenght3) \
#         = batch.text1, batch.text2, batch.text3
#     feature1 = feature1.data.t()
#     feature2 = feature2.data.t()
#     feature3 = feature3.data.t()
#     features = None
#     position = None
#     max_length = feature1.size()[-1] + feature2.size()[-1] + feature3.size()[-1]
#     for index, (feat_1, feat_2, feat_3) in enumerate(zip(feature1, feature2, feature3)):
#         pad_length = max_length - lenght1[index] - lenght2[index] - lenght3[index]
#         pad_tensor = torch.LongTensor([args.text_field.vocab.stoi['<pad>']]).expand(pad_length)
#         feat_tensor = torch.cat(
#             (feat_1[:lenght1[index]], feat_2[:lenght2[index]], feat_3[:lenght3[index]], pad_tensor), -1).unsqueeze(0)
#         pad_tensor = torch.cat(
#             (torch.ones(lenght1[index]), torch.zeros(max_length - lenght1[index])), -1).unsqueeze(0)
#         if features is None:
#             features = feat_tensor
#             position = pad_tensor
#         else:
#             features = torch.cat([features, feat_tensor], 0)
#             position = torch.cat([position, pad_tensor], 0)
#     max_length = features.ne(1).sum(1).max()
#     return features[:, :max_length], position[:, :max_length].long(), batch.label


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1 = 0
    patience = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        patience += 1
        steps = 0
        if args.lr_decay != 0.:
            optimizer = decay_learning_rate(optimizer, args.lr_decay, epoch, args.lr)
        for batch in train_iter:
            if args.crf:
                target, features, length = batch
                target = torch.LongTensor(target).data.sub(1).view(1, -1)
                features = torch.LongTensor(features)
                length = torch.LongTensor(length)
            else:
                (features, length), target = batch.text, batch.label
                features = features.data.t()
                target = target.data.sub(1)
            if args.cuda:
                features, target = features.cuda(), target.cuda()
            optimizer.zero_grad()
            if args.crf:
                loss = (-model.loss(features, target)) / target.size(1)
                preds = model(features).view(target.size())
                if args.cuda:
                    preds = preds.cuda()
                corrects = (preds.data == target.data).sum()
            else:
                logits = model(features)
                loss = F.cross_entropy(logits, target)
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                if args.crf:
                    batch_size = len(length)
                else:
                    batch_size = batch.batch_size
                train_acc = 100.0 * corrects / batch_size
                sys.stdout.write(
                    '\rpatience[{}]- epoch[{}]-Batch[{}] - loss: {:.6f}  acc: {:.4f}% ({}/{})'.format(patience,
                                                                                                      epoch,
                                                                                                      steps,
                                                                                                      loss.item(),
                                                                                                      train_acc,
                                                                                                      corrects,
                                                                                                      batch_size))
        dev_acc, f1 = eval(dev_iter, model, args)
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            if args.save_best:
                print('Saving best model, f1: {:.4f}%\n'.format(best_f1))
                save(model, args)
        else:
            if patience >= args.max_patience:
                print('\nearly stop by {} patience, f1: {:.4f}%'.format(args.max_patience, best_f1))
                break


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    preds = []
    golds = []
    for batch in data_iter:
        if args.crf:
            target, features, length = batch
            target = torch.LongTensor(target).data.sub(1).view(1, -1)
            features = torch.LongTensor(features)
            length = torch.LongTensor(length)
        else:
            (features, length), target = batch.text, batch.label
            features = features.data.t()
            target = target.data.sub(1)
        if args.cuda:
            features, target = features.cuda(), target.cuda()
        with torch.no_grad():
            if args.crf:
                loss = (-model.loss(features, target)) / target.size(1)
                predicts = model(features).view(target.size())
                if args.cuda:
                    predicts = predicts.cuda()
                corrects = (predicts.data == target.data).sum()
                preds.extend(predicts.view(-1).data.cpu().numpy().tolist())
            else:
                logits = model(features)
                loss = F.cross_entropy(logits, target)
                corrects += (torch.max(logits, 1)
                             [1].view(target.size()).data == target.data).sum()
                preds.extend(list(torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy()))
        avg_loss += loss.item()
        golds.extend(target.view(-1).data.cpu().numpy().tolist())
        assert len(golds) == len(preds)
    if args.crf:
        size = len(data_iter)
    else:
        size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    f1 = f1_score(y_true=golds, y_pred=preds, average="macro")
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%   f1: {:.4f} %({}/{}) \n'.format(avg_loss,
                                                                                      accuracy,
                                                                                      f1,
                                                                                      corrects,
                                                                                      size))
    if args.crf:
        with codecs.open("cl_result.txt", "w", "utf-8") as w:
            start = 0
            for [target, features, _] in data_iter:
                for index, (label, feature) in enumerate(zip(target, features)):
                    sentence = [args.text_field.vocab.itos[ix] if ix != args.pad_index else "" for ix in feature]
                    w.write(
                        "%s\t%s\t%s\n" % (args.label[label - 1], args.label[preds[index + start]], "".join(sentence)))
                w.write("\n")
                start += len(target)
            w.write("\n%s\n" % classification_report(y_true=golds, y_pred=preds, target_names=args.label))
        return accuracy, f1
    else:
        with codecs.open("cl_result.txt", "w", "utf-8") as w:
            for index, example in enumerate(data_iter.dataset.examples):
                w.write("%s\t%s\t%s\n" % (example.label, args.label[preds[index]], "".join(example.text)))
            w.write("\n%s\n" % classification_report(y_true=golds, y_pred=preds, target_names=args.label))
        return accuracy, f1


def predict(data_iter, model, args):
    model.eval()
    preds = []
    for batch in data_iter:
        if args.crf:
            target, features, length = batch
            target = torch.LongTensor(target).data.sub(1).view(1, -1)
            features = torch.LongTensor(features)
            length = torch.LongTensor(length)
        else:
            (features, length), target = batch.text, batch.label
            features = features.data.t()
            target = target.data.sub(1)
        if args.cuda:
            features, target = features.cuda(), target.cuda()
        with torch.no_grad():
            if args.crf:
                return model(features).view(target.size())
            else:
                return torch.max(model(features), 1)[1].view(-1).data.cpu().numpy().tolist


def save(model, args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, "cl_model.pkl")
    torch.save(model.state_dict(), save_path)
    args_path = os.path.join(args.save_dir, "cl_args.pkl")
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)


def decay_learning_rate(optimizer, lr_decay, epoch, init_lr):
    """衰减学习率

    Args:
        epoch: int, 迭代次数
        init_lr: 初始学习率
    """
    lr = init_lr / (1 + lr_decay * epoch)
    print('\nlearning rate: {%s}\n' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
