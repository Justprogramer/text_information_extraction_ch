# -*-coding:utf-8-*-
import codecs
import os
import pickle
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score


def construct_features(batch, args):
    (feature1, lenght1), (feature2, lenght2), (feature3, lenght3) \
        = batch.text1, batch.text2, batch.text3
    feature1 = feature1.data.t()
    feature2 = feature2.data.t()
    feature3 = feature3.data.t()
    features = None
    position = None
    max_length = feature1.size()[-1] + feature2.size()[-1] + feature3.size()[-1]
    for index, (feat_1, feat_2, feat_3) in enumerate(zip(feature1, feature2, feature3)):
        pad_length = max_length - lenght1[index] - lenght2[index] - lenght3[index]
        pad_tensor = torch.LongTensor([args.text_field.vocab.stoi['<pad>']]).expand(pad_length)
        feat_tensor = torch.cat(
            (feat_1[:lenght1[index]], feat_2[:lenght2[index]], feat_3[:lenght3[index]], pad_tensor), -1).unsqueeze(0)
        pad_tensor = torch.cat(
            (torch.ones(lenght1[index]), torch.zeros(max_length - lenght1[index])), -1).unsqueeze(0)
        if features is None:
            features = feat_tensor
            position = pad_tensor
        else:
            features = torch.cat([features, feat_tensor], 0)
            position = torch.cat([position, pad_tensor], 0)
    max_length = features.ne(1).sum(1).max()
    return features[:, :max_length], position[:, :max_length].long(), batch.label


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_rate)
    best_f1 = 0
    patience = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        patience += 1
        steps = 0
        if args.lr_decay != 0.:
            optimizer = decay_learning_rate(optimizer, args.lr_decay, epoch, args.lr)
        for batch in train_iter:
            features, position_tensor, target = construct_features(batch, args)
            target = target.data.sub(1)
            if args.cuda:
                features, position_tensor, target = features.cuda(), position_tensor.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(features, position_tensor)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rpatience[{}]- epoch[{}]-Batch[{}] - loss: {:.6f}  acc: {:.4f}% ({}/{})'.format(patience,
                                                                                                      epoch,
                                                                                                      steps,
                                                                                                      loss.item(),
                                                                                                      train_acc,
                                                                                                      corrects,
                                                                                                      batch.batch_size))

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
        features, length_main, target = construct_features(batch, args)
        target = target.data.sub(1)
        if args.cuda:
            features, length_main, target = features.cuda(), length_main.cuda(), target.cuda()
        with torch.no_grad():
            logits = model(features, length_main)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        preds.extend(list(torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy()))
        golds.extend(list(target.data.cpu().numpy()))
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    f1 = f1_score(y_true=golds, y_pred=preds, average="macro")
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%   f1: {:.4f} %({}/{}) \n'.format(avg_loss,
                                                                                      accuracy,
                                                                                      f1,
                                                                                      corrects,
                                                                                      size))

    with codecs.open("cl_result.txt", "w", "utf-8") as w:
        for index, example in enumerate(data_iter.dataset.examples):
            w.write("%s\t%s\t%s\n" % (example.label, args.label[preds[index]], "".join(example.text1)))
        w.write("\n%s\n" % classification_report(y_true=golds, y_pred=preds, target_names=args.label))
    return accuracy, f1


def predict(data_iter, model, args):
    model.eval()
    preds = []
    for batch in data_iter:
        feature1, feature2 = batch.text1, batch.text2
        feature1 = feature1.data.t()
        feature2 = feature2.data.t()
        if args.cuda:
            feature1 = feature1.cuda()
            feature2 = feature2.cuda()
        with torch.no_grad():
            logits = model(feature1, feature2)
        preds.extend(list(torch.max(logits, 1)[1].data.cpu().numpy()))
    return preds


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
