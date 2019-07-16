# -*-coding:utf-8-*-
import codecs
import os
import pickle
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score


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
            feature1, feature2, target = batch.text1, batch.text2, batch.label
            feature1 = feature1.data.t()
            feature2 = feature2.data.t()
            target = target.data.sub(1)
            if args.cuda:
                feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature1, feature2)
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
                raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    preds = []
    golds = []
    for batch in data_iter:
        feature1, feature2, target = batch.text1, batch.text2, batch.label
        feature1 = feature1.data.t()
        feature2 = feature2.data.t()
        target = target.data.sub(1)
        if args.cuda:
            feature1, feature2, target = feature1.cuda(), feature2.cuda(), target.cuda()
        with torch.no_grad():
            logits = model(feature1, feature2)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        preds.extend(list(torch.max(logits, 1)[1].view(target.size()).data.cpu().numpy()))
        golds.extend(list(target.data.cpu().numpy()))
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    f1 = f1_score(y_true=golds, y_pred=preds, average="micro")
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%   f1: {:.4f} %({}/{}) \n'.format(avg_loss,
                                                                                      accuracy,
                                                                                      f1,
                                                                                      corrects,
                                                                                      size))

    with codecs.open("result.txt", "w", "utf-8") as w:
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
