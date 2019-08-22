# -*-coding:utf-8-*-
import codecs
import os
import dill as pickle

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

from util.log import logger


def train(train_iter, dev_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1 = 0
    patience = 0
    sequence_loss = []
    class_loss = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        patience += 1
        steps = 0
        if args.lr_decay != 0.:
            optimizer = decay_learning_rate(optimizer, args.lr_decay, epoch, args.lr)
        for batch in train_iter:
            (feature, lengths), evidence_tags, opinion_tags, label = \
                batch.text, batch.evidence_tag, batch.opinion_tag, batch.label
            tags = torch.ones(evidence_tags.size()).t()
            feature = feature.data.t()
            evidence_tags = evidence_tags.data.t()
            opinion_tags = opinion_tags.data.t()
            for i, l in enumerate(list(label.data.numpy())):
                if args.label[l] == "E":
                    tags[i] = evidence_tags[i]
                elif args.label[l] == "V":
                    tags[i] = opinion_tags[i]
            loss = torch.zeros([1])
            if args.cuda:
                feature, tags, label, loss = feature.cuda(), tags.cuda().long(), label.cuda(), loss.cuda()
            model.zero_grad()
            classifier_logit, evidence_loss, opinion_loss = model(feature, lengths, tags, label)
            if evidence_loss is not None:
                evidence_loss.backward(retain_graph=True)
                loss += evidence_loss
            if opinion_loss is not None:
                opinion_loss.backward(retain_graph=True)
                loss += opinion_loss
            classifier_loss = F.cross_entropy(classifier_logit, label)
            classifier_loss.backward()
            optimizer.step()
            sequence_loss.append(loss.item())
            class_loss.append(classifier_loss.item())
            steps += 1
            corrects = (torch.max(classifier_logit, 1)[1].view(label.size()).data == label.data).sum()
            train_acc = 100.0 * corrects / batch.batch_size
            logger.info(
                'patience[{}]- epoch[{}]-Batch[{}] - classifier loss: {:.6f}'
                ' labeling loss: {:.6f} acc: {:.4f}% ({}/{})'.format(patience,
                                                                     epoch,
                                                                     steps,
                                                                     classifier_loss.item(),
                                                                     loss.item(),
                                                                     train_acc,
                                                                     corrects,
                                                                     batch.batch_size))

        logger.info(
            'patience[{}]- epoch[{}] - classifier avg loss: {:.6f}'
            ' labeling avg loss: {:.6f}'.format(patience,
                                                epoch,
                                                sum(class_loss) / len(class_loss),
                                                sum(sequence_loss) / len(sequence_loss)
                                                ))
        avg_f1 = eval(dev_iter, model, args)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience = 0
            if args.save_best:
                print('Saving best model, avg_f1: {:.4f}%\n'.format(best_f1))
                save(model, args)
        else:
            if patience >= args.max_patience:
                print('\nearly stop by {} patience, f1: {:.4f}%'.format(args.max_patience, best_f1))
                raise KeyboardInterrupt


def predict(data_iter, model, args):
    model.eval()
    for batch in data_iter:
        (feature, lengths), evidence_tags, opinion_tags, label = \
            batch.text, batch.evidence_tag, batch.opinion_tag, batch.label
        tags = torch.ones(evidence_tags.size()).t()
        feature = feature.data.t()
        evidence_tags = evidence_tags.data.t()
        opinion_tags = opinion_tags.data.t()
        for i, l in enumerate(list(label.data.numpy())):
            if args.label[l] == "E":
                tags[i] = evidence_tags[i]
            elif args.label[l] == "V":
                tags[i] = opinion_tags[i]
        loss = torch.zeros([1])
        if args.cuda:
            feature, tags, label, loss = feature.cuda(), tags.cuda().long(), label.cuda(), loss.cuda()
        with torch.no_grad():
            labels, evidence_index, evidence, opinion_index, opinion = model.predict(feature, lengths)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    text_list = []
    preds_class = []
    golds_class = []
    evidence_golds = []
    evidence_predicts = []
    opinion_golds = []
    opinion_predicts = []
    for batch in data_iter:
        (feature, lengths), evidence_tags, opinion_tags, label = \
            batch.text, batch.evidence_tag, batch.opinion_tag, batch.label
        tags = torch.ones(evidence_tags.size()).t()
        feature = feature.data.t()
        evidence_tags = evidence_tags.data.t()
        opinion_tags = opinion_tags.data.t()
        if feature.size(1) < args.filter_sizes[-1]:
            feature = torch.cat([feature, feature, feature], -1)
            lengths = lengths * 3
            evidence_tags = torch.cat([evidence_tags, evidence_tags, evidence_tags], -1)
            opinion_tags = torch.cat([opinion_tags, opinion_tags, opinion_tags], -1)
            tags = torch.cat([tags, tags, tags], -1)
        for i, l in enumerate(list(label.data.numpy())):
            if args.label[l] == "E":
                tags[i] = evidence_tags[i]
            elif args.label[l] == "V":
                tags[i] = opinion_tags[i]
        loss = torch.zeros([1])
        if args.cuda:
            feature, tags, label, loss = feature.cuda(), tags.cuda().long(), label.cuda(), loss.cuda()
        with torch.no_grad():
            classifier_logit, predict_evidence, gold_evidence, predict_opinion, gold_opinion = model.eval_forward(
                feature,
                lengths,
                tags,
                label)
            if len(predict_evidence) > 0:
                evidence_golds.extend(gold_evidence)
                evidence_predicts.extend(predict_evidence)
            if len(predict_opinion) > 0:
                opinion_golds.extend(gold_opinion)
                opinion_predicts.extend(predict_opinion)

        loss = F.cross_entropy(classifier_logit, label)
        avg_loss += loss.item()
        corrects += (torch.max(classifier_logit, 1)
                     [1].view(label.size()).data == label.data).sum()
        preds_class.extend(list(torch.max(classifier_logit, 1)[1].data.cpu().numpy()))
        golds_class.extend(list(label.data.cpu().numpy()))
        text_list.extend(reverse_batch_text(feature, lengths, args.text_field.vocab.itos))
    size = len(data_iter.dataset)
    avg_loss /= size
    classifier_accuracy = 100.0 * corrects / size
    classifier_f1 = f1_score(y_true=golds_class, y_pred=preds_class, average="micro")
    evidence_sequence_f1 = f1_score(y_true=evidence_golds, y_pred=evidence_predicts, average="macro")
    opinion_sequence_f1 = f1_score(y_true=opinion_golds, y_pred=opinion_predicts, average="macro")
    evidence_target = [t for t in args.evidence_tag if t != "<pad>"]
    opinion_target = [t for t in args.opinion_tag if t != "<pad>"]
    logger.info('Evaluation - classifier_loss: {:.6f}  classifier_acc: {:.4f}%  classifier_f1: {:.4f} ({}/{})'
                .format(avg_loss,
                        classifier_accuracy,
                        classifier_f1,
                        corrects,
                        size))
    logger.info('Evaluation - [evidence]sequence_report:\n{}'
                .format(classification_report(y_true=evidence_golds, y_pred=evidence_predicts,
                                              target_names=evidence_target)))
    logger.info('Evaluation - [opinion]sequence_report:\n{}'
                .format(classification_report(y_true=opinion_golds, y_pred=opinion_predicts,
                                              target_names=opinion_target)))
    with codecs.open("result.txt", "w", "utf-8") as w:
        for index, text in enumerate(text_list):
            w.write("%s\t%s\t%s\n" % (args.label[golds_class[index]], args.label[preds_class[index]], "".join(text)))
        w.write("\n%s\n" % classification_report(y_true=golds_class, y_pred=preds_class, target_names=args.label))
        w.write(
            "\n%s\n" % classification_report(y_true=evidence_golds, y_pred=evidence_predicts,
                                             target_names=evidence_target))
        w.write("\n%s\n" % classification_report(y_true=opinion_golds, y_pred=opinion_predicts,
                                                 target_names=opinion_target))
    return classifier_f1


def save(model, args):
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, "joint_model.pkl")
    torch.save(model.state_dict(), save_path)
    args_path = os.path.join(args.save_dir, "joint_args.pkl")
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)


def reverse_batch_text(text_tensor, text_lengths, text_vocab):
    text_list = []
    text_index = text_tensor.data.cpu().numpy()
    text_lens = text_lengths.data.cpu().numpy()
    for (ix, l) in zip(text_index, text_lens):
        text = "".join([text_vocab[word_ix] for word_ix in ix][:l])
        text_list.append(text)
    return text_list


def decay_learning_rate(optimizer, lr_decay, epoch, init_lr):
    """衰减学习率

    Args:
        epoch: int, 迭代次数
        init_lr: 初始学习率
    """
    lr = init_lr / (1 + lr_decay * epoch)
    logger.info('learning rate: {%s}' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
