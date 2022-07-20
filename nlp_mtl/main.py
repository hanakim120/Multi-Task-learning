# https://github.com/andy-yangz/nlp_multi_task_learning_pytorch.git
import argparse
import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import optim
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import random
from data import Corpus
from util import *
from model import *
import os
import itertools
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set Seed
seed = 2022
torch.manual_seed(seed)
random.seed(2022)

# wandb init
wandb.init(project="220713-lstm-base", name='lstm-level-ros', entity="hanadul")

# Set Parameters
parser = argparse.ArgumentParser(description='Pytorch NLP multi-task leraning for level test.')
parser.add_argument('--data', type=str, default='./data',
                    help='data file location')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nlevel_layers', type=int, default=1,
                    help='number of review level tagging layers')
parser.add_argument('--nconfi_layers', type=int, default=0,
                    help='number of none layers')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip')
parser.add_argument('--epochs', type=int, default=40,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=15,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout rate')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='RNN Cell types, among LSTM, GRU, and Elman')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--bi', action='store_true',
                    help='use bidirection RNN')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--train_mode', type=str, default='Joint',
                    help='Training mode of model.')
parser.add_argument('--test_times', type=int, default=1,
                    help='run several times to get trustable result.')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Load Data
corpus_path = args.save.strip() + '_corpus.pt'
print('Loading corpus...')
if os.path.exists(corpus_path):
    corpus = torch.load(corpus_path)
else:
    corpus = Corpus(args.data)
    # torch.save(corpus, corpus_path)


# print(dir(corpus),'############corpus###########')
# torch.save(corpus, corpus_path)


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.gcf().subplots_adjust(bottom=0.10)
    name_count = 0

    plt.savefig('./confusion/%s_nlayers1%d_nlayers2%d.png' % (args.train_mode,args.nlevel_layers, args.nconfi_layers))


def plot_confusion_matrix_1(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    plt.gcf().subplots_adjust(bottom=0.10)
    name_count = 0

    plt.savefig(
        './confusion/%s_level_nlayers1%d_nlayers2%d.png' % (args.train_mode, args.nlevel_layers, args.nconfi_layers))


def plot_confusion_matrix_2(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    plt.gcf().subplots_adjust(bottom=0.10)
    name_count = 0

    plt.savefig(
        './confusion/%s_2nd_layer_nlayers1%d_nlayers2%d.png' % (args.train_mode, args.nlevel_layers, args.nconfi_layers))
###############################################################################
# Training Funcitons
###############################################################################
def train(loss_log):
    # Select train mode
    if args.train_mode == 'Joint':
        target_data = (corpus.level_train, corpus.confi_train)
    elif args.train_mode == 'level':
        target_data = (corpus.level_train,) # (tensor([0, 1, 2,  ..., 0, 0, 0]),)
    elif args.train_mode == 'none':
        target_data = (corpus.none_train,)

    # Turn on training mode
    model.train()

    total_loss = 0
    start_time = time.time()
    n_iteration = corpus.word_train.size(0) // (args.batch_size * args.seq_len)
    iteration = 0
    for X, ys in get_batch(corpus.word_train, *target_data, batch_size=args.batch_size,
                           seq_len=args.seq_len, cuda=args.cuda):
        iteration += 1
        model.zero_grad()

        if args.train_mode == 'Joint':
            if args.nlevel_layers == args.nconfi_layers:
                hidden = model.rnn.init_hidden(args.batch_size)
                outputs1, outputs2, hidden = model(X, hidden)
            else:
                hidden1 = model.rnn1.init_hidden(args.batch_size)
                hidden2 = model.init_rnn2_hidden(args.batch_size)
                outputs1, outputs2, hidden1, hidden2 = model(X, hidden1, hidden2)
            loss1 = criterion(outputs1.view(-1, nlevel_tags), ys[0].view(-1))
            loss2 = criterion(outputs2.view(-1, nconfi_tags), ys[1].view(-1))
            loss = loss1 + loss2
        else:
            hidden = model.rnn.init_hidden(args.batch_size)
            outputs, hidden = model(X, hidden)
            loss = criterion(outputs.view(-1, ntags), ys[0].view(-1))

        loss.backward()

        # Prevent the exploding gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data

        if iteration % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            cur_loss = cur_loss.cpu().numpy()
            elapsed = time.time() - start_time

            wandb.log({"train_loss": cur_loss})

            print('| epoch {:3d} | {:5d}/{:5d} iteration | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                epoch, iteration, n_iteration,
                elapsed * 1000 / args.log_interval,
                cur_loss))

            loss_log.append(cur_loss)
            total_loss = 0
            start_time = time.time()
    return loss_log


def evaluate(source, target):
    model.eval()
    n_iteration = source.size(0) // (args.batch_size * args.seq_len)

    total_loss = 0
    for X_val, y_vals in get_batch(source, *target, batch_size=args.batch_size,
                                   seq_len=args.seq_len, cuda=args.cuda, evalu=True):

        if args.train_mode == 'Joint':
            if args.nlevel_layers == args.nconfi_layers:
                hidden = model.rnn.init_hidden(args.batch_size)
                outputs1, outputs2, hidden = model(X_val, hidden)
            else:
                hidden1 = model.rnn1.init_hidden(args.batch_size)
                hidden2 = model.init_rnn2_hidden(args.batch_size)
                outputs1, outputs2, hidden1, hidden2 = model(X_val, hidden1, hidden2)
            loss1 = criterion(outputs1.view(-1, nlevel_tags), y_vals[0].view(-1))
            loss2 = criterion(outputs2.view(-1, nconfi_tags), y_vals[1].view(-1))
            loss = loss1 + loss2

            # Make predict and calculate accuracy
            _, pred1 = outputs1.data.topk(1)
            _, pred2 = outputs2.data.topk(1)
            accuracy1 = torch.sum(pred1.squeeze(2) == y_vals[0].data) / (y_vals[0].size(0) * y_vals[0].size(1))
            accuracy2 = torch.sum(pred2.squeeze(2) == y_vals[1].data) / (y_vals[1].size(0) * y_vals[1].size(1))
            accuracy = (accuracy1, accuracy2)

            y_true_1 = y_vals[0].data.flatten()
            y_pred_1 = pred1.squeeze(2).flatten()
            # print(y_true_1, "y_true_1")
            # print(y_pred_1, "y_pred_1")

            y_true_2 = y_vals[1].data.flatten()
            y_pred_2 = pred2.squeeze(2).flatten()

            global confusion_1
            global confusion_2

            confusion_1 = confusion_matrix(y_true_1, y_pred_1)
            con_accuracy_1 = accuracy_score(y_true_1, y_pred_1)
            con_precision_1 = precision_score(y_true_1, y_pred_1, average='weighted')
            con_recall_1 = recall_score(y_true_1, y_pred_1, average='weighted')
            print('confusion matrix_1')
            print(confusion_1)
            print('accuracy:{}, precision:{}, recall:{}'.format(con_accuracy_1, con_precision_1, con_recall_1))
            print('f1 score : ', f1_score(y_true_1, y_pred_1, average='weighted'))

            confusion_2 = confusion_matrix(y_true_2, y_pred_2)
            con_accuracy_2 = accuracy_score(y_true_2, y_pred_2)
            con_precision_2 = precision_score(y_true_2, y_pred_2, average='weighted')
            con_recall_2 = recall_score(y_true_2, y_pred_2, average='weighted')
            print('confusion matrix_2')
            print(confusion_2)
            print('accuracy:{}, precision:{}, recall:{}'.format(con_accuracy_2, con_precision_2, con_recall_2))
            print('f1 score : ', f1_score(y_true_2, y_pred_2, average='weighted'))


        else:

            hidden = model.rnn.init_hidden(args.batch_size)
            outputs, hidden = model(X_val, hidden)
            loss = criterion(outputs.view(-1, ntags), y_vals[0].view(-1))
            _, pred = outputs.data.topk(1)

            accuracy = torch.sum(pred.squeeze(2) == y_vals[0].data) / (y_vals[0].size(0) * y_vals[0].size(1))

            y_true = y_vals[0].data.flatten()
            y_pred = pred.squeeze(2).flatten()
            print(y_true,"y_true")
            print(y_pred, "y_pred")
            global confusion

            confusion = confusion_matrix(y_true, y_pred)
            con_accuracy = accuracy_score(y_true, y_pred)
            con_precision = precision_score(y_true, y_pred, average='weighted')
            con_recall = recall_score(y_true, y_pred, average='weighted')
            print('confusion matrix')
            print(confusion)
            print('accuracy:{}, precision:{}, recall:{}'.format(con_accuracy, con_precision, con_recall))
            print('f1 score : ', f1_score(y_true, y_pred, average='weighted'))

        total_loss += loss

    return total_loss / n_iteration, accuracy


best_val_accuracies = []
test_accuracies = []
best_epoches = []
patience = 25  # How many epoch if the accuracy have no change use early stopping
for i in range(args.test_times):
    ###############################################################################
    # Build Model
    ###############################################################################
    nwords = corpus.word_dict.nwords
    nlevel_tags = corpus.level_dict.nwords
    nconfi_tags = corpus.confi_dict.nwords

    if args.train_mode == 'Joint':

        model = JointModel(nwords, args.emsize, args.nhid, nlevel_tags, args.nlevel_layers,
                           nconfi_tags, args.nconfi_layers, args.dropout, bi=args.bi,
                           train_mode=args.train_mode)


    else:
        if args.train_mode == 'level':
            ntags = nlevel_tags
            nlayers = args.nlevel_layers
        elif args.train_mode == 'none':
            ntags = nnone_tags
            nlayers = args.nnone_layers
        model = JointModel(nwords, args.emsize, args.nhid, ntags, nlayers,
                           args.dropout, bi=args.bi, train_mode=args.train_mode)

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Loop over epochs
    best_val_loss = None
    best_accuracy = None
    best_epoch = 0
    early_stop_count = 0
    loss_log = []
    # You can break training early by Ctr+C

    try:

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            print('Begin training...')
            loss_log = train(loss_log)
            # Evaluation
            print('Evaluating on the valid data')
            if args.train_mode == 'Joint':
                valid_target_data = (corpus.level_valid, corpus.confi_valid)
            elif args.train_mode == 'level':
                valid_target_data = (corpus.level_valid,)
            elif args.train_mode == 'none':
                valid_target_data = (corpus.none_valid,)

            val_loss, accuracy = evaluate(corpus.word_valid, valid_target_data)
            print('-' * 100)

            if args.train_mode == 'Joint':
                print(
                    '| end of epoch {:3d} | valid loss {:5.3f} | level accuracy {:5.3f} | confi accuracy {:5.3}'.format(
                        epoch, val_loss.data.cpu().numpy(), accuracy[0], accuracy[1]
                    ))
                wandb.log({"val_loss": val_loss.data.cpu().numpy()})
                wandb.log({"acc_0": accuracy[0]})
                wandb.log({"acc_1": accuracy[1]})
            else:
                print('| end of epoch {:3d} | valid loss {:5.3f} | accuracy {:5.3f} |'.format(
                    epoch, val_loss.data.cpu().numpy(), accuracy
                ))
                wandb.log({"val_loss": val_loss.data.cpu().numpy()})
                wandb.log({"val_acc": accuracy})

            if not best_val_loss or (val_loss.data < best_val_loss):
                with open(args.save.strip() + '%d_nlayers1_%d_nlayers2_model.pt'%(args.nlevel_layers, args.nconfi_layers), 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss.data
                best_accuracy = accuracy
                best_epoch = epoch
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count >= patience:
                print('\nEarly Stopping! \nBecause %d epochs the accuracy have no improvement.' % (patience))
                break
    except KeyboardInterrupt:
        print('-' * 50)
        print('Exiting from training early.')

    ###############################################################################
    # Test Model
    ###############################################################################
    # Load the best saved model

    with open(args.save.strip() + '%d_nlayers1_%d_nlayers2_model.pt'%(args.nlevel_layers, args.nconfi_layers), 'rb') as f:
        print(f)
        model = torch.load(f)

    if args.train_mode == 'Joint':
        test_target_data = (corpus.level_test, corpus.confi_test)
    elif args.train_mode == 'level':
        test_target_data = (corpus.level_test,)
    elif args.train_mode == 'none':
        test_target_data = (corpus.none_test,)
    test_loss, test_accuracy = evaluate(corpus.word_test, test_target_data)
    print('=' * 100)
    print("Evaluating on test data.")
    if args.train_mode == 'Joint':
        print(
            '| end of epoch {:3d} | test loss {:5.3f} | level test accuracy {:5.3f} | confi test accuracy {:5.3}'.format(
                epoch, test_loss.data.cpu().numpy(), test_accuracy[0], test_accuracy[1]
            ))
        plot_confusion_matrix_1(confusion_1, cmap=None, normalize=False,
                              labels=True, title='Confusion matrix')

        plot_confusion_matrix_2(confusion_2, cmap=None, normalize=False,
                              labels=True, title='Confusion matrix')
    else:
        print('| end of epoch {:3d} | test loss {:5.3f} | accuracy {:5.3f} |'.format(
            epoch, test_loss.data.cpu().numpy(), test_accuracy
        ))
        plot_confusion_matrix(confusion, cmap=None, normalize=False,
                              labels=True, title='Confusion matrix')

    # Log Accuracy
    best_val_accuracies.append(best_accuracy)
    test_accuracies.append(test_accuracy)
    best_epoches.append(best_epoch)

# Save results
results = {
    'corpus': corpus,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'best_accuracy': best_accuracy,
    'test_accuracy': test_accuracy,
    'loss_log': loss_log,

    'best_val_accuracies': best_val_accuracies,
    'test_accuracies': test_accuracies,
    'best_epoches': best_epoches
}

torch.save(results, '%s_emsize%d_nlayers1%d_n_layers2%d_nhid%d_dropout%3.1f_seqlen%d_bi%d_%s_result.pt' \
                    %(args.save.strip(), args.emsize, args.nlevel_layers, args.nconfi_layers,
                      args.nhid, args.dropout, args.seq_len, args.bi, args.rnn_type))


