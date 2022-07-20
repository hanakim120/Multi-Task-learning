# https://github.com/andy-yangz/nlp_multi_task_learning_pytorch.git
import argparse
import os
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import random
from data import Corpus
from util import *
from model import *
import os
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 2022 
torch.manual_seed(seed)
random.seed(2022)
wandb.init(project="level-test", name='lstm-level_pron-50ep', entity="hanadul")
###############################################################################
# Set Parameters
###############################################################################
parser = argparse.ArgumentParser(description='Pytorch NLP multi-task leraning for level test.')
parser.add_argument('--data', type=str, default='./data',
                    help='data file')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings') 
parser.add_argument('--nlevel_layers', type=int, default=1,
                    help='number of review level tagging layers')
parser.add_argument('--npron_layers', type=int, default=1,
                    help='number of pron layers')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip')
parser.add_argument('--epochs', type=int, default=40,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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



###############################################################################
# Load Data
###############################################################################
corpus_path = args.save.strip() + '_corpus.pt'
print('Loading corpus...')
if os.path.exists(corpus_path):
    corpus = torch.load(corpus_path)
else:
    corpus = Corpus(args.data)
    #torch.save(corpus, corpus_path)

#print(dir(corpus),'############corpus###########')
#torch.save(corpus, corpus_path)

###############################################################################
# Training Funcitons
###############################################################################
def train(loss_log):
    model.train()
    if args.train_mode == 'Joint':
        target_data = (corpus.level_train, corpus.pron_train, corpus.gram_train, corpus.vocab_train, corpus.compre_train, corpus.confi_train)
    elif args.train_mode == 'Level':
        target_data = (corpus.level_train, )
    elif args.train_mode == 'Pron':
        target_data = (corpus.pron_train, )

    # Turn on training mode
    total_loss = 0
    start_time = time.time()
    n_iteration = corpus.word_train.size(0) // (args.batch_size*args.seq_len) 
    iteration = 0
    for X, ys in get_batch(corpus.word_train, *target_data, batch_size=args.batch_size,
                           seq_len=args.seq_len, cuda=args.cuda):
        iteration += 1
        model.zero_grad()
        
        if args.train_mode == 'Joint':
            if args.nlevel_layers == args.npron_layers:
                hidden = model.rnn.init_hidden(args.batch_size)
                outputs1, outputs2, hidden = model(X, hidden)
            else:
                hidden1 = model.rnn1.init_hidden(args.batch_size)
                hidden2 = model.init_rnn2_hidden(args.batch_size)
                outputs1, outputs2, hidden1, hidden2 = model(X, hidden1, hidden2)    
            loss1 = criterion(outputs1.view(-1, nlevel_tags), ys[0].view(-1))
            loss2 = criterion(outputs2.view(-1, npron_tags), ys[1].view(-1))
            loss3 = criterion(outputs3.view(-1, ngram_tags), ys[2].view(-1))
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

            wandb.log({"train_loss_epoch": cur_loss})

            print('| epoch {:3d} | {:5d}/{:5d} iteration | {:5.2f} ms/batch | loss {:5.2f} |'.format(
                epoch, iteration, n_iteration,
                elapsed*1000/args.log_interval, 
                cur_loss))
                
            loss_log.append(cur_loss)
            total_loss = 0
            start_time = time.time()
    return loss_log

def evaluate(source, target):
    model.eval()
    n_iteration = source.size(0) // (args.batch_size*args.seq_len)
    
    total_loss = 0
    for X_val, y_vals in get_batch(source, *target, batch_size=args.batch_size,
                           seq_len=args.seq_len, cuda=args.cuda, evalu=True):
        
        if args.train_mode == 'Joint':
            if args.nlevel_layers == args.npron_layers:
                hidden = model.rnn.init_hidden(args.batch_size)
                outputs1, outputs2, hidden = model(X_val, hidden)
            else:
                hidden1 = model.rnn1.init_hidden(args.batch_size)
                hidden2 = model.init_rnn2_hidden(args.batch_size)
                outputs1, outputs2, hidden1, hidden2 = model(X_val, hidden1, hidden2)    
            loss1 = criterion(outputs1.view(-1, nlevel_tags), y_vals[0].view(-1))
            loss2 = criterion(outputs2.view(-1, npron_tags), y_vals[1].view(-1))
            loss = loss1 + loss2
            
            # Make predict and calculate accuracy
            _, pred1 = outputs1.data.topk(1)
            _, pred2 = outputs2.data.topk(1)
            accuracy1 = torch.sum(pred1.squeeze(2) == y_vals[0].data) / (y_vals[0].size(0) * y_vals[0].size(1))
            accuracy2 = torch.sum(pred2.squeeze(2) == y_vals[1].data) / (y_vals[1].size(0) * y_vals[1].size(1))
            accuracy = (accuracy1, accuracy2)
            
        else:
            
            hidden = model.rnn.init_hidden(args.batch_size)
            outputs, hidden = model(X_val, hidden)
            loss = criterion(outputs.view(-1, ntags), y_vals[0].view(-1))
            _, pred = outputs.data.topk(1)
            accuracy = torch.sum(pred.squeeze(2) == y_vals[0].data) / (y_vals[0].size(0) * y_vals[0].size(1))
        total_loss += loss
    
    return total_loss/n_iteration, accuracy

best_val_accuracies = []
test_accuracies = []
best_epoches = []
patience = 25 #How many epoch if the accuracy have no change use early stopping
for i in range(args.test_times):
###############################################################################
# Build Model
###############################################################################
    nwords = corpus.word_dict.nwords
    nlevel_tags = corpus.level_dict.nwords
    npron_tags = corpus.pron_dict.nwords
    
    if args.train_mode == 'Joint':
        
        model = JointModel(nwords, args.emsize, args.nhid, nlevel_tags, args.nlevel_layers,
                           npron_tags, args.npron_layers, args.dropout, bi=args.bi, 
                           train_mode=args.train_mode)

        
    else:
        if args.train_mode == 'Level':
            ntags = nlevel_tags
            nlayers = args.nlevel_layers
        elif args.train_mode == 'Pron':
            ntags = npron_tags
            nlayers = args.npron_layers
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
        
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            print('Begin training...')
            loss_log = train(loss_log)
            # Evaluation
            print('Evaluating on the valid data')
            if args.train_mode == 'Joint':
                valid_target_data = (corpus.level_valid, corpus.pron_valid)
            elif args.train_mode == 'Level':
                valid_target_data = (corpus.level_valid, ) 
            elif args.train_mode == 'Pron':
                valid_target_data = (corpus.pron_valid, ) 

            
            val_loss, accuracy = evaluate(corpus.word_valid, valid_target_data)
            print('-'*100)
            
            if args.train_mode == 'Joint':
                print('| end of epoch {:3d} | valid loss {:5.3f} | Level accuracy {:5.3f} | Pron accuracy {:5.3}'.format(
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
                with open(args.save.strip() + '_model.pt', 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss.data
                best_accuracy = accuracy
                best_epoch = epoch
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count >= patience:
                print('\nEarly Stopping! \nBecause %d epochs the accuracy have no improvement.'%(patience))
                break
    except KeyboardInterrupt:
        print('-'*50)
        print('Exiting from training early.')


###############################################################################
# Test Model
###############################################################################
    #Load the best saved model
    with open(args.save.strip() + '_model.pt', 'rb') as f:
        model = torch.load(f)

    if args.train_mode == 'Joint':
        test_target_data = (corpus.level_test, corpus.pron_test)
    elif args.train_mode == 'Level':
        test_target_data = (corpus.level_test, ) 
    elif args.train_mode == 'Pron':
        test_target_data = (corpus.pron_test, ) 
    test_loss, test_accuracy = evaluate(corpus.word_test, test_target_data)
    print('='*100)
    print("Evaluating on test data.")
    if args.train_mode == 'Joint':
        print('| end of epoch {:3d} | test loss {:5.3f} | Level test accuracy {:5.3f} | Pron test accuracy {:5.3}'.format(
            epoch, test_loss.data.cpu().numpy(), test_accuracy[0], test_accuracy[1]
        ))
    else:
        print('| end of epoch {:3d} | test loss {:5.3f} | accuracy {:5.3f} |'.format(
            epoch, test_loss.data.cpu().numpy(), test_accuracy
        ))
    
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

torch.save(results, '%s_emsize%d_npos_layers%d_nchunk_layers%d_nhid%d_dropout%3.1f_seqlen%d_bi%d_%s_result.pt' \
                    %(args.save.strip(), args.emsize, args.nlevel_layers, args.npron_layers,
                      args.nhid, args.dropout, args.seq_len, args.bi, args.rnn_type))

            
