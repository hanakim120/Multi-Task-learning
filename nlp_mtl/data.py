import os
import torch 


class Dictionary(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.nwords
            self.idx2word.append(word)
            self.nwords += 1

    def __str__(self):
        return "%s dictionary has %d kinds of tokens." \
                % (self.name, self.nwords)


class Corpus(object):
    def __init__(self, path):
        self.word_dict = Dictionary('word')
        self.level_dict = Dictionary('level')
        self.pron_dict = Dictionary('pron')
        self.gram_dict = Dictionary('gram')
        self.vocab_dict = Dictionary('vocab')
        self.compre_dict = Dictionary('compre')
        self.confi_dict = Dictionary('confi')
        
        self.word_train, self.level_train, self.pron_train, self.gram_train, self.vocab_train, self.compre_train, self.confi_train = self.tokenize(os.path.join(path, 'train.txt'))
        self.word_valid, self.level_valid, self.pron_valid, self.gram_valid, self.vocab_valid, self.compre_valid, self.confi_valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.word_test, self.level_test, self.pron_test, self.gram_test, self.vocab_test, self.compre_test, self.confi_test = self.tokenize(os.path.join(path, 'test.txt'))
        # print(os.path.join(path, 'test.txt'), '#####path#####')
        # print(self.word_train,'########word_train')
        # print(self.level_train, '########level_train')
        # print(self.pron_train, '########pron_train')


    def tokenize(self, path):
        "Tokenizes text data file"
        assert os.path.exists(path)
        # Build the dictionaries from corpus
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                try: 
                    word, level, pron, gram, vocab, compre, confi = line.strip().split('|')
                    #print(word,level,pron,gram,vocab,compre,confi,'!!!!!!!!!!!!!!!!!!!!!!!!!!')
                except: 
                    continue
                tokens += 1
                self.word_dict.add_word(word)
                self.level_dict.add_word(level)
                self.pron_dict.add_word(pron)
                self.gram_dict.add_word(gram)
                self.vocab_dict.add_word(vocab)
                self.compre_dict.add_word(compre)
                self.confi_dict.add_word(confi)
               

        with open(path, 'r') as f:
            word_ids = torch.LongTensor(tokens)
            level_ids = torch.LongTensor(tokens)
            pron_ids = torch.LongTensor(tokens)
            gram_ids = torch.LongTensor(tokens)
            vocab_ids = torch.LongTensor(tokens)
            compre_ids = torch.LongTensor(tokens)
            confi_ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                try: 
                    word, level, pron, gram, vocab, compre, confi = line.strip().split('|')
                except: 
                    continue
                word_ids[token] = self.word_dict.word2idx[word]
                level_ids[token] = self.level_dict.word2idx[level]
                pron_ids[token] = self.pron_dict.word2idx[pron]
                gram_ids[token] = self.gram_dict.word2idx[gram]
                vocab_ids[token] = self.vocab_dict.word2idx[vocab]
                compre_ids[token] = self.compre_dict.word2idx[compre]
                confi_ids[token] = self.confi_dict.word2idx[confi]
                token += 1
            #print(word_ids,level_ids, pron_ids, gram_ids, vocab_ids, compre_ids, confi_ids)
        return word_ids, level_ids, pron_ids, gram_ids, vocab_ids, compre_ids, confi_ids
