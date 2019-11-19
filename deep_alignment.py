#!/usr/bin/env python
# coding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split
from torch import autograd
from process_sentences import process

torch.manual_seed(1)

def get_precision_recall_fscore(truth, pred):
    precision, recall, fscore, support = precision_recall_fscore_support(truth, pred, average=None)
    return precision[1], recall[1], fscore[1]


# In[29]:


def prepare_sequence(seq, to_ix, cuda=False):
    seq = seq[0]
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var


def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var


def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix


def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, bert_weights):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        weight = torch.cuda.FloatTensor(bert_weights)
        weight = weight.to(torch.device('cuda:1'))
        self.word_embeddings= nn.Embedding.from_pretrained(weight)
        self.word_embeddings.weight.requires_grad=False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).to(torch.device('cuda:1'))),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).to(torch.device('cuda:1'))))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)


def train(train_data, dev_data, word_to_ix, label_to_ix, bert_weights):
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 768
    EPOCH = 50
    best_dev_f = 0.0
    best_dev_p = 0.0
    best_dev_r = 0.0
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix),bert_weights=bert_weights)
    model.to(torch.device('cuda:1'))

    w = [1.0, 1.0]
    class_weights = torch.FloatTensor(w).to(torch.device('cuda:1'))
    loss_function = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

    no_up = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        #print('now best dev acc:',best_dev_acc)
        print('best dev fscore:',best_dev_f,"precision",best_dev_p,"recall",best_dev_r)
        rec, prec, fscore, pred_res, pred_logs = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        if fscore > best_dev_f:
            best_dev_f = fscore
            best_dev_r = rec
            best_dev_p = prec
            #os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            with open("deep_results", "w+") as f:
                for j, p in enumerate(pred_res):
                    f.write(str(p[0]) + "\t" + str(pred_logs[j][0]) + "\t" + str(pred_logs[j][1]) + "\n")
            #torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(dev_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()


def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    pred_logs = []

    for sent, label in data:
        label = label.tolist()[0]
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, word_to_ix)
        label = prepare_label(label, label_to_ix)
        sent = sent.to(torch.device('cuda:1'))
        label = label.to(torch.device('cuda:1'))
        pred = model(sent)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_logs.append(pred.data.cpu().numpy().tolist()[0])
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    #acc = get_accuracy(truth_res, pred_res)
    precision, recall, fscore = get_precision_recall_fscore(truth_res, pred_res)
    #print(name + ' avg_loss:%g dev acc:%g' % (avg_loss, acc ))
    print(name + ' avg_loss:%g dev prec:%g dev rec:%g dev fscore:%g' % (avg_loss, precision, recall, fscore ))
    return precision, recall, fscore, pred_res, pred_logs


def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        label = label.tolist()[0]
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, word_to_ix)
        label = prepare_label(label, label_to_ix)
        sent = sent.to(torch.device('cuda:1'))
        label = label.to(torch.device('cuda:1'))
        pred = model(sent)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.item()))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    precision, recall, fscore = get_precision_recall_fscore(truth_res, pred_res)
    print('epoch: %d done! \n train avg_loss:%g , train prec:%g train rec:%g train fscore:%g'%(i, avg_loss, precision, recall, fscore))
    #print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))


def main():

    sentences, np_to_indices, word2idx, voc, voc_to_id, bert_weights = process()

    data_true = []
    labels_true = []
    data_false = []
    labels_false = []
    for sen in word2idx:
        for np_en in word2idx[sen]['sentence_en']:
            for np_hu in word2idx[sen]['sentence_hun']:
                if (str(np_en[0]), str(np_hu[0])) in sentences[sen]['aligns_filtered']:
                    data_true.append((" ".join(np_en[1]) + " " + " ".join(np_hu[1]), 1))
                    labels_true.append(1)
                else:
                    data_false.append((" ".join(np_en[1]) + " " + " ".join(np_hu[1]), 0))
                    labels_false.append(0)

    tag_to_ix = {0: 0, 1: 1}

    #random.shuffle(data_true)
    #random.shuffle(data_false)

    tr_data_true, tst_data_true = split(data_true, test_size=0.2, random_state=20)
    tr_data_false, tst_data_false= split(data_false, test_size=0.2, random_state=20)
    #tr_data_false = random.choices(tr_data_false, k=13946)
    tr_data = tr_data_true + tr_data_false
    tst_data = tst_data_true + tst_data_false

    batch_size = 1
    use_gpu = torch.cuda.is_available()
    learning_rate = 0.01
    from torch.utils.data import DataLoader, TensorDataset

    weights = []
    for i in tr_data:
        if i[1] == 1:
            weights.append(6.0)
        else:
            weights.append(1.0)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(tr_data,
                            batch_size=batch_size,
                            sampler=sampler,
                           # shuffle=True,
                            num_workers=4
                            )

    test_loader = DataLoader(tst_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4
                            )
    train(train_loader, test_loader, voc_to_id, tag_to_ix, bert_weights)


if __name__ == '__main__':
    main()
