import torch
from torch.autograd import Variable
import random
import numpy as np
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def readLinesFromFile(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.rstrip('\n') for x in content]
    f.close()
    return content


# function to prepare input
# returns a list of torch autograd variables, each variable is a tensor of the appropriate sized input to nn.rnn
def prepareInput(src_file, tgt_file, embeddings_index, embeddings_size):

    src = readLinesFromFile(src_file)
    tgt = readLinesFromFile(tgt_file)
    src_data = []
    tgt_data = []
    data_tokens = []

    for i in range(len(src)):

        src_w = src[i].lower().split(" ") # list of words, make lower case
        src_s = torch.FloatTensor(len(src_w), 1, embeddings_size).cuda() # empty tensor placeholder
        for w in range(len(src_w)):
            try:
                src_s[w] = embeddings_index[src_w[w]]  # move data to GPU and wrap it in Variable
            except: # when the token is not found
                src_s[w] = embeddings_index['UNK']
        src_data.append(Variable(src_s))

        tgt_w = tgt[i].lower().split(" ") # list of words
        tgt_s = torch.FloatTensor(len(tgt_w), 1, embeddings_size).cuda() # 1 in the middle because now batch size is 1
        for w in range(len(tgt_w)):
            try:
                tgt_s[w] = embeddings_index[tgt_w[w]]
            except:
                tgt_s[w] = embeddings_index['UNK']
        tgt_data.append(Variable(tgt_s))

        data_tokens += src_w + tgt_w
        data_tokens = list(set(data_tokens))

    return src, tgt, src_data, tgt_data, data_tokens
# test
# path_to_glove = '/home/jack/Documents/QA_QG/data/glove.6B/glove.6B.100d.txt'
# embeddings_index, embeddings_size = readGlove(path_to_glove)
# # read and prepare input data
# cq_path = '/home/jack/Documents/QA_QG/data/squad_openNMT/train/cq_sent_concat_min.txt'
# a_path = '/home/jack/Documents/QA_QG/data/squad_openNMT/train/as_min_NoAnnotate.txt'
# cq, a, cq_data, a_data, tokens = prepareInput(src_file=cq_path, tgt_file=a_path, embeddings_index=embeddings_index,
#                                               embeddings_size=embeddings_size)
# print('read ' + str(len(cq)) + ' context+question, and ' + str(len(a)) + ' answers.')
# print('type of element in cq: ' + str(type(cq_data[0])) + ' , and type inside it is: ' + str(type(cq_data[0].data)))
# print('type of element in a:  ' + str(type(a_data[0]))  + ' , and type inside it is: ' + str(type(a_data[0].data)))


def prepareBatchInput(qc_file, a_file):
    return None


# read the glove pretrained embeddings
def readGlove(path_to_data):
    embeddings_index = {}
    f = open(path_to_data)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        coefs = torch.from_numpy(coefs)
        embeddings_index[word] = coefs.cuda()
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # get dimension from a random sample in the dict
    embeddings_size = random.sample( embeddings_index.items(), 1 )[0][1].size(-1)
    print('dimension of word embeddings: ' + str(embeddings_size))

    # a few definitions of special tokens
    SOS_token = -torch.ones(embeddings_size).cuda() # start of sentence token, all zerons
    EOS_token = torch.ones(embeddings_size).cuda() # end of sentence token, all ones
    UNK_token = torch.ones(embeddings_size).cuda() + torch.ones(embeddings_size).cuda() # these choices are pretty random
    PAD_token = torch.zeros(embeddings_size).cuda()

    # add special tokens to the embeddings
    embeddings_index['SOS'] = SOS_token
    embeddings_index['EOS'] = EOS_token
    embeddings_index['UNK'] = UNK_token
    embeddings_index['PAD'] = PAD_token

    return embeddings_index, embeddings_size


# generate word index and index word look up tables
def generate_look_up_table(effective_tokens, effective_num_tokens, use_cuda = True):
    word2index = {}
    index2word = {}
    for i in range(effective_num_tokens):
        index2word[i] = effective_tokens[i]
        word2index[effective_tokens[i]] = i
    return word2index, index2word


# count the number of tokens in both the word embeddings and the corpus
def count_effective_num_tokens(data_tokens, embeddings_index, sos_eos = True):
    ## find all unique tokens in the data (should be a subset of the number of embeddings)
    data_tokens = list(set(data_tokens)) # find unique
    if sos_eos:
        data_tokens = ['SOS', 'EOS', 'UNK', 'PAD'] + data_tokens
    else:
        data_tokens = ['UNK', 'PAD']

    effective_tokens = list(set(data_tokens).intersection(embeddings_index.keys()))
    effective_num_tokens = len(effective_tokens)

    return effective_tokens, effective_num_tokens