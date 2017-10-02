import torch
from torch.autograd import Variable
import random

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

        src_w = src[i].split(" ") # list of words
        src_s = torch.FloatTensor(len(src_w), 1, embeddings_size) # empty tensor placeholder
        for w in range(len(src_w)):
            src_s[w] = Variable(embeddings_index[src_w[w]].cuda())  # move data to GPU and wrap it in Variable
        src_data.append(src_s)

        tgt_w = tgt[i].split(" ") # list of words
        tgt_s = torch.FloatTensor(len(tgt_w), 1, embeddings_size)
        for w in range(len(tgt_w)):
            tgt_s[w] = Variable(embeddings_index[tgt_w[w]].cuda())
        tgt_data.append(tgt_s)

        data_tokens += src_w + tgt_w
        data_tokens = list(set(data_tokens))

    return src, tgt, src_data, tgt_data, data_tokens


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
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # get dimension from a random sample in the dict
    embeddings_size = random.sample( embeddings_index.items(), 1 )[0][1].size(-1)
    print('dimension of word embeddings: ' + str(embeddings_size))

    # a few definitions of special tokens
    SOS_token = -torch.ones(embeddings_size) # start of sentence token, all zerons
    EOS_token = torch.ones(embeddings_size) # end of sentence token, all ones
    UNK_token = torch.ones(embeddings_size) + torch.ones(embeddings_size) # these choices are pretty random
    PAD_token = torch.zeros(embeddings_size)

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