from __future__ import division
# import sys
# import os
# sys.path.append(os.path.abspath(__file__ + "/../../"))
import torch
from torch import optim
import torch.nn as nn
import random
import time
from QA_model import *
from util import *

# for doing stuff in console
# import os
# os.chdir('/home/jack/Documents/QA_QG/QA_dist_msr/src')
# import src.QA_model
# imp.reload(src.QA_model)
# import src.util
# imp.reload(src.util)
# from src.QA_model import *
# from src.util import *

# experiment result path
result_path = '../../results/QA_dist_model_1003/'

# read embedding
path_to_glove = '../../data/glove.6B/glove.6B.100d.txt'
embeddings_index, embeddings_size = readGlove(path_to_glove)

# read and prepare input data
cq_path = '../../data/squad_openNMT/train/cq_sent_concat_min.txt'
a_path = '../../data/squad_openNMT/train/as_min_NoAnnotate.txt'
cq, a, cq_data, a_data, tokens = prepareInput(src_file=cq_path, tgt_file=a_path, embeddings_index=embeddings_index,
                                              embeddings_size=embeddings_size)
print('read ' + str(len(cq)) + ' context+question, and ' + str(len(a)) + ' answers, for test.')
print('type of element in cq: ' + str(type(cq_data[0])) + ' , and type inside it is: ' + str(type(cq_data[0].data)))
print('type of element in a:  ' + str(type(a_data[0]))  + ' , and type inside it is: ' + str(type(a_data[0].data)))
# find effective tokens
effective_tokens, effective_num_tokens = count_effective_num_tokens(tokens, embeddings_index)

# read and prepare validation data
test_cq_path = '../../data/squad_openNMT/test/test_cq_sent_concat_min.txt'
test_a_path = '../../data/squad_openNMT/test/test_as_min_NoAnnotate.txt'
test_cq, test_a, test_cq_data, test_a_data, test_tokens = prepareInput(src_file=test_cq_path, tgt_file=test_a_path,
                                                                       embeddings_index=embeddings_index,
                                                                       embeddings_size=embeddings_size)
print('read ' + str(len(test_cq)) + ' context+question, and ' + str(len(test_a)) + ' answers, for validation.')
print('type of element in test_cq: ' + str(type(test_cq_data[0])) + ' , and type inside it is: ' + str(type(test_cq_data[0].data)))
print('type of element in test_a:  ' + str(type(test_a_data[0]))  + ' , and type inside it is: ' + str(type(test_a_data[0].data)))
# find effective tokens in validation set
test_effective_tokens, test_effective_num_tokens = count_effective_num_tokens(test_tokens, embeddings_index)

effective_tokens = list(set(test_effective_tokens + effective_tokens))
effective_num_tokens = len(effective_tokens)

# set up model
enc_hidden_size = 200
enc_num_layers = 2
enc_bidirectional = 2
encoder = nn.LSTM(input_size=embeddings_size, hidden_size=enc_hidden_size,
                  num_layers=enc_num_layers, bidirectional=enc_bidirectional)
decoder = distDecoder(input_size=enc_hidden_size, output_size=embeddings_size)
qa = QA(encoder=encoder, decoder=decoder).cuda()

# set up optimnizer and loss
lr = 1
optimizer = optim.SGD(qa.parameters(), lr=lr)
crit = nn.MSELoss()

# start training
record_loss_every = 100
num_epoch=15
loss_vec = []
print()
print('input size: '+str(cq_data[0].size()))
print()
print('start training...')
for epoch in range(1,num_epoch+1):

    # set to train mode
    qa.train()
    begin_time = time.time()
    print('training epoch at ' + str(epoch))

    for iter in range(len(cq)):

        # get a single example
        train_cq = cq[iter]
        train_a = a[iter]
        train_cq_data = cq_data[iter]
        train_a_data = a_data[iter]
        ans1stWord = train_a.lower().split(" ")[0] if train_a.lower().split(" ")[0] in effective_tokens else 'UNK'

        # forward pass to get the last hidden state of encoder
        # enc_out size = (num_layers * num_directions, batch, hidden_size)
        enc_out = qa.forward(train_cq_data)[0] # the first one is the hidden state, if using LSTM

        # process the hidden state to be (1, hidden_size)
        dec_in = enc_out[-1] + enc_out[-2]

        # calculate loss
        optimizer.zero_grad()
        # print(dec_in.size())
        # print(qa.decoder.out(dec_in).size())
        # print(embeddings_index[train_a.lower().split(" ")[0]].unsqueeze(0).size())
        loss = qa.decoder.calculateLoss(crit, dec_in, ans1stWord, embeddings_index)

        # record loss
        if iter != 0 and iter % record_loss_every == 0:
            loss_vec.append(loss)
        # print at the last iteration in each epoch
        if iter == len(cq) - 1:
            print('training loss at last iteration of epoch ' + str(epoch) + ' is: ' + str(loss))
            print('%s (%d %d%%) %.4f' % (timeSince(begin_time, epoch/float(num_epoch)), iter, epoch/num_epoch*100))
        # print progress once in a while
        if iter % 5000 == 0:
            print('   in iteration %d, all goes well.' % iter)
        # update
        loss.backward()
        optimizer.step()

    # perform evaluation (average loss) on the validation set every epoch
    qa.eval() # set to evaluation mode
    validate_loss = 0.0
    true_ans1stWords = []
    gen_ans1stWords = []
    for i in range(len(test_cq)):
        # prepare data - split, replace words not in glove with UNK
        testAns1stWord = test_a[i].lower().split(" ")[0] if test_a[i].lower().split(" ")[0] in effective_tokens else 'UNK'
        test_cq_words = test_cq[i].lower().split(" ")
        for w in range(len(test_cq_words)):
            if test_cq_words[w] not in effective_tokens:
                test_cq_words[w] = 'UNK'
        test_enc_out = qa.forward(test_cq_data[i])[0]
        test_dec_in = test_enc_out[-1] + test_enc_out[-2]
        # accumulate loss
        validate_loss += qa.decoder.calculateLoss(crit, test_dec_in, testAns1stWord, embeddings_index)
        true_ans1stWords.append(test_a[i].lower().split(" ")[0])
        gen_ans1stWords.append(qa.decoder.generateWord(crit, test_dec_in, test_cq_words, embeddings_index))
    validate_loss = validate_loss / len(test_cq)
    print('average loss over validation set is: ' + str(validate_loss))
    # sample true answer first word and generated answer first word
    sample_idx = random.sample(range(len(test_cq)), 20)
    sampled_true_ans1stWords = [true_ans1stWords[i] for i in sample_idx]
    sampled_gen_ans1stWords = [gen_ans1stWords[i] for i in sample_idx]

    # save a checkpoint model for every epoch
    checkpoint = {
        'model': qa.state_dict(),
        'encoder': qa.encoder.state_dict(),
        'generator': qa.decoder.state_dict(),
        'epoch': epoch,
        'optim': optimizer
    }
    torch.save(checkpoint, result_path+'%s_loss_%.2f_e%d.pt' % ('QA_dist_model', validate_loss, epoch))

