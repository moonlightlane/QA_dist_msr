from __future__ import division
# import sys
# import os
# sys.path.append(os.path.abspath(__file__ + "/../../"))
from torch import optim
from src.QA_model import *
from src.util import *

# evaluation function. measure the distance between the embedding of the best generated word and
# the embedding of the first word of true answer
def eval(generated_word, ans1stWord):

    return None


# read embedding
path_to_glove = '../../../data/glove.840B/glove.840B.300d.txt'
embeddings_index, embeddings_size = readGlove(path_to_glove)

# read and prepare input data
cq_path = '../../../data/squad_openNMT/train/cq_sent_concat_min.txt'
a_path = '../../../data/squad_openNMT/train/as_min_NoAnnotate.txt'
cq, a, cq_data, a_data, tokens = prepareInput(src_file=cq_path, tgt_file=a_path)
print('read ' + str(len(cq)) + ' context+question, and ' + str(len(a)) + ' answers')
print('type of element in cq: ' + type(cq[0]) + ' , and type inside it is: ' + type(cq[0].data))
print('type of element in a:  ' + type(a[0])  + ' , and type inside it is: ' + type(a[0].data))

# find effective tokens
effective_tokens, effective_num_tokens = count_effective_num_tokens(tokens, embeddings_index)

# set up model
enc_hidden_size = 500
enc_num_layers = 2
enc_bidirectional = 2
encoder = nn.LSTM(input_size=embeddings_index, hidden_size=enc_hidden_size,
                  num_layers=enc_num_layers, bidirectional=enc_bidirectional)
decoder = distDecoder(input_size=enc_hidden_size, output_size=embeddings_index)
qa = QA(encoder=encoder, decoder=decoder).cuda()

# set up optimnizer
optimizer = optim.SGD(lr=0.001)

# start training
for epoch in range(1,16):

    print('training epoch at ' + str(epoch))

    for iter in range(len(cq)):

        # get a single example
        train_cq = cq[iter]
        train_a = a[iter]
        train_cq_data = cq_data[iter]
        train_a_data = a_data[iter]

        # forward pass to get the last hidden state of encoder
        # enc_out size = (num_layers * num_directions, batch, hidden_size)
        enc_out = qa.forward(train_cq)

        # process the hidden state to be (1, hidden_size)
        dec_in = enc_out[-1] + enc_out[-2]

        # calculate loss
        optimizer.zero_grad()
        loss = qa.decoder.calculateLoss(dec_in, train_a.split(" ")[0], embeddings_index)
        loss.backward()
        optimizer.step()




