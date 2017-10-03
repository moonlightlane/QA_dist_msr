from __future__ import division
# import sys
# import os
# sys.path.append(os.path.abspath(__file__ + "/../../"))
from torch import optim
import random
from src.QA_model import *
from src.util import *

# read embedding
path_to_glove = '../../../data/glove.840B/glove.840B.300d.txt'
embeddings_index, embeddings_size = readGlove(path_to_glove)

# read and prepare input data
cq_path = '../../../data/squad_openNMT/train/cq_sent_concat_min.txt'
a_path = '../../../data/squad_openNMT/train/as_min_NoAnnotate.txt'
cq, a, cq_data, a_data, tokens = prepareInput(src_file=cq_path, tgt_file=a_path)
print('read ' + str(len(cq)) + ' context+question, and ' + str(len(a)) + ' answers, for test.')
print('type of element in cq: ' + type(cq[0]) + ' , and type inside it is: ' + type(cq[0].data))
print('type of element in a:  ' + type(a[0])  + ' , and type inside it is: ' + type(a[0].data))
# find effective tokens
effective_tokens, effective_num_tokens = count_effective_num_tokens(tokens, embeddings_index)

# read and prepare validation data
test_cq_path = '../../../data/squad_openNMT/test/test_cq_sent_concat_min.txt'
test_a_path = '../../../data/squad_openNMT/test/test_as_min_NoAnnotate.txt'
test_cq, test_a, test_cq_data, test_a_data, test_tokens = prepareInput(src_file=test_cq_path, tgt_file=test_a_path)
print('read ' + str(len(test_cq)) + ' context+question, and ' + str(len(test_a)) + ' answers, for validation.')
print('type of element in test_cq: ' + type(test_cq[0]) + ' , and type inside it is: ' + type(test_cq[0].data))
print('type of element in test_a:  ' + type(test_a[0])  + ' , and type inside it is: ' + type(test_a[0].data))
# find effective tokens in validation set
test_effective_tokens, test_effective_num_tokens = count_effective_num_tokens(test_tokens, embeddings_index)

effective_tokens = list(set(test_effective_tokens + effective_tokens))
effective_num_tokens = len(effective_tokens)

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
record_loss_every = 100
loss_vec = []
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

        # record loss
        if iter != 0 and iter % record_loss_every == 0:
            loss_vec.append(loss)
        # print at the last iteration in each epoch
        if iter == len(cq) - 1:
            print('training loss at last iteration of epoch ' + str(epoch) + ' is: ' + str(loss))
        # update
        loss.backward()
        optimizer.step()

    # perform evaluation (average loss) on the validation set every epoch
    validate_loss = 0.0
    for i in range(len(test_cq)):
        test_enc_out = qa.forward(test_cq_data[i])
        test_dec_in = test_enc_out[-1] + test_enc_out[-2]
        # accumulate loss
        validate_loss += qa.decoder.calculateLoss(test_dec_in, test_a[i].split(" ")[0], embeddings_index)
        true_ans1stWord.append(test_a.split(" ")[0])
        gen_ans1stWord.append(qa.decoder.generateWord(test_dec_in, test_cq[i].split(" "), embeddings_index))
    validate_loss = validate_loss / len(test_cq)
    print('average loss over validation set is: ' + str(validate_loss))
    # sample true answer first word and generated answer first word
    sample_idx = random.sample(range(len(test)cq), 20)
    true_ans1stWord = [test_a[i].split(" ")[0] for i in sample_idx]
    gen_ans1stWord = [gen_ans1stWord[i] for i in sample_idx]

    # save a checkpoint model for every epoch
    checkpoint = {
        'model': qa.state_dict(),
        'encoder': qa.encoder.state_dict(),
        'generator': qa.decoder.state_dict(),
        'epoch': epoch,
        'optim': optimizer
    }
    torch.save(checkpoint,
               '%s_acc_%.2f_ppl_%.2f_e%d.pt'
               % (opt.save_model, valid_stats.accuracy(),
                  valid_stats.ppl(), epoch))

