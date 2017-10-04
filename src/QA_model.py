from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

# the decoder class
class distDecoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(distDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.out = nn.Linear(self.input_size, self.output_size)

    def calculateLoss(self, loss, inputVec, ans1stWord, embeddings_index):
        # can modify the below loss to be other distance metric
        # here, using euclidean distance
        # NOTE here the inputVec is the processed vector of encoder outputted hidden state
        #      inputVec dim = (1, hidden_size)
        # loss = torch.norm(embeddings_index[ans1stWord].unsqueeze(0) - self.out(inputVec).data, p=2)**2
        # return Variable(loss)
        return loss(self.out(inputVec), Variable(embeddings_index[ans1stWord].unsqueeze(0)))

    def generateWord(self, crit, inputVec, wordsInInput, embeddings_index):
        # NOTE hidden layer output size from encoder  = (num_layers * num_directions, batch, hidden_size), alternating between layers
        #      see https://discuss.pytorch.org/t/how-can-i-know-which-part-of-h-n-of-bidirectional-rnn-is-for-backward-process/3883
        # inputVec dim = (1, hidden_size)
        decoded_word = None
        min_dist_old = 1e15 # an arbitrary large number
        for word in wordsInInput:
            min_dist_new = self.calculateLoss(crit, inputVec, word, embeddings_index)
            if min_dist_new < min_dist_old:
                min_dist_old = min_dist_new
                decoded_word = word
        return word


class QA(nn.Module):

    def __init__(self, encoder, decoder):
        super(QA, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Input):
        _, enc_output = self.encoder(Input)
        return enc_output

    def backward(self, optimizer, inputVec, ans1stWord, embeddings_index):
        loss = self.decoder.calculateLoss(inputVec, ans1stWord, embeddings_index)
        loss.backward()
        optimizer.step()