import torch
import torch.nn as nn

# the decoder class
class distDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(distDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.out = nn.Linear(self.input_size, self.output_size)
    def calculateLoss(self, inputVec, ans1stWord, embeddings_index):
        # can modify the below loss to be other distance metric
        # here, using euclidean distance
        # NOTE here the inputVec is the processed vector of encoder outputted hidden state
        #      inputVec dim = (1, hidden_size)
        # TODO verify value for dim
        loss = torch.norm(embeddings_index[ans1stWord] - self.out(inputVec), p=2, dim=1, out=True)^2
        return loss
    def generateWord(self, inputVec, wordsInInput, embeddings_index):
        # NOTE hidden layer output size from encoder  = (num_layers * num_directions, batch, hidden_size), alternating between layers
        #      see https://discuss.pytorch.org/t/how-can-i-know-which-part-of-h-n-of-bidirectional-rnn-is-for-backward-process/3883
        # inputVec dim = (1, hidden_size)
        decoded_word = None
        min_dist_old = 1e15 # an arbitrary large number
        for word in wordsInInput:
            min_dist_new = self.calculateLoss(inputVec, word, embeddings_index)
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
        enc_output = self.encoder(Input)


# dd = distDecoder(a, b);
# qa = QA(nn.LSTM(input_size, hidden_size, dropout, bidirectional),
#         dd.input_size, dd.output_size))

# function to prepare input
def prepareInput(qc_file, a_file):
    return None

