import torch.nn as nn
import torch
import utils

class Predictor(nn.Module):
    def __init__(self, emsize, num_of_classes, d1, d2, use_cuda=False):
        super(Predictor, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.emsize = emsize
        self.num_of_classes = num_of_classes
        self.use_cuda = use_cuda
        self.weights_FF_1 = []
        self.bias_FF_1 = torch.zeros(1, num_of_classes)

        self.weights_FF_2 = []
        self.bias_FF_2 = torch.zeros(1, num_of_classes)

        # From the paper,
        # d1 and d2 are the history sizes used in the first and second layers
        self.d1 = d1
        self.d2 = d2

        # initialise the weights of the FF network
        self.initialise_weights()
        self.cudify()

    def cudify(self):
        if self.use_cuda:
            self.bias_FF_1 = self.bias_FF_1.cuda()
            self.bias_FF_2 = self.bias_FF_2.cuda()
            self.weights_FF_1 = [w.cuda() for w in self.weights_FF_1]
            self.weights_FF_2 = [w.cuda() for w in self.weights_FF_2]

    def initialise_weights(self):
        # Use normal distribution to initialise weights of the first feed forward layer
        # dimensions are (emsize * num_of_classes) ==> (n, k) according to paper
        for i in range(self.d1 + 1):
            weight = torch.empty(self.emsize, self.num_of_classes)
            weight = nn.init.normal_(weight)
            self.weights_FF_1.append(weight)

        # Use normal distribution to initialise weights of the second feed forward layer
        # dimensions are (num_of_classes * num_of_classes) ==> (k, k) according to paper
        for i in range(self.d2 + 1):
            weight = torch.empty(self.num_of_classes, self.num_of_classes)
            weight = nn.init.normal_(weight)
            self.weights_FF_2.append(weight)

    def forward(self, structural_representations):

        # This will be of the shape N * B * H where
        # N is number of sentences, B is batch size and H is the dimension of class representations derived
        # in previous step.
        number_of_sentences_to_classify = structural_representations.shape[0]

        predictions = []

        # Let us first get our class representations
        for i in range(number_of_sentences_to_classify):

            class_representations = []
            # The j loop according to the paper for FF1
            if i >= self.d2:
                for j in range(i - self.d2, i + 1):
                    # The inner loop for FF1
                    summation = 0
                    for k in range(self.d1 + 1):
                        if j >= k:
                            linear = torch.mm(structural_representations[j - k], self.weights_FF_1[k]) + self.bias_FF_1
                            # We have to sum all these terms
                            summation += linear

                    # tanh of the summed values is our class representation
                    class_representations.append(self.tanh(summation))

            # We didn't get enough terms for summation, then we simply use our current sentence as is
            if not class_representations:
                linear = torch.mm(structural_representations[i], self.weights_FF_1[0]) + self.bias_FF_1
                class_representations.append(self.tanh(linear))

            # Use the class activations now, feed to the second layer of NN and fetch the required sum.
            summation = 0
            for j in range(len(class_representations)):
                linear = torch.mm(class_representations[j], self.weights_FF_2[j]) + self.bias_FF_2
                summation += linear

            # Predicitions' probability distributions using softmax
            sent_pred = self.softmax(summation)
            predictions.append(sent_pred)

        # Stack the predictions into a single tensor and cuda() if applicable
        predictions = torch.stack(predictions)
        if self.use_cuda:
            predictions = predictions.cuda()

        return predictions

class Pooling(nn.Module):
    def __init__(self, type):
        super(Pooling, self).__init__()
        self.type = type

    def forward(self, hidden_states):
        # Stack them up to get one single tensor of dimension
        # N * B * H
        # B = Batch size
        # N = Number of hidden states i.e. the time steps i.e. number of words per abstract
        # H = Hidden dimension of the LSTM
        combine_states = torch.stack(hidden_states)

        # Apply max pooling to the hidden states. Element wise max.
        if self.type == "max":
            # Gives us a vector of dimension B * H. So,
            # for each sentence we have H dimensional sentence representation vector essentially
            sentence_representations = torch.max(combine_states, 1)[0]
        else:
            sentence_representations = torch.mean(combine_states, 1)

        return sentence_representations

class Annotator(nn.Module):
    def __init__(self, embedding, emsize, hidden_size, pooling_dropout, number_of_structural_labels, config, use_cuda=False):
        super(Annotator, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.emsize = emsize
        self.pooling_dropout = nn.Dropout(p=pooling_dropout)
        self.embedding = embedding
        self.rnn_cell = nn.LSTMCell(emsize, self.hidden_size)
        self.pooling = Pooling("max")
        self.predictor = Predictor(self.hidden_size, number_of_structural_labels, config.d1, config.d2, use_cuda)

    def forward(self, abstracts):
        embedded = self.embedding(abstracts)

        # B = Batch size
        # S = Number of sentences in each abstract
        # W = Number of words in each sentence
        # E = Embedding dimension
        # embedded B * S * W * E --> S * B * W * E
        embedded = embedded.t()

        # number of sentences
        num_of_sentences = embedded.shape[0]

        # number of words in each sentence
        num_of_words = embedded.shape[2]

        # Number of abstracts being processed at once
        batch_size = embedded.shape[1]

        # Initial hidden and cell states of the LSTMCell
        h, c = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
        if self.use_cuda:
            h = h.cuda()
            c = c.cuda()

        # List to keep track of the hidden states
        hidden_states = []

        # Process one sentence at a time
        for j in range(num_of_sentences):

            # List to keep track of the hidden states
            sent_hidden_states = []

            # Feed one word at a time to the LSTMCell to get the individual hidden states
            for i in range(num_of_words):

                # Current input to the LSTMCell. It would be of dim B * E
                lstm_input = embedded[j, :, i, :]

                # One call to the LSTM. Gives us the hidden state after this time step
                h, c = self.rnn_cell(lstm_input, (h, c))

                # The hidden states at every time-step
                sent_hidden_states.append(h)

            hidden_states.append(torch.stack(sent_hidden_states))

        # Once we are done processing the individual words, one at a time for the
        # j^th sentence of each abstract of the batch, we need to apply max-pooling
        # which is element wise max of each hidden states.
        sentence_representations = self.pooling(hidden_states)

        # Apply dropout after the pooling layer
        sentence_representations = self.pooling_dropout(sentence_representations)

        # Use the second part of the network to make predictions for each sentence of each abstract
        predictions = self.predictor(sentence_representations)

        # The predictions we get are of the dimension N * B * C
        # N = number of sentences
        # B = Batch size
        # C are the number of output classes
        # We need B * C * N
        predictions = predictions.t().transpose(1,2)
        return predictions
