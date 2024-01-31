import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.lr = lr

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Note here that the forward step takes as input the input and the hidden state
        # It then combines them bby concatenation before feeding them to the network
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, device):
        # the hidden state needs to be initialized (for the first iteration)
        return torch.zeros(1, self.hidden_size).to(device)

    ## The loss function - Here, we will use Negative Log Likelihood
    def loss(self, y_hat, y):
        fn = nn.NLLLoss()
        return fn(y_hat, y)

    ## The optimization algorithm
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def evaluate(self, X, device):
        hidden = self.initHidden(device)

        for i in range(X.size()[0]):
            output, hidden = self.forward(X[i], hidden)

        return output


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, input_size, hidden_size, output_size, lr):
        super(LSTMTagger, self).__init__()
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.hidden2tag = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2tag2 = nn.Linear(input_size + hidden_size, output_size)

        self.lr = lr

    def initHidden(self):
        # the hidden state needs to be initialized (for the first iteration)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, input, hidden):
        embeds = self.word_embeddings(input)
        lstm_out, hidden = self.lstm(embeds.view(len(input), 1, -1), hidden)

        output = input.view(len(input), 1, -1)
        # TODO not finish yet
        return output, hidden
