import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()
        
        self.optimizer.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

            # print('output contains nan?: ', np.isnan(output.data.numpy()).any())

        loss = self.criterion(output, category_tensor)
        loss.backward()

        self.optimizer.step()

        return output, loss.data

    # Just return an output given a line
    def evaluate(self, line_tensor):
        hidden = self.init_hidden()
        
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
        
        return output


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.LSTM(input_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):

        combined=torch.cat((input[0],hidden[0]),1);
        hidden, (hidden, cell) = self.i2h(input, (hidden, cell))
        output = self.i2o(combined) # should i2o get cell state? no right?
        output = self.softmax(output)
        return output, hidden, cell

    def init_hidden(self):
        return Variable(torch.zeros(1, 1,  self.hidden_size))

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()
        cell = self.init_hidden()
        self.optimizer.zero_grad()

        line_tensor.unsqueeze_(1) # make input 3-dimensional, with dimension sequence length x mini-batch size x input dimension

        for i in range(line_tensor.size()[0]):
            output, hidden, cell = self(line_tensor[i], hidden, cell)

        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()

        return output, loss.data

    # Just return an output given a line
    def evaluate(self, line_tensor):
        hidden = self.init_hidden()
        cell = self.init_hidden()
        line_tensor.unsqueeze_(1)
        for i in range(line_tensor.size()[0]):
            output, hidden, cell = self(line_tensor[i], hidden, cell)
        
        return output

