import torch
from data import *
from model import *
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import fileinput
import sys

n_hidden = 128
n_epochs = 100000
print_every = 1000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(country_dict[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn=0

def train_model(file_name):

    print(rnn)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for epoch in range(1, n_epochs + 1):

        category, line, category_tensor, line_tensor = random_training_pair()

        output, loss = rnn.train(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

            # just printing the sum of the weights to check that theyre not exploding or vanishing
            # print(np.sum(rnn.i2o.weight.data.numpy()), np.sum(rnn.i2h.weight_ih_l0.data.numpy())) 
        
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


    # plot all losses
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    torch.save(rnn, file_name) # save model



if __name__ == '__main__':

    print(str(sys.argv))

    if(len(sys.argv) < 2):
        print('usage: train.py <model name>, where <model name> is either RNN or LSTM')
        quit()

    model_type = str(sys.argv[1])

    print(model_type)
    if(model_type=="RNN"):
        global rnn
        rnn = RNN(n_letters, n_hidden, n_categories)
        file_name='model.pt'
    elif(model_type=='LSTM'):
        global rnn
        rnn = RNN_LSTM(n_letters, n_hidden, n_categories)
        file_name='LSTM_model.pt'
    else:
        print('input: model type (either RNN or LSTM)')
        quit()


    rnn.optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    rnn.criterion = nn.NLLLoss(weight=class_weights)

    train_model(file_name)




