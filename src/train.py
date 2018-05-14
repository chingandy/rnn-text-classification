import torch
from data import *
from model import *
import random
import time
import math

import copy
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

def random_training_pair(current_dict):
    category = random.choice(all_categories)
    line = random.choice(current_dict[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor


def size_dict(current_dict):
    num=0
    for country in current_dict:
        num+=len(current_dict[country])
    return num


def accuracy(curr_set):
    correct=0
    tot=0

    for num in range(0, 1000):
        category, line, category_tensor, line_tensor = random_training_pair(curr_set)
        output = rnn.evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i=all_categories.index(category)
        if category_i==guess_i:
            correct+=1
        tot+=1

    acc=(correct/tot)
    return acc


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

rnn=0

def train_model(title, file_name):

    print(rnn)
    # Keep track of losses for plotting
    current_loss = 0
    current_loss_val = 0

    all_losses_train = []
    all_losses_val = []

    start = time.time()

    for epoch in range(1, n_epochs + 1):

        category, line, category_tensor, line_tensor = random_training_pair(train_set)
        output, loss = rnn.train(category_tensor, line_tensor)
        current_loss += loss

        _, _, category_tensor, line_tensor = random_training_pair(val_set)
        out = rnn.evaluate(line_tensor)
        loss = rnn.criterion(out, category_tensor)
        current_loss_val += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % code_dict[category]
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, code_dict[guess], correct))

            # just printing the sum of the weights to check that theyre not exploding or vanishing
            print(np.sum(rnn.i2o.weight.data.numpy()), np.sum(rnn.i2h.weight.data.numpy())) 
        
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses_train.append(current_loss / plot_every)
            all_losses_val.append(current_loss_val / plot_every)
            current_loss = 0
            current_loss_val=0


    # plot all losses
    plt.figure()

    plt.plot(np.arange(0, n_epochs, plot_every), all_losses_train, label="train loss")
    plt.plot(np.arange(0, n_epochs, plot_every), all_losses_val, label="val loss")
    np.save('train_error_LSTM_model_5.npy', all_losses_train)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(loc=1)
    plt.show()


    torch.save(rnn, file_name) # save model

def train_model_deterministic(title, file_name):
    
    # instead of randomly sampling data points, go through entire data set every epoch
    print(rnn)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for epoch in range(1, n_epochs + 1):

        for num in range(0, tot_train):

            category=y_train[num]
            line=X_train[num]
            category_tensor=Variable(torch.LongTensor([all_categories.index(category)]))
            line_tensor=Variable(line_to_tensor(line))

            output, loss = rnn.train(category_tensor, line_tensor)
            current_loss += loss

            # Print epoch number, loss, name and guess
            if num % print_every == 0:
                guess, guess_i = category_from_output(output)
                correct = '✓' if guess == category else '✗ (%s)' % code_dict[category]                
                print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, code_dict[guess], correct))
        
                # just printing the sum of the weights to check that theyre not exploding or vanishing
                print(np.sum(rnn.i2o.weight.data.numpy()), np.sum(rnn.i2h.weight.data.numpy())) 
        
            # Add current loss avg to list of losses
            if num % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    # plot all losses
    plt.figure()

    plt.plot(np.arange(0, n_epochs, plot_every), all_losses)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('cost')
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
        title = 'RNN model'
    elif(model_type=='LSTM'):
        global rnn
        rnn = RNN_LSTM(n_letters, n_hidden, n_categories)
        file_name='LSTM_model_5.pt'
        title = 'LSTM model'
    else:
        print('input: model type (either RNN or LSTM)')
        quit()


    rnn.optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    rnn.criterion = nn.NLLLoss()

    train_model(title, file_name)
    # train_model_deterministic(title, file_name)


