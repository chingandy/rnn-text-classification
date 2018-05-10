
import torch
from data import *
from model import *
from train import random_training_pair, category_from_output

import matplotlib.ticker as ticker
import sys

rnn=0

def confusion_matrix():

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000 # how many samples will be looked at 


    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_pair()
        output = rnn.evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



if __name__ == '__main__':

    if(len(sys.argv) < 2):
        print('usage: train.py <model name>, where <model name> is either RNN or LSTM')
        quit()

    model_type = str(sys.argv[1])

    print(model_type)
    if(model_type=="RNN"):
        global rnn
        rnn = torch.load('model.pt')
    elif(model_type=='LSTM'):
        global rnn
        rnn = torch.load('LSTM_model.pt')
    else:
        print('input: model type (either RNN or LSTM)')
        quit()


    confusion_matrix()
