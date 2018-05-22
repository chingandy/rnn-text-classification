import torch
from data import *
from model import *
from train import random_training_pair, category_from_output
import matplotlib
matplotlib.use('Agg')

import matplotlib.ticker as ticker
import sys
import math
torch.nn.Module.dump_patches = True


np.random.seed(0) 

def confusion_matrix():

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000 # how many samples will be looked at 

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_pair(test_set)
        output = rnn.evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    precision=0 # precision=true positive/(true positive + false negative)=true positive/sum(actual category)
    recall=0  # recall=true positive/(true positive+false positive) = true positive / sum(predicted as category)
    avg_f1=0
    print('precision\trecall\t\tf1 score\tcategory')
    for i in range(n_categories):
        precision=confusion[i][i] / confusion[i].sum()
        recall=confusion[i][i] / confusion[:, i].sum()
        f1=2*(precision*recall)/(precision+recall)
        avg_f1 += 0 if math.isnan(f1) else f1

        print('num of this category', confusion[i].sum().data.numpy(), \
        'num predictions of this category', confusion[:,i].sum().data.numpy())
        print('%.4f\t\t%.4f\t\t%.4f\t\t%s' % (precision.data.numpy(), recall.data.numpy(), f1.data.numpy(), all_categories[i]))


    print('average f1 score', avg_f1.data.numpy()/n_categories)

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
    plt.savefig('latestcm.png')
    
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

def test_accuracy():
    train_acc=accuracy(country_dict)
    test_acc=accuracy(country_dict)
    print('train accuracy =', train_acc * 100, '%' , 'test accuracy =', test_acc * 100, '%')



if __name__ == '__main__':

    global rnn
    if(len(sys.argv) < 2):
        print('usage: evaluate.py <model name>, where <model name> is either RNN or LSTM or GRU')
        quit()

    model_type = str(sys.argv[1])

    print(model_type)
    if(model_type=="RNN"):
        rnn = torch.load('model.pt')
    elif(model_type=='LSTM'):
        rnn = torch.load('LSTM_model.pt')
    elif(model_type=="GRU"):
        rnn = torch.load('GRU_model_15.pt')
    else:
        print('input: model type (either RNN or LSTM or GRU)')
        quit()


    confusion_matrix()
    test_accuracy()
