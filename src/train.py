import torch
from data import *
from model import *
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_hidden = 128
n_epochs = 100000
print_every = 1000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def category_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    # print('category i', category_i)
    return all_categories[category_i], category_i

def random_choice(l):
    #print('len', len(l))
    #print('l', l)
    return l[random.randint(0, len(l) - 1)]

def random_training_pair():
    #print('random training pair', all_categories)
    category = random_choice(all_categories)
    #print('chosen category', category)
    #print('random training pair', all_categories)

    line = random_choice(country_dict[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    # print(line_tensor.size())

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

        # if(np.isnan(output.data.numpy()).any()):
        #     print('output contains nan')

        # print('output contains nan?: ', np.isnan(output.data.numpy()).any())


    # print(output)
    # print(category_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(weight=class_weights)

def train():


    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    print(rnn.i2o.weight.data.numpy())
    print(rnn.i2h.weight.data.numpy())

    for epoch in range(1, n_epochs + 1):

        # print('epoch', epoch)
        category, line, category_tensor, line_tensor = random_training_pair()

        # print(category, line)
        output, loss = train(category_tensor, line_tensor)
        # print(output, loss)
        # print(output.data.topk(1))
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

            print(np.sum(rnn.i2o.weight.data.numpy()), np.sum(rnn.i2h.weight.data.numpy()))
        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0



    print(rnn.i2o.weight.data.numpy())
    print(rnn.i2h.weight.data.numpy())

    # plot all losses
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    torch.save(rnn, 'model.pt')



if __name__ == '__main__':
    train()
