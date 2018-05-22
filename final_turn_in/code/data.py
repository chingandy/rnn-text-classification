import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import string
import unicodedata

import torch
from torch.autograd import Variable

from math import sqrt, ceil

np.random.seed(0) 

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


# taken from "Classifying Names with a Character-Level RNN" pytorch tutorial
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# taken from "Classifying Names with a Character-Level RNN" pytorch tutorial
# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)

# taken from "Classifying Names with a Character-Level RNN" pytorch tutorial
# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def display_distribution(all_categories, country_dict):
    # display histogram of distribution of country data points
    num = [] # counts number of cities for each country
    index=0
    for idx, country in enumerate(all_categories):
        for city in country_dict[country]:
            num.append(idx)

    n, bins, patches = plt.hist(num, 244, facecolor='green', alpha=0.75)

    plt.xlabel('country nr')
    plt.ylabel('number of cities')
    plt.title('world-cities distribution')

    plt.grid(True)
    plt.show()

def build_dict_world_cities():

    # Build dictionary with city/country pairs - world-cities data set
    country_dict = {}
    all_categories = []
    old_country = ""
    temp_list = []
    with open('../data/world-cities_csv.csv', 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            city=row[0]
            country=row[1]

            if country == "country":
                continue

            ascii_name = unicode_to_ascii(city)

            if country not in all_categories:
                if len(old_country) > 0:
                    country_dict[old_country]=temp_list
                old_country=country

                all_categories.append(country)
                temp_list=[]

                if(len(ascii_name) > 0):
                    temp_list.append(ascii_name)
            else:
                if(len(ascii_name) > 0):
                    temp_list.append(ascii_name)
        
        # last country
        country_dict[old_country] = temp_list

    n_categories = len(all_categories)

    return country_dict, all_categories, n_categories


def build_dict_geonames():
    # Build dictionary with city/country pairs - geonames data set
    country_dict = {}
    all_categories = []
    old_country = ""
    temp_list = []
    with open('cities1000.txt', encoding="utf8") as f:
        for line in f:
            a=line.split('\t')
            country=a[8]
            city=a[2]
            if country not in all_categories:
                if len(old_country) > 0:
                    country_dict[old_country]=temp_list
                old_country=country
                all_categories.append(country)
                temp_list=[]

            if(len(city) > 0):
                # X.append(city)
                # y.append(country)
                temp_list.append(city)
        # last country
        country_dict[old_country] = temp_list
    n_categories = len(all_categories)

    # Build dictionary with geonames country code/country name pairs
    code_dict = {}
    with open('countryInfo.txt', encoding="utf8") as f:
        for line in f:
            a=line.split('\t')
            code=a[0]
            countryname=a[4]
            code_dict[code]=countryname

    return country_dict, all_categories, n_categories, code_dict #, X, y


def partition_data():

    train_set = {}
    val_set = {}
    test_set = {}

    tot_test=0
    tot_val=0
    tot_train=0
    for category in all_categories:
        cities=country_dict[category]
        np.random.shuffle(cities)
        temptot=len(cities)
        endtrain=ceil(0.7 * temptot)
        endval=ceil(0.7 * temptot) + ceil(0.2 * temptot)
        endtest=temptot

        print(category, endtrain, (endval-endtrain), (endtest-endval), \
            endtrain + (endval-endtrain) + (endtest-endval), temptot)

        tot_test += (endtest-endval)
        tot_val += (endval - endtrain)
        tot_train += endtrain

        if endtrain > 0:
            train_set[category] = cities[0:endtrain]
        if endval > 0:
            val_set[category] = cities[endtrain:endval]
        if endval < endtest:
            test_set[category] = cities[endval:endtest]


    # print how many samples there are in the training, validation and test
    # set respectively
    print('tot train', tot_train, 'tot val', tot_val, 'tot test', tot_test)

    return train_set, val_set, test_set, tot_train, tot_val, tot_test

def partition_x_y():

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for country in train_set:
        cities=train_set[country]
        for city in cities:
            X_train.append(city)
            y_train.append(country)

    for country in val_set:
        cities=val_set[country]
        for city in cities:
            X_val.append(city)
            y_val.append(country)

    for country in test_set:
        cities=test_set[country]
        for city in cities:
            X_test.append(city)
            y_test.append(country)            

    tot_tr=np.concatenate((np.expand_dims(X_train, axis=1), np.expand_dims(y_train, axis=1)), axis=1)
    np.random.shuffle(tot_tr)
    X_train=tot_tr[:, 0]
    y_train=tot_tr[:, 1]

    tot_tr=np.concatenate((np.expand_dims(X_test, axis=1), np.expand_dims(y_test, axis=1)), axis=1)
    np.random.shuffle(tot_tr)
    X_test=tot_tr[:, 0]
    y_test=tot_tr[:, 1]

    tot_tr=np.concatenate((np.expand_dims(X_val, axis=1), np.expand_dims(y_val, axis=1)), axis=1)
    np.random.shuffle(tot_tr)
    X_val=tot_tr[:, 0]
    y_val=tot_tr[:, 1]

    return X_train, y_train, X_val, y_val, X_test, y_test


country_dict, all_categories, n_categories, code_dict  = build_dict_geonames()

# only keeping countries with at least 300 cities -> this will amount to 55 categories (for geonames data set)
big_country_dict={}
big_all_cats = []
tot=0
totfiltered=0

for country in all_categories:
    tot += len(country_dict[country])
    if(len(country_dict[country]) > 300):
        big_all_cats.append(country)
        big_country_dict[country] = country_dict[country]
        totfiltered+=len(big_country_dict[country])


print('remaining categories after filtering')
for idx, country in enumerate(big_all_cats):
    print('(' + country + ',' + code_dict[country] + ')') if(idx==len(big_all_cats)-1) else print('(' + country + ',' + code_dict[country] + ')', end=', ')


n_categories = len(big_all_cats)


# # creating weight vector to handle unbalanced training set
num = [] # counts number of cities for each country
index=0
for country in big_all_cats:
    num.append(len(big_country_dict[country]))

mx = np.max(num)
class_weights = []
for n in num:
    class_weights.append(int(mx/n))
class_weights=torch.FloatTensor(class_weights)


# comment this (and comment awaycreation of class_weights vector abovee) if you want to run on ~all~ the data
all_categories=big_all_cats
country_dict=big_country_dict

train_set, val_set, test_set, tot_train, tot_val, tot_test = partition_data()
X_train, y_train, X_val, y_val, X_test, y_test = partition_x_y() # for "train_model_deterministic"

