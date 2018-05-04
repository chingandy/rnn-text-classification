import csv
import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import string
import unicodedata

import torch

from math import sqrt

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


def build_dict():

    # Build dictionary with city/country pairs
    country_dict = {}
    all_categories = []
    old_country = ""
    temp_list = []
    with open('../data/world-cities_csv.csv', 'r') as f:
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



country_dict, all_categories, n_categories = build_dict()
print("number of categories:", n_categories)
# display_distribution(all_categories, country_dict)

print(country_dict['Vietnam'])
line = line_to_tensor(country_dict['Vietnam'][0])
print(line.shape)
print(country_dict['Macedonia']) 


for country in all_categories:
    for city in country_dict[country]:
        line=line_to_tensor(city);
        if(line.size()==0):
            print(city)


# only keeping countries with at least 100 cities -> this will amount to 40 categories
big_country_dict={}
big_all_cats = []

for country in all_categories:
    if(len(country_dict[country]) > 100):
        big_all_cats.append(country)
        big_country_dict[country] = country_dict[country]

n_categories = len(big_all_cats)

# creating weight vector to handle unbalanced training set
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

