"""
Created by Siyang Zhang
Date: 2022-11-10

Do the preprocessing of the data, including: chop the ingredients into words.
"""

import csv
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
import pickle

# order, Title, Ingredients, Instructions, ImageName, Cleaned_Ingredients
#   0  ,   1  ,      2     ,      3      ,     4     ,         5

nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '¼', '½', '¾', '⅓', '⅔', '⅛', '⅜', '⅝', '⅞', '⅕', '⅖', '⅗',
        '⅘', '⅙', '⅚', '⅐', '⅑', '⅒']
units = ['tsp.', 'Tbsp.', 'lb.', 'oz.', 'cup', 'cups', 'tbsp', 'tsp', 'lb', 'oz', 'g', 'kg', 'ml', 'l', 'gallon',
         'quart', 'can', 'jar']


def extract_ingredients(ingredient_str):
    pass


def remove_commas(s):
    return re.sub(',.*', '', s)


def remove_parentheses(s):
    parentheses = '[\(\[].*?[\)\]]'
    return re.sub(parentheses, '', s)


def remove_units(s):
    s = re.split(' +', s)
    if s[0][0] in nums:
        if s[1] in units:
            s = s[2:]
        else:
            s = s[1:]
    return ' '.join(s)


def remove_spaces(s):
    return re.sub(' +', ' ', s)



def load_data(path, num=10000, ner=False):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    with open(path, 'r', newline='', encoding='gbk') as csvfile:
        reader = csv.DictReader(csvfile)
        if ner:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        X = []
        Y = []
        vocab_dish = Counter()
        vocab_ingredient = Counter()
        vocab = Counter
        for i, row in enumerate(reader):
            # print(row)
            ingredients = row['ingredients'].strip('[\']')
            ingredients = ingredients.split('\', \'')

            print(f'\rparsing {i+1} row')
            if ner:

                for j, ingredient in enumerate(ingredients):

                    ingredient = nlp(ingredient)
                    lst = []
                    for token in ingredient:
                        lst.append(token.lemma_)
                    vocab.update(lst)
                    ingredients[j] = ' '.join(lst)

            Y.append(row['name'])

            # dish_vocab = []
            # for dish_name in row['name']:
            #     dish_tokens = nlp(dish_name)
            #     for token in dish_tokens:
            #         if not token.is_punct:
            #             dish_vocab.append(token.lemma_)
            #
            # vocab_dish.update(dish_vocab)
            # vocab.update(dish_vocab)
            #
            # X.append(ingredients)
            #
            # vocab_ingredient.update(ingredients)

            if num > 0 and i >= num - 1:
                break

        return np.array(X), np.array(Y), vocab_dish, vocab_ingredient, vocab

def load(path):
    with open(path, 'r', newline='', encoding='gbk') as csvfile:
        reader = csv.DictReader(csvfile)
        d = dict()
        for i, row in enumerate(reader):
            ingredients = row['ingredients'].strip('[\']')
            ingredients = ingredients.split('\', \'')
            d[row['name']] = ingredients
        return d
def create_pickle(path):
    with open(f'data.p', 'wb') as pickle_file:
        pickle.dump(load(path), pickle_file)
        # pickle.dump({'name': '1', 'age': '2'}, pickle_file)
    print(f'Data has been dumped into data.p!')


def truncate(arr, size):
    if len(arr) > size:
        return arr[:size + 1]
    else:
        arr += (size + 1 - len(arr)) * ['<pad>']
        return arr


def preprocess_dish_sentences(sentences, window_size):
    for i, sentence in enumerate(sentences):
        sentence_no_punc = re.sub('[^a-zA-Z0-9 ]', ' ', sentence.lower())
        clean_words = [word for word in sentence_no_punc.split() if ((len(word) > 1) and word.isalpha())]

        sentence_new = ['<start>'] + clean_words[:window_size - 1] + ['<end>']
        sentences[i] = truncate(sentence_new, window_size)

def preprocess_ingredient_sentences(sentences, window_size):
    for i, sentence in enumerate(sentences):
        sentence_no_punc = re.sub('[^a-zA-Z0-9 ]', ' ', sentence.lower())
        clean_words = [word for word in sentence_no_punc.split() if ((len(word) > 1) and word.isalpha())]
        sentence_new = ['<start>'] + clean_words[:window_size - 1] + ['<end>']
        sentences[i] = truncate(sentence_new, window_size)

def load_from_data(path=None):
    if path is None:
        data = {'Glazed Finger Wings': ['chicken-wings', 'sugar,', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water', 'lemon juice', 'soy sauce'], 'Country Scalloped Potatoes &amp; Ham (Crock Pot)': ['potatoes', 'onion', 'cooked ham', 'country gravy mix', 'cream of mushroom soup', 'water', 'cheddar cheese'], 'Fruit Dream Cookies': ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder', 'vanilla', 'all-purpose flour', 'white chocolate chips', 'orange drink mix', 'colored crystal sugar'], 'Tropical Breakfast Risotti': ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk', 'raisins', 'sweetened flaked coconut', 'toasted sliced almonds', 'banana'], 'Linguine W/ Olive, Anchovy and Tuna Sauce': ['anchovy fillets', 'tuna packed in oil', 'kalamata olive', 'garlic cloves', 'fresh parsley', 'fresh lemon juice', 'salt %26 pepper', 'olive oil', 'linguine']}
    else:
        with open(path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

    import spacy
    nlp = spacy.load('en_core_web_sm')
    vocab_dish = Counter()
    vocab_ingredient = Counter()
    window_size = 10
    X = []
    Y = []
    word2idx = {}
    vocab_size = 0
    num_data = len(data)
    for i, (dish, ingredients) in enumerate(data.items()):

        print(f'\rparsing {i + 1}/{num_data} row', end='')
        Yi = []
        for i, ingredient_list in enumerate(ingredients):
            # ingredient_list = nlp(ingredient)
            lst = []
            for token in ingredient:
                if not token.is_punct:
                    lst.append(token.lemma_.lower())
            # vocab_ingredient.update(lst)

            ingredient = ' '.join(lst)

            if ingredient not in word2idx:
                word2idx[ingredient] = vocab_size
                vocab_size += 1

            Yi.append(ingredient)
            ingredients[i] = ingredient
        vocab_ingredient.update(ingredients)
        Y.append(truncate(Yi, window_size))

        dish = nlp(dish)
        d_list = []
        for token in dish:
            if not token.is_punct:
                word = token.lemma_.lower()
                d_list.append(word)

                if word not in word2idx:
                    word2idx[word] = vocab_size
                    vocab_size += 1

        vocab_dish.update(d_list)
        X.append(truncate(d_list, window_size))
    print()
    vocab = vocab_dish + vocab_ingredient
    print(vocab_dish)
    print(vocab_ingredient)
    print(len(vocab), vocab)

    for x,y in zip(X, Y):
        print(x)
        print(y)
        print()
    X = np.array(X)
    Y = np.array(Y)
    idx2word = {i: w for w, i in word2idx.items()}
    print(X.shape, Y.shape)
    print(len(word2idx), word2idx)
    print(idx2word)
    print(vocab_size)

    # write into file
    with open(f'prep_data.p', 'wb') as processed_file:
        pickle.dump(load(path), processed_file)
        # pickle.dump({'name': '1', 'age': '2'}, pickle_file)
    print(f'Data has been dumped into prep_data.p!')


def load_and_preprocess(path=None):
    if path is None:
        data = {'Glazed Finger Wings': ['chicken-wings', 'sugar,', 'cornstarch', 'salt', 'ground ginger', 'pepper', 'water', 'lemon juice', 'soy sauce'], 'Country Scalloped Potatoes &amp; Ham (Crock Pot)': ['potatoes', 'onion', 'cooked ham', 'country gravy mix', 'cream of mushroom soup', 'water', 'cheddar cheese'], 'Fruit Dream Cookies': ['butter', 'shortening', 'granulated sugar', 'eggs', 'baking soda', 'baking powder', 'vanilla', 'all-purpose flour', 'white chocolate chips', 'orange drink mix', 'colored crystal sugar'], 'Tropical Breakfast Risotti': ['water', 'instant brown rice', 'pineapple tidbits', 'skim evaporated milk', 'raisins', 'sweetened flaked coconut', 'toasted sliced almonds', 'banana'], 'Linguine W/ Olive, Anchovy and Tuna Sauce': ['anchovy fillets', 'tuna packed in oil', 'kalamata olive', 'garlic cloves', 'fresh parsley', 'fresh lemon juice', 'salt %26 pepper', 'olive oil', 'linguine']}
    else:
        with open(path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

    vocab_dish = Counter()
    vocab_ingredient = Counter()
    window_size = 10
    X = []
    Y = []
    word2idx = {}
    vocab_size = 0
    num_data = len(data)

    for i, (dish, ingredients) in enumerate(data.items()):
        preprocess_ingredient_sentences(ingredients, window_size)
        # vocab_ingredient.update(ingredients)
        Y.append(ingredients)

        X.append(dish)

    preprocess_dish_sentences(X, window_size)

    for x, y in zip(X, Y):
        print(x)
        print(y)
        print()
def test():
    import spacy
    nlp = spacy.load('en_core_web_sm')
    s = '(Papas Rellenas De Picadillo) Meat-Stuffed Potato, Croquettes'
    tokens = nlp(s)
    for token in tokens:
        print(token.lemma_, token.pos_, token.tag_, token.dep_, token.is_punct)

if __name__ == '__main__':
    # X, Y, vocab_dish, vocab_ingredient, vocab = load_data('recipes_w_search_terms_truncated.csv', num=-1, ner=False)
    # print(X.shape)
    # print(Y.shape)
    # print(X)
    # for i, (name, count) in enumerate(sorted(vocab.items(), key=lambda x: x[1], reverse=True)):
    #     print(name, count)
    # create_pickle('recipes_w_search_terms_truncated.csv')


    # with open('data.p', 'rb') as pickle_file:
    #     data = pickle.load(pickle_file)
    #     print(data)

    # load_from_data()
    load_and_preprocess()
    # s = ['Papas Rellenas De Picadillo) Meat-Stuffed Potato, Croquettes', 'Glazed Finger Wings', 'Country Scalloped Potatoes &amp; Ham (Crock Pot)', 'Fruit Dream Cookies', 'Tropical Breakfast Risotti', 'Linguine W/ Olive, Anchovy and Tuna Sauce']
    # preprocess_sentences(s, 10)
    # print(s)



