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
        vocab_dish = set()
        vocab_ingredient = set()
        for i, row in enumerate(reader):
            # print(row)
            ingredients = row['ingredients'].strip('[\']')
            ingredients = ingredients.split('\', \'')

            if ner:
                for j, ingredient in enumerate(ingredients):
                    ingredient = nlp(ingredient)
                    lst = []
                    for token in ingredient:
                        lst.append(token.lemma_)
                    ingredients[j] = ' '.join(lst)

            Y.append(row['name'])
            vocab_dish.add(row['name'])
            X.append(ingredients)
            for ingredient in ingredients:
                vocab_ingredient.add(ingredient)

            if num > 0 and i >= num - 1:
                break
        return np.array(X), np.array(Y), vocab_dish, vocab_ingredient


if __name__ == '__main__':
    X, Y, vocab_dish, vocab_ingredient = load_data('recipes_w_search_terms_truncated.csv', num=100000)
    print(X.shape)
    print(Y.shape)
    print(X)
    print(len(vocab_dish), len(vocab_ingredient))
