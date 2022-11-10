"""
Created by Siyang Zhang
Date: 2022-11-10

Do the preprocessing of the data, including: chop the ingredients into words.
"""

import csv
import re

# order, Title, Ingredients, Instructions, ImageName, Cleaned_Ingredients
#   0  ,   1  ,      2     ,      3      ,     4     ,         5

nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '¼', '½', '¾', '⅓', '⅔', '⅛', '⅜', '⅝', '⅞', '⅕', '⅖', '⅗',
        '⅘', '⅙', '⅚', '⅐', '⅑', '⅒']

def extract_ingredients(ingredient_str):
    pass


def load_data(path):
    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for i, row in enumerate(reader):
            print(row['Title'])
            ingredients = row['Cleaned_Ingredients'].strip('[\']')
            ingredients = ingredients.split('\', ')
            for each in ingredients:
                each = each.strip('\'')
                print(each)
                comma = each.find(',')
                if comma != -1:
                    each = each[:comma]
                each = each.split(' ')
                if each[0] in nums:
                    each = each[2:]

                print(' '.join(each))
                print()
            break


if __name__ == '__main__':
    load_data('data.csv')

