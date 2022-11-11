import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import spacy
from collections import Counter

if __name__ == '__main__':
    a = ['water', 'grit', 'salt', 'cheddar cheese', 'garlic', 'olive oil']
    s = 'I love you but he loves her and my friends are all nlp big fans.'

    counter = Counter()
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(s)
    counter.update(doc)
    print(counter)