import re
import pickle

import argparse
import numpy as np
import pickle
import tensorflow as tf
from preprocess import pipeline_for_tokenization

def train_model(model, src_inputs, tgt_inputs, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    # print('train_model is called!')
    try:
        for epoch in range(args.epochs):
            stats += [model.train(tgt_inputs, src_inputs, pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else:
            raise e

    return stats

if __name__ == '__main__':

    with open('prep_data.p', 'rb') as f:
        data = pickle.load(f)

    # prep_data = pipeline_for_tokenization(data, use_spacy=True, save_to_file=True)
    print(data.keys())


