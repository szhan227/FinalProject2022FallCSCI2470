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

    src_inputs = data['X']
    tgt_inputs = data['Y']
    src_w2i = data['dish_word2idx']
    tgt_w2i = data['ingredient_word2idx']
    src_i2w = data['dish_idx2word']
    tgt_i2w = data['ingredient_idx2word']
    src_vocab_size = data['dish_vocab_size']
    tgt_vocab_size = data['ingredient_vocab_size']

    from decoder import RNN

    src_inputs = tf.convert_to_tensor(src_inputs)
    tgt_inputs = tf.convert_to_tensor(tgt_inputs)

    model = RNN(src_vocab_size, tgt_vocab_size, hidden_size=256, window_size=20)
    output = model(src_inputs[:100], tgt_inputs[:100])
    print(output.shape)
    print(src_vocab_size)


