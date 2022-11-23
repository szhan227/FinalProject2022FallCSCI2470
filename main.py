import re
import pickle

import argparse
import numpy as np
import pickle
import tensorflow as tf
from model import accuracy_function, loss_function, DishIngredientPredictorModel
from decoder import RNN, Transformer

def compile_model(model):
    '''Compiles model'''
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss_function,
                  metrics=[accuracy_function])


def train_model(model, src_inputs, tgt_inputs, src_pad_idx, tgt_pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    # print('train_model is called!')
    try:
        for epoch in range(args.epochs):
            stats += [model.train(tgt_inputs, src_inputs, src_pad_idx, tgt_pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], src_pad_idx, tgt_pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else:
            raise e

    return stats


def build_model(args):
    with open('prep_data.p', 'rb') as f:
        data = pickle.load(f)

    src_train_inputs = data['X'][:-1000]
    tgt_train_inputs = data['Y'][:-1000]
    src_test_inputs = data['X'][-1000:]
    tgt_test_inputs = data['Y'][-1000:]
    src_w2i = data['dish_word2idx']
    tgt_w2i = data['ingredient_word2idx']
    src_i2w = data['dish_idx2word']
    tgt_i2w = data['ingredient_idx2word']
    src_vocab_size = data['dish_vocab_size']
    tgt_vocab_size = data['ingredient_vocab_size']

    predictor_class ={
        'rnn': RNN,
        'transformer': Transformer
    }[args.type]

    hidden_size = 256
    window_size = 20

    predictor = predictor_class(src_vocab_size, tgt_vocab_size, hidden_size, window_size)

    model = DishIngredientPredictorModel(
        predictor,
    )

    compile_model(model)

    train_model(model, src_train_inputs, tgt_train_inputs, src_w2i['<pad>'], tgt_w2i['<pad>'], args, (src_test_inputs, tgt_test_inputs))

    return model, (src_test_inputs, tgt_test_inputs)


    src_inputs = tf.convert_to_tensor(src_train_inputs)
    tgt_inputs = tf.convert_to_tensor(tgt_train_inputs)

    model = RNN(src_vocab_size, tgt_vocab_size, hidden_size=256, window_size=20)
    output = model(src_inputs[:100], tgt_inputs[:100])
    print(output.shape)
    print(src_vocab_size)

def parse_args():
    return None

if __name__ == '__main__':

    build_model(parse_args())


