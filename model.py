import numpy as np
import tensorflow as tf

import transformer
from preprocess import truncate


class DishIngredientPredictorModel(tf.keras.Model):

    def __init__(self, predictor, src_w2i, src_i2w, tgt_w2i, tgt_i2w, **kwargs):
        super().__init__(**kwargs)
        self.predictor = predictor
        self.src_w2i = src_w2i
        self.src_i2w = src_i2w
        self.tgt_w2i = tgt_w2i
        self.tgt_i2w = tgt_i2w


    @tf.function
    def call(self, dish_names, ingredient_names, src_padding_mask=None, tgt_padding_mask=None):
        print('dish_names shape', dish_names.shape)
        print('ingredient_names shape', ingredient_names.shape)
        print('predictor type', type(self.predictor))
        return self.predictor(dish_names, ingredient_names, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

    def predict(self, dish_names):
        if type(self.predictor) == transformer.Transformer:
            dish_names = truncate(dish_names, self.predictor.window_size - 1)
        tokens = [self.src_w2i[word] for word in dish_names]
        src_token = tf.convert_to_tensor(tokens)

        # to make src_token a 3D tensor [batch, window, embedding],
        # where here batch is 1, since we only predict one dish.
        src_token = tf.expand_dims(src_token, axis=0)

        tgt_token = self.predict_token(src_token)
        lst = []
        for sentence in tgt_token:
            for each in sentence:
                lst.append(self.tgt_i2w[tf.get_static_value(each)])

        return ', '.join(lst[1:])

    def predict_token(self, src_tokens):
        num_tokens = src_tokens.shape[0]
        tgt_token = self.greedy_decode(src_tokens, max_len=num_tokens + 5)
        return tgt_token

    def encode(self, src_tokens):
        print('show predictor type', type(self.predictor))
        return self.predictor.encode(src_tokens)

    def decode(self, tgt_inputs, encoder_state):
        return self.predictor.decode(tgt_inputs, encoder_state)

    def greedy_decode(self, src_tokens, max_len, start_symbol='<start>', end_symbol='<end>'):

        if False:
        # if type(self.predictor) == transformer.Transformer:

            hidden_state = self.encode(src_tokens)
            pad_idx = self.tgt_w2i['<pad>']
            sentence = [self.tgt_w2i[start_symbol]]

            for i in range(max_len):
                sentence_pad = truncate(sentence, self.predictor.window_size - 1)
                for i, word in enumerate(sentence_pad):
                    if word == '<pad>':
                        sentence_pad[i] = pad_idx
                # print('sentence_pad', sentence_pad)
                ys = tf.convert_to_tensor([sentence_pad])
                out = self.decode(ys, hidden_state)
                # out = out.transpose(1, 0, 2)
                next_word = tf.math.argmax(out[:, -1], axis=1, output_type=tf.int32)
                next_word = tf.get_static_value(next_word[0])
                sentence += [next_word]
                if next_word == pad_idx:
                    break
            return ys

        else:
            hidden_state = self.encode(src_tokens)
            # hidden_output, hidden_state = self.encode(src_tokens)

            sentence = [self.tgt_w2i[start_symbol]]
            ys = tf.convert_to_tensor([sentence])
            for i in range(max_len):
                out = self.decode(ys, hidden_state)
                # out = out.transpose(1, 0, 2)
                next_word = tf.math.argmax(out[:, -1], axis=1, output_type=tf.int32)
                ys = tf.concat([ys, tf.expand_dims(next_word, axis=1)], axis=1)
                if self.tgt_i2w[tf.get_static_value(next_word[0])] == end_symbol:
                    break
            return ys

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]

    def train(self, train_ingredients, train_dishes, src_padding_index, tgt_padding_index, batch_size=100):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0

        num_batches = max(1, int(len(train_ingredients) / batch_size))

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_ingredients)+1, batch_size)):
            start = end - batch_size
            batch_dishes = train_dishes[start:end, :-1]
            decoder_input = train_ingredients[start:end, :-1]
            decoder_labels = train_ingredients[start:end, 1:]
            src_padding_mask = tf.cast(tf.math.equal(batch_dishes, src_padding_index), tf.float32)
            tgt_padding_mask = tf.cast(tf.math.equal(decoder_input, tgt_padding_index), tf.float32)
            with tf.GradientTape() as tape:
                predictions = self.call(batch_dishes, decoder_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                mask = decoder_labels != tgt_padding_index
                loss = self.loss_function(predictions, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            accuracy = self.accuracy_function(predictions, decoder_labels, mask)

            total_loss += loss
            total_seen += num_predictions
            total_correct += accuracy

            avg_loss = total_loss / total_seen
            avg_acc = total_correct / total_seen
            avg_prp = np.exp(avg_loss)
            print(f'\rTrain {index+1}/{num_batches} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - prp: {avg_prp:.4f}', end='')

        print()

        return avg_loss, avg_acc, avg_prp


def accuracy_function(prbs, labels, mask):
    correct_classes = tf.math.argmax(prbs, axis=-1, output_type=tf.int32) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):

    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

