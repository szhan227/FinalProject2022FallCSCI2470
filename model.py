import numpy as np
import tensorflow as tf


class DishIngredientPredictorModel(tf.keras.Model):

    def __init__(self, predictor, src_w2i, src_i2w, tgt_w2i, tgt_i2w, **kwargs):
        super().__init__(**kwargs)
        self.predictor = predictor
        self.src_w2i = src_w2i
        self.src_i2w = src_i2w
        self.tgt_w2i = tgt_w2i
        self.tgt_i2w = tgt_i2w


    @tf.function
    def call(self, dish_names, ingredient_names):
        return self.predictor(dish_names, ingredient_names)

    def predict(self, dish_names):
        tokens = [self.src_w2i[word] for word in dish_names]
        print(tokens)
        src_token = tf.convert_to_tensor(tokens)
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
        return self.predictor.encode(src_tokens)

    def decode(self, tgt_inputs, encoder_state):
        return self.predictor.decode(tgt_inputs, encoder_state)

    def greedy_decode(self, src_tokens, max_len, start_symbol='<start>', end_symbol='<end>'):
        hidden_output, hidden_state = self.encode(src_tokens)
        ys = tf.convert_to_tensor([[self.tgt_w2i[start_symbol]]])
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
            batch_dishes = train_dishes[start:end]
            decoder_input = train_ingredients[start:end, :-1]
            decoder_labels = train_ingredients[start:end, 1:]
            with tf.GradientTape() as tape:
                predictions = self.call(batch_dishes, decoder_input)
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

