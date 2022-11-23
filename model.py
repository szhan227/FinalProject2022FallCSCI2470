import numpy as np
import tensorflow as tf


class DishIngredientPredictorModel(tf.keras.Model):

    def __init__(self, predictor, **kwargs):
        super().__init__(**kwargs)
        self.predictor = predictor


    @tf.function
    def call(self, dish_names, ingredient_names):
        return self.predictor(dish_names, ingredient_names)


    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]

    def train(self, train_ingredients, train_dishes, padding_index, batch_size=100):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0

        num_batches = max(1, int(len(train_ingredients) / batch_size))

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_ingredients)+1, batch_size)):
            start = end - batch_size
            batch_ingredients = train_ingredients[start:end]
            decoder_input = train_dishes[start:end, :-1]
            decoder_labels = train_dishes[start:end, 1:]
            with tf.GradientTape() as tape:
                predictions = self.call(decoder_input, batch_ingredients)
                mask = decoder_labels != padding_index
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
            print(f'\rTrain {index+1}/{num_batches} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f} - prp: {avg_prp:.4f}', end='')

        print()

        return avg_loss, avg_acc, avg_prp


def accuracy_function(prbs, labels, mask):

    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):

    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

