import numpy as np
import tensorflow as tf


class DishIngredientPredictorModel(tf.keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    @tf.function
    def call(self, dish_names, ingredient_names):
        encodeded_dish_names = self.encoder(dish_names)
        output = self.decoder(encodeded_dish_names, ingredient_names)
        return output

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
                loss = self.loss_function(decoder_labels, predictions)
                accuracy = self.accuracy_function(decoder_labels, predictions)
                perplexity = self.perplexity_function(decoder_labels, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            total_loss += loss
            total_seen += 1
            total_correct += accuracy
            avg_loss = total_loss / total_seen
            avg_acc = total_correct / total_seen
            avg_prp = perplexity
            print(f'Batch {index+1}/{num_batches} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f} - prp: {avg_prp:.4f}')
