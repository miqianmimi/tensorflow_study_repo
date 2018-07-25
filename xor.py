# Question: Write a XOR model to prevent the result of a XOR b
# Question: Write your own metrics, loss and optimizer
import tensorflow as tf
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


class XOR(tf.keras.Model):
    def __init__(self):
        super(XOR, self).__init__(self)
        self.dense1 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, input_shape=(2, ))
        self.dense2 = tf.keras.layers.Dense( units=1, activation=tf.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


def accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.to_float(tf.equal(y_true,tf.round(y_pred))))

def loss(y_true, y_pred):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_pred, logits=y_true)
    return loss


class Optimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate):
        super(Optimizer, self).__init__()
        self._learning_rate = learning_rate

    def get_updates(self, loss, params):
        updates = []
        grads = tf.gradients(loss, params)
        for p, g in zip(params, grads):
            v = self._learning_rate*g
            op = tf.assign_sub(p, v)
            updates.append(op)
        # TODO: Implement here
        return updates


def run():
    m = XOR()
    m.compile(loss=loss, optimizer=Optimizer(0.01), metrics=[accuracy])
    m.fit(X, y, epochs=10)
    print(m.predict(X))


if __name__ == "__main__":
    run()
