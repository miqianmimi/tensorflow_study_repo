import tensorflow as tf


def f(x): return 3*x


x_init = 1
y_expected = 5
EPOCHS = 30


def find_root(f, x_init, y_expected, learning_rate):
    x = tf.get_variable(
        'x', shape=[1], initializer=tf.constant_initializer(x_init))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(EPOCHS):
            y_predict = f(x)
            loss = (y_predict-y_expected)*(y_predict-y_expected)
            y_grad = tf.gradients(loss, [x])
            op = tf.assign_sub(x, y_grad[0]*learning_rate)
            op = tf.Print(op, [y_predict, y_expected,loss], "This is what i want: ")
            sess.run(op)

        return sess.run(x)


if __name__ == "__main__":
    find_root(f, x_init, y_expected, learning_rate=0.03)
