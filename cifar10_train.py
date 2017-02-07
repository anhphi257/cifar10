import cifar10
import tensorflow as tf
import input_data
MAX_STEPS = 1000000

def trainning():
    (X, Y), (X_test, Y_test) = cifar10.load_data()
    Y = cifar10.to_categorical(Y, 10)
    Y_test = cifar10.to_categorical(Y_test, 10)
    data_set = cifar10.read_data_sets(X, Y, X_test, Y_test)
    # mnist = input_data.read_data_sets("tmp/mnist", one_hot=True)
    # batch_x, batch_y = data_set.train.next_batch(96)

    x_placeholder = tf.placeholder("float", [None, 32 * 32 * 3])
    y_placeholder = tf.placeholder("float", [None, 10])

    logits = cifar10.inference(x_placeholder)
    loss = cifar10.loss(logits, y_placeholder)
    train_op = cifar10.train_op(loss=loss, learning_rate=0.001)
    accuracy = cifar10.accuracy(logits, y_placeholder)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(MAX_STEPS):
            # print('step = {:d}'.format(step + 1))
            batch_x, batch_y = data_set.train.next_batch(96)
            # print(batch_x.shape)
            # print(batch_y.shape)
            _, Loss, acc = sess.run([train_op, loss, accuracy], feed_dict={x_placeholder: batch_x,
                                                                           y_placeholder: batch_y})
            if (step + 1) % 100 == 0:
                print("step: {:d} loss: {:f} acc: {:f}".format(step + 1, Loss, acc))
            # print('loss = ')
            # print(Loss)

if __name__ == '__main__':
    trainning()

