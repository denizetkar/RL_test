import tensorflow as tf
import numpy as np
import gym

game_name = 'Taxi-v2'
env = gym.make(game_name)
Q_table = np.loadtxt(game_name + ".csv", delimiter=';')
s = env.reset()
env.render()
while True:
    a = np.argmax(Q_table[s, :])
    s1, r, d, _ = env.step(a)
    env.render()
    if d is True:
        break
    s = s1


class Giver:
    def __init__(self):
        self.x = 0

    def give(self):
        if self.x < 10:
            retval = self.x
            self.x += 1
            return retval
        else:
            return None

    def reset(self, val=0):
        self.x = val


giver_obj = Giver()


def gen():
    giver_obj.reset(0)
    while True:
        retval = giver_obj.give()
        if retval is None:
            break
        else:
            yield retval


ds = tf.data.Dataset.from_generator(gen, tf.int32, ())
iterator = ds.make_initializable_iterator()
ds_val = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    try:
        while True:
            print(sess.run(ds_val))
    except tf.errors.OutOfRangeError:
        pass
    sess.run(iterator.initializer)
    try:
        while True:
            print(sess.run(ds_val))
    except tf.errors.OutOfRangeError:
        pass


def my_gen():
    x = [(1, 1), (2, 2), (3, 3)]
    y = [0.5, 1.5, 2.5]
    yield from zip(np.array(x), np.array(y))


ds = tf.data.Dataset.from_generator(my_gen, (tf.int32, tf.float32), ((2,), ()))
for value in ds:
    print(value)
