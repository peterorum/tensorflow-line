#!/usr/local/bin/python3

import tensorflow as tf
import numpy as np

np.random.seed(30)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input, output & params
m = tf.Variable([1.0], tf.float32)
b = tf.Variable([1.0], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# model
prediction = m * x + b

# loss
loss = tf.reduce_sum(tf.square(prediction - y))

# optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
# expected
M = 30
B = -27
R = 3  # random perturb

X_input = [0, 1, 2, 3, 4]
# calc result
y_result = [x * M - B + np.random.uniform(-R, R) for x in X_input]

# params
epochs = 1000

# session to run our code
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train our model on training data
    for epoch in range(epochs):
        sess.run(train, {x: X_input, y: y_result})

    # results estimates
    est_m, est_b = sess.run([m, b])

    print("Value of m is %s and value of b is %s." % (est_m, est_b))
