# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np 

#构造一元二次方程
x_data=np.linspace(-1,1,300)[:,np.newaxis]    # np.newaxis为None的别名
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

# palceholder
xs= tf.placeholder(tf.float32, [None,1])
ys= tf.placeholder(tf.float32, [None,1])

# define layer
def add_layer(inputs, in_size ,out_size ,activation_function=None):
  weights= tf.Variable(tf.random_normal([in_size,out_size]))
  biases= tf.Variable(tf.zeros([1,out_size])+0.1)
  Wx_plus_b=tf.matmul(inputs,weights)+biases
  if activation_function is None:
    ouputs= Wx_plus_b
  else:
    ouputs=activation_function(Wx_plus_b)
  return ouputs

# hide layer
h1=add_layer(xs,1,20,activation_function=tf.nn.relu)
# ouput layer
prediction= add_layer(h1,20,1,activation_function=None)

# loss function
loss= tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step= tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# model
init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for i in range(1000):
  sess.run(train_step,feed_dict={
    xs:x_data,
    ys:y_data
    })
  if i%10==0:
    print(sess.run(loss,feed_dict={
      xs:x_data,
      ys:y_data
      }))
