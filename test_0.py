#!/usr/bin/env python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

Guide web
http://blog.csdn.net/nxcxl88/article/details/52719325
"""
# Import data
#from tensorflow.examples.tutorials.mnist import input_data
import input_data
import tensorflow as tf


#constant define
ImageSize_Width=28;        #图片宽度
ImageSize_Height=28;       #图片高度
Image_Classes=10;          #图片种类(预期的输出宽度)
Train_Image_Batch=100;           #每次训练的图片数（数据长度）
Test_Image_Batch=1000;           #每次训练的图片数（数据长度）
TrainStepCount=1000;       #训练迭代次数
Data_Dimension=ImageSize_Width*ImageSize_Height;      #数据维度，此处将图片堆叠成一维数组
Train_rate=0.01;        #定义学习率,要合适，否则会不收敛
update_train_data=True;
update_test_data=False;



#load dataset
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)



# Create the model
x = tf.placeholder(tf.float32, [None, Data_Dimension])
w = tf.Variable(tf.zeros([Data_Dimension, Image_Classes]))
b = tf.Variable(tf.zeros([Image_Classes]))

init=tf.initialize_all_variables();

y = tf.nn.softmax(tf.matmul(x, w) + b)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, Image_Classes])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(Train_rate).minimize(cross_entropy)

is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))





# Train

with tf.Session() as sess:
   sess.run(init);
   for i in range(TrainStepCount):
      batch_xs, batch_ys = mnist.train.next_batch(Train_Image_Batch);
      batch_xt, batch_yt = mnist.test.next_batch(Test_Image_Batch);
   #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});
      train,acc_on_train=sess.run((train_step,accuracy),feed_dict={x: batch_xs, y_: batch_ys});
      acc_on_test = sess.run(accuracy, feed_dict={x:batch_xt, y_: batch_yt});
      # Test trained model

      # compute training values for visualisation


      print("训练集准确度：",acc_on_train,"测试集准确度：",acc_on_test);

