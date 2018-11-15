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
import  math as math


#constant define
ImageSize_Width=28;        #图片宽度
ImageSize_Height=28;       #图片高度
Image_Classes=10;          #图片种类(预期的输出宽度)
Train_Image_Batch=100;           #每次训练的图片数（数据长度）
Test_Image_Batch=1000;           #每次训练的图片数（数据长度）
TrainStepCount=1000;       #训练迭代次数

Data_Dimension=ImageSize_Width*ImageSize_Height;      #数据维度，此处将图片堆叠成一维数组
MiddleLayer_Dimension=[200];          #中间层数量，此处设置为不确定

Train_rate=0.01;        #定义学习率,要合适，否则会不收敛
update_train_data=True;
update_test_data=False;



#load dataset
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True);



# -----------------------------------------------Create the model
x = tf.placeholder(tf.float32, [None, Data_Dimension]);
# feed in 1 when testing, 0.75 when training
keep=0.75;
pkeep = tf.placeholder(tf.float32);

#layer 1
if len(MiddleLayer_Dimension)!=0:       #------------------------表示是多层网络
    layer_number=len(MiddleLayer_Dimension);
    w=[];
    b=[];
    y_middle=[];
    for index in range(layer_number):
        if index==0:
            w.append( tf.Variable(tf.truncated_normal([Data_Dimension, MiddleLayer_Dimension[index]],stddev=0.1)));
            b.append( tf.Variable(tf.zeros([MiddleLayer_Dimension[index]])/10));
            temp= tf.nn.relu(tf.matmul(x, w[index]) + b[index]);
      #      Y1d = tf.nn.dropout(temp, pkeep);                                                   #----dropout
            y_middle.append(temp);

        else:
            w.append(tf.Variable(tf.truncated_normal([MiddleLayer_Dimension[index-1], MiddleLayer_Dimension[index]],stddev=0.1)));
            b.append(tf.Variable(tf.zeros([MiddleLayer_Dimension[index]])/10));
            y_middle.append(tf.nn.relu(tf.matmul(y_middle[index-1], w[index]) + b[index]));

    w_final = tf.Variable(tf.truncated_normal([MiddleLayer_Dimension[layer_number-1], Image_Classes],stddev=0.1));  # ---------------此处的随机化赋值很重要
    b_final = tf.Variable(tf.zeros([Image_Classes])/10);
    y_logits=tf.matmul(y_middle[layer_number-1], w_final) + b_final;
    y = tf.nn.softmax(y_logits);
else:           #---------表示是单层网络
    w1 = tf.Variable(tf.truncated_normal([Data_Dimension, Image_Classes],stddev=0.1)) ; # ---------------此处的随机化赋值很重要
    b1 = tf.Variable(tf.zeros([Image_Classes])/10);
    y_logits=tf.matmul(x, w1) + b1;
    y = tf.nn.softmax(y_logits);








# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, Image_Classes]);
cross_entropy = -tf.reduce_sum(y_ * tf.log(y));
#------train the nn with GradientDescentOptimizer method
train_step = tf.train.GradientDescentOptimizer(Train_rate).minimize(cross_entropy);

#------train the nn with AdamOptimizer method
# variable learning rate
# lr = tf.placeholder(tf.float32);
# # step for variable learning rate
# step = tf.placeholder(tf.int32);
# lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e);
# train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy);


#####-----------------------define the evaluation factors
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32));





# Train
init=tf.initialize_all_variables();             #-----------此处变量初始化一定要放在训练之前，和所有变量之后
with tf.Session() as sess:
   sess.run(init);
   for i in range(TrainStepCount):
      batch_xs, batch_ys = mnist.train.next_batch(Train_Image_Batch);
      batch_xt, batch_yt = mnist.test.next_batch(Test_Image_Batch);
   #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});         #-------------with GradientDescentOptimizer method
  #    train,acc_on_train=sess.run((train_step,accuracy),feed_dict={x: batch_xs, y_: batch_ys,step:i});     #--------with AdamOptimizer method
      train, acc_on_train = sess.run((train_step, accuracy), feed_dict={x: batch_xs, y_: batch_ys});

      # Test trained model
      acc_on_test = sess.run(accuracy, feed_dict={x:batch_xt, y_: batch_yt});

      # compute training values for visualisation
      print("训练周期：", i, "训练集准确度：", acc_on_train, "测试集准确度：", acc_on_test);

