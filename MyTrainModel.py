import NetModel_CNN_NN_SOFTMAX as mymodel
import  tensorflow as tf
import  math as math
import input_data
import numpy as np
import  os as os

Train_it_Continue=False;    #----是否接着训练，True表示接着，否则重新训练

#constant define
Image_Depth=1;          #图像深度(通道数)
ImageSize_Width=28;        #图片宽度
ImageSize_Height=28;       #图片高度
Image_Classes=10;          #图片种类(预期的输出宽度)
Train_Image_Batch=100;           #每次训练的图片数（数据长度）
Test_Image_Batch=1000;           #每次训练的图片数（数据长度）
TrainStepCount=1000;       #训练迭代次数

Train_rate=0.01;        #定义学习率,要合适，否则会不收敛
update_train_data=True;
update_test_data=False;

p_keep_cnn=0.5;

#load dataset
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True);

#define input& correct output
x = tf.placeholder(tf.float32, [None,ImageSize_Width, ImageSize_Height,1],name='x');
y_ = tf.placeholder(tf.float32, [None, Image_Classes],name='y_');  # input correct label

#---------SET model
CNN_parameters=[];
CNN_parameters.append(mymodel.NetModel_CNN_NN_SOFTMAX.CNN_Parameter(conv_filter_size=3,conv_stride=1,conv_output_depth=6,withmaxpool=True,maxpool_kernel_size=2,maxpool_stride=1,withdropout=False,p_keep=0.8));
CNN_parameters.append(mymodel.NetModel_CNN_NN_SOFTMAX.CNN_Parameter(conv_filter_size=3,conv_stride=1,conv_output_depth=12,withmaxpool=True,maxpool_kernel_size=2,maxpool_stride=2,withdropout=False,p_keep=0.8));
CNN_parameters.append(mymodel.NetModel_CNN_NN_SOFTMAX.CNN_Parameter(conv_filter_size=3,conv_stride=1,conv_output_depth=24,withmaxpool=True,maxpool_kernel_size=2,maxpool_stride=2,withdropout=False,p_keep=0.8));
CNN_parameters.append(mymodel.NetModel_CNN_NN_SOFTMAX.CNN_Parameter(conv_filter_size=3,conv_stride=1,conv_output_depth=24,withmaxpool=True,maxpool_kernel_size=2,maxpool_stride=2,withdropout=False,p_keep=0.8));
NN_parameters=[];
NN_parameters.append(mymodel.NetModel_CNN_NN_SOFTMAX.NN_parameter(output_dimension=100,withdropout=False,p_keep=0.8));

y=mymodel.NetModel_CNN_NN_SOFTMAX(CNN_parameters,NN_parameters,x,Image_Classes).CreateModel();


#----------caculate the cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y),name='cross_entropy');

#----------evaluate the model
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32),name='accuracy');
predict=tf.argmax(y,1,name='predict');     #---预测结果
actual_y=tf.argmax(y_,1,name='actual_y');     #---预测结果

#------train the nn with AdamOptimizer method
# variable learning rate
lr = tf.placeholder(tf.float32);
# step for variable learning rate
step = tf.placeholder(tf.int32);
lr = 0.00001 +  tf.train.exponential_decay(0.01, step, TrainStepCount, 1/math.e,staircase=False);
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy,name='train_step');



#-------------------save the model
ckpt_dir="./ckpt_dir";
if not os.path.exists(ckpt_dir):
   os.makedirs(ckpt_dir);
#global_step=tf.Variable(0,name='global_step',trainable=False);
saver=tf.train.Saver();
#non_storable_variable=tf.Variable(777);


# Train
init=tf.initialize_all_variables();             #-----------此处变量初始化一定要放在训练之前，和所有变量之后
with tf.Session() as sess:
   sess.run(init);

   #------------------------if the model has been saved ,here load the trained model,and continue to train it.
   if Train_it_Continue:
      ckpt = tf.train.get_checkpoint_state(ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
         print("已加载当前模型：", ckpt.model_checkpoint_path);
         print("继续训练当前模型：");
         saver.restore(sess, ckpt.model_checkpoint_path)  # 加载所有的参数



   for i in range(TrainStepCount):
      batch_xs, batch_ys = mnist.train.next_batch(Train_Image_Batch);
      batch_xt, batch_yt = mnist.test.next_batch(Test_Image_Batch);
   #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys});         #-------------with GradientDescentOptimizer method
   #   train,acc_on_train=sess.run((train_step,accuracy),feed_dict={x: batch_xs, y_: batch_ys,step:i});     #--------with AdamOptimizer method
    #  train, acc_on_train = sess.run((train_step, accuracy), feed_dict={x:np.reshape(batch_xs,newshape=[-1,ImageSize_Width,ImageSize_Height,Image_Depth])  , y_: np.reshape(batch_ys,newshape=[-1,Image_Classes])});
      train, acc_on_train = sess.run((train_step, accuracy), feed_dict={x: np.reshape(batch_xs, newshape=[-1, ImageSize_Width, ImageSize_Height, Image_Depth]), y_: np.reshape(batch_ys, newshape=[-1, Image_Classes]),step:i});

      if i%10==0:
         # Test trained model
         acc_on_test = sess.run(accuracy,feed_dict={x:np.reshape(batch_xt,newshape=[-1,ImageSize_Width,ImageSize_Height,Image_Depth])  , y_: np.reshape(batch_yt,newshape=[-1,Image_Classes])});
         # compute training values for visualisation
         print("训练周期：",i,"训练集准确度：",acc_on_train,"测试集准确度：",acc_on_test);

         if(acc_on_train>=0.98 and acc_on_test>=0.98):
            # save the model
            saver.save(sess,ckpt_dir+"/model.ckpt",global_step=i);      #----------保存参数
            tf.train.write_graph(sess.graph_def, '/tfmodel', 'train.pbtxt')
            print("模型训练完成!");
            break;





