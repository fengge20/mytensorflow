import  tensorflow as tf
import matplotlib.pyplot as plt     # plt 用于显示图片
import math as math    #
import numpy as np
import input_data
import time as time
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def caculate_factor(number):
    minus=number;
    width=1;
    height=1
    for i in range(1,number+1):
         if number%i==0:
            temp=math.fabs(i-number/i);
            if(temp<minus):
                minus=temp;
                width=i;
                height=number/i;

    return width,height;

def show_iamge(input_x,predict,actual,name='figure_0'):

  #  if input_x==None or len(input_x)==0:return;

  #  image_x_batch=np.reshape(input_x,newshape=[-1,28,28,1]);

    plt.figure(name);
    show_width_num,show_height_num=caculate_factor(input_x.shape[0]);      #----获取最优显示布局

    for index in range(input_x.shape[0]):
        image_x=input_x[index,:,:,:];

        #---------------in order to show the image ,do some ops
        pinjie=np.zeros([image_x.shape[0],image_x.shape[1],2]);
        image = np.concatenate((image_x, pinjie), axis=2);

        plt.subplot(show_width_num, show_height_num, index+1);
        plt.title("predict:" + str(predict[index]) + "; actual:" + str(actual[index]));
        plt.imshow(image,cmap='Greys_r'), plt.axis('off')   ;# 显示图片

    plt.show();

def show_CNN_iamge(input_x,predict,actual,name='figure_0'):

  #  if input_x==None or len(input_x)==0:return;

  #  image_x_batch=np.reshape(input_x,newshape=[-1,28,28,1]);
    for index in range(input_x.shape[0]):
        show_width_num, show_height_num = caculate_factor(input_x.shape[3]);  # ----获取最优显示布局
        plt.figure(name);
        image_x=input_x[index,:,:,:];
        for innercount in range(input_x.shape[3]):
            # ---------------in order to show the image ,do some ops
            pinjie = np.zeros([image_x.shape[0], image_x.shape[1], 2]);
            image = np.concatenate((image_x[:,:,innercount:innercount+1], pinjie), axis=2);

            plt.subplot(show_width_num, show_height_num, innercount + 1);
            plt.title("predict:" + str(predict[index]) + "; actual:" + str(actual[index])+";kernel number:"+str(innercount));
            plt.imshow(image, cmap='Greys_r'), plt.axis('off');  # 显示图片
        plt.show();


#------------------------------------------实际的测试程序部分--------------------------------
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True);
ckpt_dir = "./ckpt_dir";
meta_dir="./ckpt_dir/model.ckpt-500.meta"
#--------------加载模型----
ckpt = tf.train.get_checkpoint_state(ckpt_dir);
if ckpt and ckpt.model_checkpoint_path:
  #  a=
    batch_xt, batch_yt = mnist.test.next_batch(2);

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_dir);
        saver.restore(sess, ckpt.model_checkpoint_path);  # 加载所有的参数
        print("已加载当前模型：", ckpt.model_checkpoint_path);
     #   print("=========All Variables==========")
     #   print_tensors_in_checkpoint_file(ckpt_dir+'/model.ckpt-200', tensor_name=None, all_tensors=True,all_tensor_names=True)

        x=tf.get_default_graph().get_tensor_by_name('x:0');
        y_ = tf.get_default_graph().get_tensor_by_name('y_:0');
        predict = tf.get_default_graph().get_tensor_by_name('predict:0');
        actual_y = tf.get_default_graph().get_tensor_by_name('actual_y:0');

        y_cnn_pool1=tf.get_default_graph().get_tensor_by_name('y_cnn_pool1:0');
        y_cnn_pool2 = tf.get_default_graph().get_tensor_by_name('y_cnn_pool2:0');
        y_cnn_pool3 = tf.get_default_graph().get_tensor_by_name('y_cnn_pool3:0');
        y_cnn_pool4 = tf.get_default_graph().get_tensor_by_name('y_cnn_pool4:0');
        # y_cnn1=tf.get_default_graph().get_tensor_by_name('y_cnn1:0');
        # y_cnn2 = tf.get_default_graph().get_tensor_by_name('y_cnn2:0');
        # y_cnn3 = tf.get_default_graph().get_tensor_by_name('y_cnn3:0');
        print("已加载各输入输出节点!");

        time1=time.time();
        y_cnn_pool1_result,y_cnn_pool2_result,y_cnn_pool3_result,y_cnn_pool4_result,predict_result,actual_result = sess.run((y_cnn_pool1,y_cnn_pool2,y_cnn_pool3,y_cnn_pool4,predict,actual_y), feed_dict={x:np.reshape(batch_xt,newshape=[-1,28,28,1]), y_: np.reshape(batch_yt,newshape=[-1,10])});
        time2=time.time();
        runtime=time2-time1;
        print("已检测完成，下面显示结果!","检测消耗时间：",runtime);

        show_iamge(np.reshape(batch_xt,newshape=[-1,28,28,1]),predict_result,actual_result,name="predict result");   #------------以图片的方式显示结果

        show_CNN_iamge(y_cnn_pool1_result, predict_result, actual_result,name="CNN_1");  # ------------以图片的方式显示卷积层后池化结果
        show_CNN_iamge(y_cnn_pool2_result, predict_result, actual_result,name="CNN_2");  # ------------以图片的方式显示卷积层后池化结果
        show_CNN_iamge(y_cnn_pool3_result, predict_result, actual_result,name="CNN_3");  # ------------以图片的方式显示卷积层后池化结果
        show_CNN_iamge(y_cnn_pool4_result, predict_result, actual_result, name="CNN_4");  # ------------以图片的方式显示卷积层后池化结果
     #   print("检测结果:",predict_result,";实际输入：",actual_result);


