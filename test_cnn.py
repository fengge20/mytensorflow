# Import data
#from tensorflow.examples.tutorials.mnist import input_data
import input_data
import tensorflow as tf
import  math as math
import  numpy as np

#constant define
Image_Depth=1;          #图像深度(通道数)
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

p_keep_cnn=0.75;
#load dataset
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

#define input
x = tf.placeholder(tf.float32, [None,ImageSize_Width, ImageSize_Height,1]);

####-----------------create the CNN
class parameter:
    def __init__(self,kernel_size=5,stride=1,output_depth=2):
        self.kernel_size=kernel_size;
        self.stride=stride;
        self.output_depth=output_depth;

k=[parameter(3,1,12),parameter(3,2,24),parameter(3,2,48)];       #cnn层各自的输出通道数
n=200;      #CNN后面的全连接层

#---------------添加cnn相关层的参数
image_size = 28;  # 初始图像大小(如果加入了cnn则会变化，表示最后一层图像大小，从而构造后面的全连接层)
if len(k)!=0:
    cnn_layers_number=len(k);
    w = [];
    b = [];
    y_middle = [];
    for index in range(cnn_layers_number):
        if index == 0:
            w.append(tf.Variable(tf.truncated_normal([k[index].kernel_size,k[index].kernel_size,Image_Depth,k[index].output_depth], stddev=0.1)));
            b.append(tf.Variable(tf.zeros(k[index].output_depth) / 10));
            temp_nn = tf.nn.relu(tf.nn.conv2d(x, w[index], strides=[1, 1, 1, 1], padding='SAME') + b[index]);               #-----卷积层
            temp = tf.nn.max_pool(temp_nn,ksize=[1, k[index].stride, k[index].stride, 1], strides=[1, k[index].stride, k[index].stride, 1], padding='SAME')  #-----最大池化层
        #    temp=tf.nn.dropout(temp,p_keep_cnn);       #-----------增加dropout
        else:
            w.append(tf.Variable(tf.truncated_normal([k[index].kernel_size,k[index].kernel_size,k[index-1].output_depth,k[index].output_depth], stddev=0.1)));
            b.append(tf.Variable(tf.zeros(k[index].output_depth) / 10));
            temp_nn = tf.nn.relu(tf.nn.conv2d(y_middle[index-1], w[index], strides=[1,1,1, 1], padding='SAME') + b[index]);     #-----卷积层
            temp = tf.nn.max_pool(temp_nn, ksize=[1, k[index].stride, k[index].stride, 1],strides=[1, k[index].stride, k[index].stride, 1], padding='SAME');    #-----最大池化层
         #   temp = tf.nn.dropout(temp, p_keep_cnn);    #-----------增加dropout
        y_middle.append(temp);
        image_size=(int)(image_size/ k[index].stride);
    data_dimension = (int)(image_size * image_size * k[cnn_layers_number - 1].output_depth);
    yy=tf.reshape(y_middle[cnn_layers_number-1],shape=[-1,data_dimension]);  #---转换为1维数据
else:
    yy = tf.reshape(x,shape=[-1, image_size * image_size * Image_Depth]);  # ---转换为1维数据
    data_dimension=(int)(image_size * image_size * Image_Depth);

#-------加入全连接层
w1=tf.Variable(tf.truncated_normal([data_dimension,n],stddev=0.1));
b1=tf.Variable(tf.zeros([n])/10);
y1=tf.nn.relu(tf.matmul(yy,w1)+b1);
#y1 = tf.nn.dropout(y1, p_keep_cnn);  # -----------增加dropout
#-------加入softmax层
w2=tf.Variable(tf.truncated_normal([n,Image_Classes],stddev=0.1));
b2=tf.Variable(tf.zeros([Image_Classes])/10);
y_logits=tf.matmul(y1,w2)+b2;
y=tf.nn.softmax(y_logits);

##----------------------------------------上面完成了模型搭建
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, Image_Classes]);
cross_entropy = -tf.reduce_sum(y_ * tf.log(y));
#------train the nn with GradientDescentOptimizer method
# train_step = tf.train.GradientDescentOptimizer(Train_rate).minimize(cross_entropy);

#------train the nn with AdamOptimizer method
# variable learning rate
lr = tf.placeholder(tf.float32);
# step for variable learning rate
step = tf.placeholder(tf.int32);
lr = 0.0001 +  tf.train.exponential_decay(0.01, step, 2000, 1/math.e,);
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy);
#
# #------train the nn with RMSPropOptimizer method
#train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy);

# #------train the nn with RMSPropOptimizer method
#train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy);



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
   #   train,acc_on_train=sess.run((train_step,accuracy),feed_dict={x: batch_xs, y_: batch_ys,step:i});     #--------with AdamOptimizer method
    #  train, acc_on_train = sess.run((train_step, accuracy), feed_dict={x:np.reshape(batch_xs,newshape=[-1,ImageSize_Width,ImageSize_Height,Image_Depth])  , y_: np.reshape(batch_ys,newshape=[-1,Image_Classes])});
      train, acc_on_train = sess.run((train_step, accuracy), feed_dict={x: np.reshape(batch_xs, newshape=[-1, ImageSize_Width, ImageSize_Height, Image_Depth]), y_: np.reshape(batch_ys, newshape=[-1, Image_Classes]),step:i});

      # Test trained model
      acc_on_test = sess.run(accuracy,feed_dict={x:np.reshape(batch_xt,newshape=[-1,ImageSize_Width,ImageSize_Height,Image_Depth])  , y_: np.reshape(batch_yt,newshape=[-1,Image_Classes])});

      # compute training values for visualisation
      print("训练周期：",i,"训练集准确度：",acc_on_train,"测试集准确度：",acc_on_test);

