
import tensorflow as tf
import  math as math

class NetModel_CNN_NN_SOFTMAX:
    input_data=[-1,28,28,1];

    #------cnn的网络参数
    w=[];       #---Weights
    b=[];       #---Bias
    y_middle=[];    #---output of middle layers
    image_size=28;  #image size

    #----nn的网络参数
    nn_w=[];
    nn_b=[];
    nn_y_middle=[];

    class CNN_Parameter:
        # define the parameters to describe the cnn structure
        def __init__(self, conv_filter_size=4, conv_stride=1, conv_output_depth=2,withmaxpool=False,maxpool_kernel_size=2,maxpool_stride=2,withdropout=False,p_keep=0.8):
            self.conv_filter_size = conv_filter_size;
            self.conv_stride = conv_stride;
            self.conv_output_depth = conv_output_depth;
            #-------if with max_pool
            self.withmaxpool=withmaxpool;
            self.maxpool_kernel_size = maxpool_kernel_size;
            self.maxpool_stride = maxpool_stride
            #-----if with dropout
            self.withdropout = withdropout;
            self.p_keep=p_keep;

        def __iter__(self):
            return  self;

    class NN_parameter:
        def __init__(self,output_dimension=10,withdropout=False,p_keep=0.8):
       #     self.input_dimension=input_dimension;
            self.output_dimension=output_dimension;
            self.withdropout=withdropout;
            self.p_keep=p_keep;

        def __iter__(self):
             return self;


   # create the model by the input parameters: SOME LAYER OF CNNs-------SOME LAYER OF NNs----------ONE LAYER OF SOFTMAX

    def __init__(self,cnnparameters=[],nnparmeters=[],input_data=tf.placeholder(float,[None,28,28,1]),output_classes=10):
        self.CNN_parameters = cnnparameters;
        self.NN_parameters=nnparmeters;
        self.input_data=input_data;
        self.image_size=input_data.shape[1].value;
        self.output_classes=output_classes;

    def CreateModel(self):
        # ---------------construct CNNs
        if self.CNN_parameters!=None and len(self.CNN_parameters)!=0:
            cnn_layers_number = len(self.CNN_parameters);
            for index in range(cnn_layers_number):
                if index == 0:
                    self.w.append(tf.Variable(tf.truncated_normal([self.CNN_parameters[index].conv_filter_size, self.CNN_parameters[index].conv_filter_size,
                                                                   self.input_data.shape[3].value, self.CNN_parameters[index].conv_output_depth], stddev=0.1),name='w_cnn'+str(index+1)));
                    self.b.append(tf.Variable(tf.zeros(self.CNN_parameters[index].conv_output_depth) / 10,name='b_cnn'+str(index+1)));
                    temp = tf.nn.relu(tf.nn.conv2d(self.input_data, self.w[index], strides=[1, self.CNN_parameters[index].conv_stride, self.CNN_parameters[index].conv_stride, 1], padding='SAME') + self.b[index],name='y_cnn'+str(index+1));  # -----卷积层

                else:
                    self.w.append(tf.Variable(tf.truncated_normal([self.CNN_parameters[index].conv_filter_size, self.CNN_parameters[index].conv_filter_size,
                                                                   self.CNN_parameters[index - 1].conv_output_depth, self.CNN_parameters[index].conv_output_depth],stddev=0.1),name='w_cnn'+str(index+1)));
                    self.b.append(tf.Variable(tf.zeros(self.CNN_parameters[index].conv_output_depth) / 10,name='b_cnn'+str(index+1)));
                    temp = tf.nn.relu( tf.nn.conv2d( self.y_middle[index - 1],  self.w[index], strides=[1, self.CNN_parameters[index].conv_stride, self.CNN_parameters[index].conv_stride, 1], padding='SAME') + self.b[index],name='y_cnn'+str(index+1));  # -----卷积层


                self.image_size = math.ceil(self.image_size / self.CNN_parameters[index].conv_stride);           #--------计算经过卷积的尺寸

                if self.CNN_parameters[index].withmaxpool:
                    temp = tf.nn.max_pool(temp, ksize=[1, self.CNN_parameters[index].maxpool_kernel_size, self.CNN_parameters[index].maxpool_kernel_size, 1],
                                              strides=[1, self.CNN_parameters[index].maxpool_stride, self.CNN_parameters[index].maxpool_stride, 1],padding='SAME',name='y_cnn_pool'+str(index+1))  # -----最大池化层
                    self.image_size = math.ceil(self.image_size / self.CNN_parameters[index].maxpool_stride);       #------池化之后图片尺寸
                    #   temp = tf.nn.dropout(temp, p_keep_cnn);    #-----------增加dropout
                if self.CNN_parameters[index].withdropout:
                    temp = tf.nn.dropout(temp, self.CNN_parameters[index].p_keep,name='y_cnn_dropout'+str(index+1))  # -----dropout层

                self.y_middle.append(temp);

            data_dimension =  math.ceil(self.image_size * self.image_size * self.CNN_parameters[cnn_layers_number - 1].conv_output_depth);
            yy = tf.reshape(self.y_middle[cnn_layers_number - 1], shape=[-1, data_dimension]);  # ---转换为1维数据
        else:
            data_dimension = (int)(self.image_size * self.image_size * self.input_data.shape[3].value);
            yy = tf.reshape(self.input_data, shape=[-1,data_dimension]);  # ---转换为1维数据



        # -------加入全连接层
        if self.NN_parameters!=None and len(self.NN_parameters)!=0:
            nn_layers_number = len(self.NN_parameters);
            for index in range(nn_layers_number):
                if index == 0:
                    self.nn_w.append(tf.Variable(tf.truncated_normal([data_dimension, self.NN_parameters[index].output_dimension], stddev=0.1),name='w_nn'+str(index+1)));
                    self.nn_b.append(tf.Variable(tf.zeros([self.NN_parameters[index].output_dimension]) / 10,name='b_nn'+str(index+1)));
                    temp = tf.nn.relu(tf.matmul(yy,   self.nn_w[index]) +  self.nn_b[index],name='y_nn'+str(index+1));

                else:
                    self.nn_w.append(tf.Variable(tf.truncated_normal([self.NN_parameters[index-1].output_dimension, self.NN_parameters[index].output_dimension], stddev=0.1),name='w_nn'+str(index+1)));
                    self.nn_b.append(tf.Variable(tf.zeros([self.NN_parameters[index].output_dimension]) / 10,name='b_nn'+str(index+1)));
                    temp = tf.nn.relu(tf.matmul( self.nn_y_middle[index-1], self.nn_w[index]) + self.nn_b[index],name='y_nn'+str(index+1));

                if self.NN_parameters[index].withdropout:
                    temp = tf.nn.dropout(temp, self.NN_parameters[index].p_keep,name='y_nn_dropout'+str(index+1))  # -----dropout层
                data_dimension = self.NN_parameters[index].output_dimension;
                self.nn_y_middle.append(temp);
        else:
            self.nn_y_middle.append(yy);


        # -------加入softmax层
        w1 = tf.Variable(tf.truncated_normal([data_dimension, self.output_classes], stddev=0.1),name='w_softmax');
        b1 = tf.Variable(tf.zeros([self.output_classes]) / 10,name='b_softmax');
        y_logits = tf.matmul(self.nn_y_middle[len(self.nn_y_middle)-1], w1) + b1;
        y = tf.nn.softmax(y_logits,name='y_softmax');

        return  y;




