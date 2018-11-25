import  tensorflow as tf
import time
import os
import  MyImageDo

class MyClassifier:
    #----------用于分类数字的神经网络分类器
    detecttime=0;    #分类耗时
    sess=None;      #检测对话
    ModelLoadFlag=False;    #表示模型是否加载成功
    myimagedo=MyImageDo.MyImageDo();
    def __init__(self,modelpath=''):
        #---读取模型
        #ckpt_dir = "./ckpt_dir";

        if os.path.exists(modelpath):
            #--------------先获取有效模型和参数路径
            ckpt = tf.train.get_checkpoint_state(modelpath);         #获取模型参数
            meta_dir = "";                                  #获取模型路径
            for file in os.listdir(modelpath):
                if file.__contains__("meta"):
                    meta_dir=modelpath+"\\"+file;
                    break;
            #------------------加载模型和参数
            if ckpt and ckpt.model_checkpoint_path:
                try:
                    self.sess = tf.Session();
             #       with self.sess:
                    saver = tf.train.import_meta_graph(meta_dir);
                    saver.restore(self.sess, ckpt.model_checkpoint_path);  # 加载所有的参数
                    print("成功加载模型：",ckpt.model_checkpoint_path);
                    self.ModelLoadFlag = True;
                except:
                    print("模型加载失败：", ckpt.model_checkpoint_path);
                    self.ModelLoadFlag = False;
            else:
                print("模型加载失败：", ckpt.model_checkpoint_path);
                self.ModelLoadFlag = False;
        else:
            print("请输入有效路径,模型加载失败!");
            self.ModelLoadFlag=False;

    #---------检测函数
    def Classfy(self,testimage):
        if self.ModelLoadFlag:
            try:
              #  with self.sess:
                # ----从模型中获取tensor,用于输入和输出
                    x = tf.get_default_graph().get_tensor_by_name('x:0');

                    predict = tf.get_default_graph().get_tensor_by_name('predict:0');
                    time1 = time.time();
                    predict_result = self.sess.run((predict),feed_dict={x:testimage});
                    time2 = time.time();
                    self.detecttime = time2 - time1;
                    return predict_result;
            except Exception as e:
                print(e);
                return None;
        else:
            print("请先加载模型!")
            return None;
