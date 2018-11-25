import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import math as math
from PIL import Image

class MyImageDo:
    #----进行简单图像操作

    #----私有方法
    #----私有方法计算横宽
    def __caculate_factor(self,number):
        minus = number;
        width = 1;
        height = 1
        for i in range(1, number + 1):
            if number % i == 0:
                temp = math.fabs(i - number / i);
                if (temp < minus):
                    minus = temp;
                    width = i;
                    height = number / i;

        return width, height;


    #----------------------------------------------------对外接口---------------------------------#
    #---------根据路径读取图片,异常返回None,否则返回图像数据
    def ReadImgFromFile(self,path):
        ret=None;
        if path==None:
            return ret;
        try:
           # ret = mpimg.imread(path)  # 读取图片为数组格式
            ret=Image.open(path);       #输出为图像格式
        except Exception as e:
            print(e);

        return  ret;

    #----------显示图像,返回值为0表示成功，返回值其他只表示异常
    def ShowImage(self,img,imagename,imagechannel=1):
     #   if len(img)==0:
      #      return 1;           #1---无图像
        #数据是图像
        ty=type(img);
        if  str(ty).__contains__("PIL"):
            Image._show(img);
            return  1;
        else:
        #---数据是数组
            try:
                channel=img.shape;
                if len(channel)==2:
                    #----此时表示需要显示一个二维数组（矩形为图片）
                    plt.imshow(img,'gray');
                    plt.axis('off');
                    plt.title(imagename);
                    plt.show();
                    return 0;
                elif len(channel)==3:
                    # ----此时表示需要显示一个二维数组（矩形为图片）
                    plt.imshow(img);
                    plt.axis('off');
                    plt.title(imagename);
                    plt.show();
                    return 0;
                elif len(channel)==4:
                    #----此时表示是显示一个图像集合
                    if channel[0]>100:return 2;     #图像过多，则不显示

                    show_width_num, show_height_num = self.__caculate_factor(channel[0]);  # ----获取最优显示布局
                    plt.figure(imagename);
                    if imagechannel==1:
                        for index in range(channel[0]):
                            image_x=img[index, :, :, 0];    #对于单通的图像就这么显示
                           # image_x = img[index, :, :, :];

                            # ---------------in order to show the image ,do some ops
                           # pinjie = np.zeros([image_x.shape[0], image_x.shape[1], 2]);
                           # image = np.concatenate((image_x, pinjie), axis=2);

                            plt.subplot(show_width_num, show_height_num, index + 1);
                            #   plt.title("predict:" + str(predict[index]) + "; actual:" + str(actual[index]));
                            plt.imshow(image_x,cmap=plt.cm.gray), plt.axis('off');  # 显示图片
                          #  plt.imshow(image_x,cmap='gray'), plt.axis('off');  # 显示图片
                        plt.show();
                    else:
                        for index in range(channel[0]):
                            plt.subplot(show_width_num, show_height_num, index + 1);
                            #   plt.title("predict:" + str(predict[index]) + "; actual:" + str(actual[index]));
                            plt.imshow(img), plt.axis('off');  # 显示图片
                        plt.show();


                    return 0;
                else:
                    return 3;
            except:
                return 4;

    #-------保存图像,返回值0表示正常，否则异常
    def SaveImage(self,image,path):
        ret=0;
        channel = image.shape;
        try:
            if channel[2]==1:
                #---------如果是浮点数表示
                if type(image[0,0,0]=='float'):
                    temp=np.uint8(image[:,:,0]*255);
                else:
                    temp = np.uint8(image[:, :, 0]);
                im = Image.fromarray(temp);
                # print(im.width,im.height,im.format)
                im.save(path);

            else:
                im = Image.fromarray(image);
                # print(im.width,im.height,im.format)
                im.save(path);

            # image_x=np.reshape(image,newshape=[image.shape[0],image.shape[1]]);
            # mic.imsave(path,image_x);
            #np.save(path,image);
        except:
            ret =1;
        return  ret;