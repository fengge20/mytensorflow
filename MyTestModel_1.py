import numpy as np
import MyClassify
import MyImageDo as myimgdo
import 


def binarizing(img,threshold): #input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 255
            else:
                pixdata[x, y] = 0
    return img

#------------------------------------------实际的测试程序部分--------------------------------

#----初始化分类器
ckpt_dir = ".\ckpt_dir";
classfier=MyClassify.MyClassifier(ckpt_dir);

#--------加载要检测图片
mydo= myimgdo.MyImageDo();
img=mydo.ReadImgFromFile('img5.jpg');
#-------格式归一(大小、数据格式)
img=img.resize((28,28)).convert('L');                   #先转换为28*28的标准灰度图像
img=binarizing(img,230);
mydo.ShowImage(img,"old");
img= np.array(img).astype('float')/255;                 #最终转换为浮点数三维矩阵,0~1之间
if len(img.shape)==2:                           #标准四通道转换
    img = np.reshape(img, newshape=[1, img.shape[0], img.shape[1], 1]);
else:
    img=np.reshape(img,newshape=[1,img.shape[0],img.shape[1],img.shape[2]]);
print("检测结果为:",classfier.Classfy(img)[0]);