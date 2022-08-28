import numpy,matplotlib.pyplot,cv2,pickle,os,pandas,random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path='D:\BaiduNetdiskDownload\交通标志数据集\myData'     #数据集路径
label='D:\BaiduNetdiskDownload\交通标志数据集\labels.csv'#所有类别的名字
batch=50                                                #同批处理的图像数量
steps=400                                               #迭代次数
epoch=10                                                #整个训练集训练次数
dimesion=(32,32,3)                                      #32*32彩色图
testratio=0.2                                           #测试集占比
validationratio=0.2                                     #验证集占比

count=0
images=[]
classnumber=[]
mylist=os.listdir(path)
print('Total Classes Detected:',len(mylist))
for x in range(0,len(mylist)):
    picturelist=os.listdir(path+'/'+str(count))
    for y in picturelist:
        curimage=cv2.imread(path+'/'+str(count)+'/'+y)#加载图像和标签，编号整理
        images.append(curimage)
        classnumber.append(count)
    print(count,end='')
    count+=1
print('')

images=numpy.array(images)
classnumber=numpy.array(classnumber)#存着对应图片信息和标签

xtrain,xtest,ytrain,ytest=train_test_split(images,classnumber,test_size=testratio)
xtrain,xvalidation,ytrain,yvalidation=train_test_split(xtrain,ytrain,test_size=validationratio)#分割测试集和验证集

print(xtrain.shape,ytrain.shape)
print(xvalidation.shape,yvalidation.shape)
print(xtest.shape,ytest.shape)
assert(xtrain.shape[0]==ytrain.shape[0],'The number of images is not equal to the number of labels in the training set.')
assert(xvalidation.shape[0]==yvalidation.shape[0],'The number of images is not equal to the number of labels in the validation set.')#检查是否图像数量与每个数据集的标签数量匹配
assert(xtest.shape[0]==ytest.shape[0],'The number of images is not equal to the number of labels in the test set.')
assert(xtrain.shape[1:]==(dimesion),'The dimesions of the training set is wrong.')
assert(xvalidation.shape[1:]==(dimesion),'The dimesions of the validation set is wrong.')
assert(xtest.shape[1:]==(dimesion),'The dimesions of the test set is wrong.')

data=pandas.read_csv(label)
print('data shape',data.shape,type(data))

samplenumber=[]
col=5
fig,axs=matplotlib.pyplot.subplots(nrows=len(mylist),ncols=col,figsize=(5,300))#可视化部分图标及类别
fig.tight_layout()
for i in range(col):
    for j in data.iterrows():
        xselected=xtrain[ytrain==j]
        axs[j][i].imshow(xselected[random.randint(0,len(xselected-1)),:,:])
        axs[j][i].axis('off')
        if i==2:
            axs[j][i].set_title(str(j)+'-'+row['name'])
            samplenumber.append(len(xselected))

print(samplenumber)                                             #统计类别分布
matplotlib.pyplot.figure(figsize=(12,4))
matplotlib.pyplot.bar(range(0,classnumber),samplenumber)
matplotlib.pyplot.title('distribution of the training dataset')
matplotlib.pyplot.xlabel('class number')
matplotlib.pyplot.ylabel('image number')
matplotlib.pyplot.show()

def grayscale(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转为灰度图
    return image

def equalize(image):
    image=cv2.equalizeHist(image)#直方图均衡化
    return image

def preprocessing(image):
    image=grayscale(image)
    image=equalize(image)
    image=image/255       #使图像值在[0,1]间
    return image

xtrain=numpy.array(list(map(preprocessing,xtrain)))          #数据预处理
xvalidation=numpy.array(list(map(preprocessing,xvalidation)))
xtest=numpy.array(list(map(preprocessing,xtest)))

xtrain=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],1)                         #增加一维
xvalidation=xvalidation.reshape(xvalidation.shape[0],xvalidation.shape[1],xvalidation.shape[2],1)
xtest=xtest.reshape(xtest.shape[0],xtest.shape[1],xtest.shape[2],1)

datagen=ImageDataGenerator(width_shift_range=0.1, #随机平移
                           height_shift_range=0.1,
                           zoom_range=0.2,        #随机缩放范围
                           sheae_range=0.1,       #逆时针剪切
                           rotation_range=10)     #随机旋转角度范围
datagen.fit(xtrain)
batches=datagen.flow(xtrain,ytrain,batch_size=20)
xbatch,ybatch=next(batches)

fig,axs=matplotlib.pyplot.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(xbatch[i].reshape(dimesion[0],dimesion[1]))
    axs[i].axis('off')
matplotlib.pyplot.show()

ytrain=to_categorical(ytrain,len(mylist))
yvalidation=to_categorical(yvalidation,len(mylist))
ytest=to_categorical(ytest,len(mylist))

def mymodel():       #定义模型
    filternumber=60
    flitersize1=(5,5)#第一个卷积核
    flitersize2=(3,3)#第二个卷积核
    poolsize=(2,2)   #池
    nodenumber=500   #节点数
    model=Sequential()
    model.add((Conv2D(filternumber,flitersize1,input_shape=(dimesion[0],dimesion[1],1),activation='relu')))#卷积
    model.add((Conv2D(filternumber,flitersize1,activation='relu')))                                        #卷积
    model.add(MaxPooling2D(poolsize))                                                                      #池化

    model.add((Conv2D(filternumber//2,flitersize2,activation='relu')))                                     #卷积
    model.add((Conv2D(filternumber//2,flitersize2,activation='relu')))                                     #卷积
    model.add(MaxPooling2D(poolsize))                                                                      #池化
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nodenumber,activation='relu'))
    model.add(Dropoyt(0.5))
    model.add(Dense(classnumber,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=mymodel()
print(model.summary())#开始训练
history=model.fit_generator(datagen.flow(xtrain,ytrain,batch,steps,epoch,validationdata=(xvalidation,yvalidation),shuffle=1))

score=model.evaluate(xtest,ytest,verbose=0)#评估模型
print('test score',score[0])
print('test accurancy',score[1])

model.save('traffic.h5')#保持模型

framewidth=640
frameheight=480
brightness=180
threshold=0.75
font=cv2.FONT_HERSHEY_SIMPLEX

model=tensorflow.keras.models.load_model('traffic.h5')

def getclassname(classno):
    if classno==0:
        return 'speed limit 20km/h'
    elif classno==1:
        return 'speed limit 30km/h'
    elif classno==2:
        return 'speed limit 50km/h'
    elif classno==3:
        return 'speed limit 60km/h'
    elif classno==4:
        return 'speed limit 70km/h'
    elif classno==5:
        return 'speed limit 80km/h'
    elif classno==6:
        return 'speed limit 80km/h'
    elif classno==7:
        return 'speed limit 30km/h'
    elif classno==8:
        return 'speed limit 30km/h'
    elif classno==9:
        return 'no passing'
    elif classno==10:
        return 'no passing for vechiles over 3.5 metric tons'
    elif classno==11:
        return 'right of way at the next intersection'
    elif classno==12:
        return 'priority road'
    elif classno==13:
        return 'yield'
    elif classno==14:
        return 'stop'
    elif classno==15:
        return 'no vechiles'
    elif classno==16:
        return 'vechiles over 3.5 metric tons prohibited'
    elif classno==17:
        return 'no rntry'
    elif classno==18:
        return 'grneral caution'
    elif classno==19:
        return 'dangerous curve to the left'
    elif classno==20:
        return 'dangerous curve to the right'
    elif classno==21:
        return 'double curve'
    elif classno==22:
        return 'bumpy road'
    elif classno==23:
        return 'slippery road'
    elif classno==24:
        return 'road narrows on the right'
    elif classno==25:
        return 'road work'
    elif classno==26:
        return 'traffic signals'
    elif classno==27:
        return 'pedestrians'
    elif classno==28:
        return 'children crossing'
    elif classno==29:
        return 'bicycles crossing'
    elif classno==30:
        return 'be aware of ice'
    elif classno==31:
        return 'wild animals crossing'
    elif classno==32:
        return 'end of all speed and passing limits'
    elif classno==33:
        return 'turn right ahead'
    elif classno==34:
        return 'turn left ahead'
    elif classno==35:
        return 'ahead only'
    elif classno==36:
        return 'go straight or right'
    elif classno==37:
        return 'go straight or left'
    elif classno==38:
        return 'keep right'
    elif classno==39:
        return 'keep left'
    elif classno==40:
        return 'roundabout mandatory'
    elif classno==41:
        return 'end of no passing'
    elif classno==42:
        return 'end of no passing by vechiles over 3.5 metric tons'
def pres(imageoringal):
    images=numpy.asarray(imageoringal)
    images=cv2.resize(images,(32,32))                 #输入图片指定32*32
    images=images.reshape(1,32,32,1)
    predictions=model.predict(images)                 #预测
    classindex=model.predict_classes(images)
    probabilityvalue=numpy.argmax(predictions,axis=-1)
    if probabilityvalue>threshold:                    #概率大于阈值才判断有效检测
        return str(getclassname(classindex))
    else:
        return 'no'
if __name__=='__main__':
    imageorignal=cv2.imread('img.png')
    out=pres(imageorignal)