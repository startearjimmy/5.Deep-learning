import cv2
import  os
train ='D:\\深度學習\\DL_HW2\\animal-10\\train'
test = 'D:\\深度學習\\DL_HW2\\animal-10\\val'
categories = ['butterfly','cat','chicken','cow','dog','elephant','horse','sheep','spider','squirrel']
train_data=[]

def getdata():
    i=0
    for category in categories:
        data=[]
        path = os.path.join(train,category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),0)
            data.append(img_array)
            print(img_array.size())
        print(len(data))
        train_data.append(data)
        i+=1
    print(len(train_data),'*',len(train_data[0]))
    
    test_data=[]
    i=0
    for category in categories:
        data=[]
        path = os.path.join(test,category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            data.append(img_array)
        print(len(data))
        test_data.append(data)
        i+=1
    print(len(test_data),'*',len(test_data[0]))
    return train_data,test_data
    
if __name__ == '__main__':
    getdata()