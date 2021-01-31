import requests
import os
import cv2
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def poster_download(data,file_destination):
    i=1
    for index,row in data.iterrows():
        img_path=str(file_destination)+'/'+str(row.imdbId)+'.jpg'
        link=str(row.Poster)
        r=requests.get(link)
        with open(img_path ,'wb') as f:
            f.write(r.content)
        if i%100==0:
            print(i)
        i+=1

def get_genre(data,id):
    return tuple((data[data['imdbId'] == id]['Genre'].values[0]).split('|'))

def preprocess(path,size=150):
    try:
        src=cv2.imread(path)
    except:
        os.remove(path)

    img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(size,size))
    img=img/255
    return img
    


def display_poster(images_arr):
    fig,axes=plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def Binarizer(y):
    m=MultiLabelBinarizer()
    y_encoded=m.fit_transform(y)
    return y_encoded,m.classes_

def genre_result(result,thresh,classes):
    indices=list(np.argwhere(result>thresh)[:,-1])
    for i in indices:
        print("{} : {%0.2f}".format(classes[i],(result[0][i]*100)))
    pass




