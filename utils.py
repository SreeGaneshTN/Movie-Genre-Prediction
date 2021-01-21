import requests
import os
import cv2
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from sklearn.preprocessing import MultiLabelBinarizer


def poster_download(data,file_destination):
    for index,row in data.iterrows():
        img_path=str(file_destination)+'/'+str(row.imdbId)+'.jpg'
        link=str(row.Poster)
        r=requests.get(link)
        with open(img_path ,'wb') as f:
            f.write(r.content)
        

def get_genre(data,id):
    return tuple((data[data['imdbId'] == id]['Genre'].values[0]).split('|'))

def preprocess(path,size=150):
    src=cv2.imread(path)
    img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(size,size))
    img=img/255
    return img
    


def display_poster(images_arr):
    fig,axes=plt.subplots(1,10,figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def Binarizer(y):
    m=MultiLabelBinarizer()
    y_encoded=m.fit_transform(y)
    return y_encoded,m.classes_

    


