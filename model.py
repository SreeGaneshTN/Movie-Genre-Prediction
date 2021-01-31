import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator as aug
from tensorflow.keras.layers import Conv2D,Flatten,Dense,BatchNormalization,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16,ResNet50,InceptionV3
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import numpy as np 


def Aug(horizontal=True,Vertical=True):
        train_aug=aug(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)

        return train_aug


class CNN:
    def __init__(self,lr:int,size:int,channel:int,num_classes):
        self.model=Sequential()
        self.model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(size, size, channel)))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(16, (3, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(Dropout(0.3))           
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=channel))
        self.model.add(Dropout(0.3))
        self.model.add(MaxPooling2D(2, 2))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        print(self.model.summary())
        self.model.compile(optimizer=Adam(lr),loss=binary_crossentropy,metrics=['accuracy'])

    def train(self,x_train,y_train,x_val,y_val,epochs,batch_size,filepath):
        history=self.model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,verbose=1,
        validation_data=(x_val,y_val))
        CNN.savemodel(self)
        self.model.save_weights("model1.h5")
        return  history
    
    def savemodel(self):
        model_json=self.model.to_json()
        with open("./model.json",'w') as f:
            f.write(model_json)


class Pretrained:
    def __init__(self,lr,num_Classes,model_name):
        self.model_name=model_name
        self.model=Sequential()
        if model_name not in ['VGG','Inception','Resnet']:
            print("Pretrained Model is Not Available")
        if model_name =='VGG':
            base_model=VGG16(include_top=False,weights='imagenet')
        elif model_name =='Inception':
            base_model=InceptionV3(include_top=False,weights='imagenet')
        else:
            base_model=ResNet50(include_top=False,weights='imagenet')
        base_model.trainable=False
        self.model.add(base_model)
        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128,activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(num_Classes,activation='sigmoid'))
        print(self.model.summary())
        self.model.compile(optimizer=Adam(lr),loss=binary_crossentropy,metrics=['accuracy'])
    
    def train(self,x_train,y_train,x_val,y_val,epochs,batch_size,filepath):
        history=self.model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,verbose=1,
        callbacks=ModelCheckpoint(filepath,monitor='val_accuracy',save_best_only=True,mode='max',),
        validation_data=(x_val,y_val))
        Pretrained.savemodel(self)
        self.model.save_weights(self.model_name+".h5")
        return  history
    
    def savemodel(self):
        model_json=self.model.to_json()
        with open('./'+self.model_name+".json",'w') as f:
            f.write(model_json)


def analysis(history):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.ylim(0,2)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.ylim(0,2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

'''
def main():
    
    Aug=Augmentation()
    x=np.random.random((1,150,150,1))
    print(x.shape())
    y=np.random.random(1)
    x_aug=Aug.apply(x)
    print(x_aug.shape())
    model=Pretrained(1,3,'VGG')
'''
#main()
