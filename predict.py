from utils import preprocess,display_poster
from tensorflow.keras.models import load_model,model_from_json
import cv2
import numpy as np

def predict(saved_model,weights,image,classes,size):
    x=[]
    x.append(preprocess(image,size))
    x=np.asarray(x)
    x=x.astype('float32')
    with open(saved_model,r) as f:
        model=model_from_json(f.read(saved_model))
    model=model.load_weights(saved_model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    result=model.predict(x)
    indices=list(np.argpartition(result,-3)[-3:])
    genres=[]
    for i in indices:
        genres.append(classes[indices[i]])
    return genres




