#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os
from PIL import Image 
from PIL import Image 


# In[ ]:


base_path = "C:/Users/HP PC/Documents/sanjay/Meat Freshness.v1-new-dataset.multiclass"
import tensorflow as tf


# In[ ]:


os.listdir(base_path+"/train")


# In[ ]:


print("Training : \n")
print(len(os.listdir(base_path+"/train/FRESH")))
print(len(os.listdir(base_path+"/train/HALF-FRESH")))
print(len(os.listdir(base_path+"/train/SPOILED")))
print("Validation : \n")
print(len(os.listdir(base_path+"/val/FRESH")))
print(len(os.listdir(base_path+"/val/HALF-FRESH")))
print(len(os.listdir(base_path+"/val/SPOILED")))


# In[ ]:


"""All images will be scaled to 1./255 to obtain 0-1 normalized image.Also image augmentation is used."""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,rotation_range = 40,
                                                                 width_shift_range = 0.2,height_shift_range = 0.2,
                                                                 shear_range = 0.2,zoom_range = 0.2,
                                                                 horizontal_flip = True,vertical_flip = True,
                                                                 fill_mode = "nearest",)
"""Validation images also will be scale dto 1./255 to obtain 0-1 normalized image,but image augmentation is NOT USED."""
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)


# In[ ]:


train_generator = train_datagen.flow_from_directory(os.path.join(base_path,"train"),target_size = (150,150),
                                                   class_mode = "categorical",batch_size = 32,seed = 42)
validation_generator = validation_datagen.flow_from_directory(os.path.join(base_path,"val"),target_size = (150,150),
                                                    class_mode = "categorical",batch_size = 32,seed = 42,shuffle = False)


# In[ ]:


from tensorflow.keras.applications.xception import Xception
base_model = Xception(input_shape = (150,150,3),weights = "imagenet",include_top = False,pooling = "max")
"""Freeze layers to stop updating weights of base model."""
for layer in base_model.layers:
    layer.trainable = False


# In[ ]:


base_model.summary()


# In[ ]:


"""Here,we can assign last layer as add_11.It means that we can start to update weights after this layer"""
last_layer = base_model.get_layer("add_11")
print(last_layer.output_shape)


# In[ ]:


"""GlobalAveragePooling layer to reduce input dim to 1D."""
x = tf.keras.layers.BatchNormalization()(last_layer.output)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
"""Fully connected layer."""
x = tf.keras.layers.Dense(128,activation = "relu")(x)
"""Add dropout layer."""
x = tf.keras.layers.Dropout(0.3)(x)
"""Output layer"""
x = tf.keras.layers.Dense(3,activation = "softmax")(x)
"""Here,we can connect model end to end."""
model = tf.keras.models.Model(base_model.input,x)


# In[ ]:


model.summary()


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate =0.0001),loss = "categorical_crossentropy",metrics = ["acc"])
"""Callback"""
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs = {}):
        if epoch >= 10 and logs.get("acc") - logs.get("val_acc") >= 0.1:
            print("Model tends to be overfitting.Stop it.")
            self.model.stop_training = True
        elif logs.get("acc") > 0.9:
            print("Model tends to be overfitting.Stop it.")
            self.model.stop_training = True
callback = myCallback()


# In[ ]:


import PIL 


# In[ ]:


history = model.fit(train_generator,epochs = 50,batch_size = 32,validation_data = validation_generator,callbacks = [callback,],verbose = 1)


# In[ ]:


import matplotlib.pyplot


# In[ ]:


import matplotlib.pyplot as plt
"""Accuracies."""
acc = history.history["acc"]
val_acc = history.history["val_acc"]
epochs = range(50)
plt.plot(epochs,acc,label = "Training accuracy")
plt.plot(epochs,val_acc,label = "Validation accuracy")
plt.legend()
plt.show()


# In[ ]:


"""Losses."""
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.plot(epochs,loss,label = "Training Loss")
plt.plot(epochs,val_loss,label = "Validation Loss")
plt.legend()
plt.show()


# In[ ]:


model.save("meat_classify.h5")


# In[ ]:


from PIL import Image
import requests
from io import BytesIO
import numpy as np 


# In[ ]:


def get_and_process(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img1 = img
    """Resize img to proper for feed model."""
    img = img.resize((150,150))
    """Convert img to numpy array,rescale it,expand dims and check vertically."""
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0 
    x = np.expand_dims(x,axis = 0)
    img_tensor = np.vstack([x])
    return img1,img_tensor


# In[ ]:


import matplotlib.pyplot as plt 
url = "https://media.istockphoto.com/photos/spoiled-steak-picture-id466978127"
img1,test_img = get_and_process(url)
"""Predict."""
pred = model1.predict(test_img)
classes = list(train_generator.class_indices.keys())
print(f"Prediction is : {classes[np.argmax(pred)]}")
plt.imshow(img1)
plt.show()

print(classes)
print(pred)


# In[ ]:


import matplotlib.pyplot as plt 
url = "https://storage.googleapis.com/kaggle-datasets-images/1254197/2091656/f1238a14261a861a458b516b95e10ab3/dataset-card.png?t=2021-04-06-12-02-51"
img1,test_img = get_and_process(url)
"""Predict."""
pred = model1.predict(test_img)
classes = list(train_generator.class_indices.keys())
print(f"Prediction is : {classes[np.argmax(pred)]}")
plt.imshow(img1)
plt.show()

print(classes)
print(pred)

