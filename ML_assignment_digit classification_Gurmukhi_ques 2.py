#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


# In[3]:


train=  ImageDataGenerator(rescale= 1/255)
val =  ImageDataGenerator(rescale= 1/255)


# In[4]:


train_dataset=train.flow_from_directory(r"C:\Users\SANTANU\Downloads\train-20230414T092137Z-001\train",
                                       target_size=(28,28),batch_size=5, class_mode='binary')


val_dataset=train.flow_from_directory(r"C:\Users\SANTANU\Downloads\val-20230414T092217Z-001\val",
                                       target_size=(28,28),batch_size=5, class_mode='binary')


# In[5]:


train_dataset.class_indices


# In[6]:


train_dataset.classes


# In[7]:


val_dataset.classes


# In[37]:


img= image.load_img(r"C:\Users\SANTANU\Downloads\train-20230414T092137Z-001\train\0\6.bmp")


# In[38]:


plt.imshow(img)


# In[55]:


dir_path=r"C:\Users\SANTANU\Downloads\train-20230414T092137Z-001\train\8"

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i)
    plt.imshow(img)
    plt.show()


# In[ ]:





# # Implementation of Neural Network

# In[60]:


from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Define the model architecture

model = Sequential([
    Flatten(input_shape=(28, 28, 3)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10, batch_size=32)
test_loss, test_accuracy = model.evaluate(val_dataset)


# In[61]:


# Print the test accuracy

print('Test Accuracy:', test_accuracy)


# Print the test loss

print('Test Loss:', test_loss)


# In[62]:


def preprocess_labels(labels):
    labels = np.array(labels)  # Convert the input labels to a NumPy array
    labels = labels.astype('int64')  # Convert the label data type to int64
    return labels


# In[63]:


train_labels =r"C:\Users\SANTANU\Downloads\train-20230414T092137Z-001\train"


# In[64]:


num_train_samples = len(train_labels)


# In[65]:


num_train_samples 


# In[66]:


predictions = model.predict(val_dataset)

# Print the predicted class labels for the first 10 test images
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)


# In[67]:


print(predictions)


# In[70]:


dir_path=r"C:\Users\SANTANU\Downloads\digit\val-20230414T092217Z-001\val\1"

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i, target_size=(28,28))
    plt.imshow(img)
    plt.show()
    
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images= np.vstack([X])
    val=model.predict(images)
    if (val == 1).any():
      print("accurate")

    
    else:
      print("error")


# In[ ]:




