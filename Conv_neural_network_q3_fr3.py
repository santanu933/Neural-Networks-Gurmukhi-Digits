#!/usr/bin/env python
# coding: utf-8

# In[307]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[308]:


train_data_dir = r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001"
val_data_dir =r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001"


# In[309]:


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)


# In[310]:


train_val_labels =r"C:\Users\SANTANU\Downloads\train_val.csv"


# In[311]:


from tensorflow.keras.preprocessing import image
img= image.load_img(r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001\train_val\12.png")
plt.imshow(img)


# In[312]:


import os

dir_path=r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001\train_val"

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i)
    plt.imshow(img)
    plt.show()


# In[313]:


#load the train dataset in np

images = []
labels =[]

for i in os.listdir(dir_path):
    img_array=np.array(img)

    images.append(img_array)
    labels.append(i)


# In[314]:


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
labels=le.fit_transform(labels)


    


images=np.array(images)
labels= np.array(labels)

np.save('x_train.npy',images)
np.save('y_train.npy',labels)


x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')


# In[315]:


x_train.shape


# In[316]:


x_train[:5]
y_train[:5]


# For splitting the train and val images into 80% and 20% used the seed=42

# In[317]:


train_generator = train_datagen.flow_from_directory(
    
    directory=train_data_dir,
    
    class_mode="categorical",
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42,
   
)


val_generator = val_datagen.flow_from_directory(
    
    directory=val_data_dir,
    
    class_mode="categorical",
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42,
   
)


# In[318]:


test_data_dir = r"C:\Users\SANTANU\Downloads\test-20230423T074543Z-001"


# In[319]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[320]:


import os

test_path=r"C:\Users\SANTANU\Downloads\test-20230423T074543Z-001\test"

for i in os.listdir(test_path):
    img = image.load_img(test_path+'//'+ i)
    plt.imshow(img)
    plt.show


# In[321]:


#load the test dataset in np

images = []
labels =[]

for i in os.listdir(test_path):
    img_array=np.array(img)

    images.append(img_array)
    labels.append(i)


# In[322]:


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
labels=le.fit_transform(labels)


    


images=np.array(images)
labels= np.array(labels)

np.save('x_test.npy',images)
np.save('y_test.npy',labels)


x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')


# In[323]:


x_test.shape


# In[324]:


test_generator = test_datagen.flow_from_directory(
    
    directory=test_data_dir,
    
    class_mode="categorical",
    target_size=(128, 128),
    batch_size=32,
    shuffle=True,
    seed=42,
   
)


# In[325]:


history = model.fit(
    test_generator,
    epochs=10,
    
    validation_data=test_generator
)



# In[327]:


test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)


# In[328]:


predictions = model.predict(test_generator)


# In[329]:


print(predictions)


# In[330]:


predictions_val = model.predict(val_generator)


# In[238]:


predictions_val


# In[333]:


plt.imshow(x_test[48])


# In[334]:


print(type(train_val_labels))
print(train_val_labels)


# In[335]:


train_val_labels=pd.read_csv(r"C:\Users\SANTANU\Downloads\train_val.csv")


# In[336]:


image_labels = ['line', 'dot_line','hbar_categorical','vbar_categorical','pie']
image_labels[0]
label_map = {'line': 0, 'dot_line': 1,'hbar_categorical':2, 'vbar_categorical':3,'pie':4}

# create a list of integer labels for each string label in train_val_labels['type']
y_train = ([label_map[label] for label in train_val_labels['type']])

# convert the list to a numpy array
y_train = np.array(y_train)

print(y_train)


# In[337]:


def image_test(x,y,index):
    
    plt.imshow(x[index])
    
    plt.xlabel(image_labels[y[index]])


# In[338]:


image_test(x_train,y_train,56)
image_test(x_train,y_train,128)


# In[339]:


x_train=x_train/255
x_test=x_train/255


# In[340]:


x_test.shape


# In[341]:


y_train_img_index=train_val_labels['image_index']
y_train_type=train_val_labels['type']
y_train_type[:1000]


# In[342]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[343]:


print(x_train.shape)
print(y_train.shape)


# In[344]:


print(x_test.shape)
print(y_test.shape)


# In[345]:


from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)


# # CNN TWO LAYER MODEL

# In[364]:


from keras.optimizers import Adam


# In[368]:


c_model = Sequential()

c_model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(128, 128, 3)))
c_model.add(MaxPooling2D((2, 2)))
c_model.add(Dropout(0.2))

c_model.add(Conv2D(64, (3, 3), activation="relu"))
c_model.add(MaxPooling2D((2, 2)))
c_model.add(Dropout(0.2))

c_model.add(Conv2D(128, (3, 3), activation="relu"))
c_model.add(MaxPooling2D((2, 2)))
c_model.add(Dropout(0.2))

c_model.add(Flatten())

c_model.add(Dense(256, activation="relu"))
c_model.add(Dropout(0.5))

c_model.add(Dense(5, activation="softmax"))

opt = Adam(lr=0.001)
c_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

history = c_model.fit(x_train,y_train,batch_size=32, epochs=5, validation_data=(x_test,y_test))


# In[ ]:





# # Plot the obtained loss

# In[369]:


# Plot loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()


# In[373]:


y_predict=c_model.predict(x_test)
y_predict[0]


# In[376]:


y_class = np.argmax(y, axis=1)
y_class=[np.argmax(element)for element in y_predict]


# In[377]:


image_test(x_test,y_test,5)
image_labels[y_class[5]]


# In[372]:


test_path=r"C:\Users\SANTANU\Downloads\test-20230423T074543Z-001\test"

for i in os.listdir(test_path):
    img = image.load_img(dir_path+'//'+ i, target_size=(128,128))
    plt.imshow(img)
    plt.show()
    
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images= np.vstack([X])
    val=model.predict(images)
    if (val<30).any():
      print("vbar categorical")

    else:
      print("line or circle")


# In[ ]:




