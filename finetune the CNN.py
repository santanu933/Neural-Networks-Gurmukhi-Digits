#!/usr/bin/env python
# coding: utf-8

# # Finetune the model

# In[ ]:


from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


# Set the input size of the images
img_width, img_height = 224, 224


# In[22]:


# Set the directories of the training and validation data
train_data_dir = r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001"
val_data_dir = r"C:\Users\SANTANU\Downloads\train_val-20230423T074600Z-001"


# In[23]:


# Create an instance of the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))


# In[24]:


# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False


# In[25]:


# Add new layers to the pre-trained model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)


# In[26]:


# Define the new model with the pre-trained model as its base and the new layers as its top
model = Model(inputs=base_model.input, outputs=predictions)


# In[27]:


# Compile the model with a binary crossentropy loss and an Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])


# In[28]:


# Set up data augmentation for the training data and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


# In[29]:


# Set the batch size
batch_size = 16

# Set the number of training and validation samples
nb_train_samples = 1600
nb_val_samples = 400

# Set the number of epochs
epochs = 10


# In[44]:


# Train the model with the data generators
history = model.fit(
    train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary'),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary'),
    validation_steps=nb_val_samples // batch_size)


# In[32]:


# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_dir =r"C:\Users\SANTANU\Downloads\test-20230423T074543Z-001"


# In[33]:


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[37]:


test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))


# In[45]:


print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


# In[ ]:





# In[ ]:




