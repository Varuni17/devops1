#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import vgg16


# In[2]:


# pre trained weights : create my model
model = vgg16.VGG16(weights='imagenet')


# In[3]:


model.layers[0].input


# In[4]:


from keras.preprocessing import image


# In[5]:


img = image.load_img('cat_or_dog_1.jpg', target_size=(224,224))


# In[6]:


img


# In[7]:


img.size


# In[8]:


type(img)


# In[9]:


img_np = image.img_to_array(img)


# In[10]:


img_np.shape


# In[11]:


type(img_np)


# In[12]:


import numpy as np


# In[13]:


a = np.array([1,2])


# In[14]:


a.shape


# In[15]:


b = np.expand_dims(a, axis=0)


# In[16]:


b.shape


# In[17]:


b[0]


# In[18]:


ae = np.expand_dims(img_np, axis=0)


# In[19]:


ae.shape


# In[20]:


from keras.applications.vgg16 import decode_predictions


# In[21]:


from keras.applications.vgg16 import preprocess_input


# In[25]:


finalimg = preprocess_input(ae)


# In[ ]:





# In[26]:


pred = model.predict(finalimg)


# In[27]:


decode_predictions(pred, top=3)[0]


# In[ ]:
