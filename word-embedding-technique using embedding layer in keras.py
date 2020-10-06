#!/usr/bin/env python
# coding: utf-8

# In[3]:


##tensorflow >2.0
from tensorflow.keras.preprocessing.text import one_hot


# In[4]:


sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]


# In[5]:


sent


# In[6]:


### Vocabulary size
voc_size=10000


# In[7]:



onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)


# # Word Embedding Represntation¶
# 
# 

# In[8]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
## pad_sequences to make all sentences in same size
from tensorflow.keras.models import Sequential


# In[9]:


pad_sequences


# In[10]:


sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[11]:


dim=10


# In[12]:



model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')


# In[13]:


model.summary()


# In[14]:


print(model.predict(embedded_docs))


# In[15]:


embedded_docs[0]


# In[17]:


print(model.predict(embedded_docs)[0])


# In[ ]:




