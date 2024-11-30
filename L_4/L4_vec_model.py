#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow

tensorflow.test.is_gpu_available()


# In[2]:


import numpy as np

_DIR_ = '/home/olkhovskiina/cross5_rank1000/L_4/'
DIM = 5
ETA = 20
PP_K = 2*ETA*(DIM - 1) + 1
print(PP_K)

dataset = np.load(_DIR_ + 'L4_vec.npy')
data_X = dataset[:,:PP_K]
data_y = dataset[:,PP_K:]
print(data_X.shape)
print(data_y.shape)


# In[3]:


X_train = data_X[:60000]
y_train = data_y[:60000]

#X_val = X[20000:30000]
#y_val = y[20000:30000]

X_test = data_X[60000:]
y_test = data_y[60000:]

print(X_train.shape)
print(y_train.shape)


# In[4]:
from tensorflow.math import l2_normalize, reduce_sum, abs
def module_loss(y_true, y_pred):
    return -abs(reduce_sum(l2_normalize(y_true, axis=-1) * l2_normalize(y_pred, axis=-1), axis=-1))


from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout

model = keras.models.Sequential(name="Retina")

model.add(Dense(PP_K, name="input"))
model.add(Activation('tanh'))
model.add(Dropout(0.))
model.add(Dense(512, name="hidden1"))
model.add(Activation('relu'))
model.add(Dropout(0.))
model.add(Dense(2048, name="middle1"))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(DIM, name="output"))

model.compile(
    loss=module_loss,
    optimizer=keras.optimizers.RMSprop(learning_rate=0.000075),
    metrics=['mae'],
)


# In[5]:


_ = model.fit(
        x=X_train, 
        y=y_train,
        epochs=300,
        batch_size=50,
        validation_data=(X_test, y_test),
#        callbacks=[WandbCallback(save_model=False, save_graph=False)],
#        verbose=2
    )


# In[6]:


model.save(_DIR_ + 'L4_vec.h5', save_format='h5')

