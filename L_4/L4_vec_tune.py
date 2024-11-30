import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from wandb.keras import WandbCallback

_DIR_ = '/home/olkhovskiina/cross5_rank1000/L_4/'
DIM = 5
ETA = 20
PP_K = 2*ETA*(DIM - 1) + 1
print(PP_K)

dataset = np.load(_DIR_ + 'L4_vec.npy')
data_X = dataset[:,:PP_K]
data_y = dataset[:,PP_K:]
X_train = data_X[:65000]
y_train = data_y[:65000]
X_test = data_X[65000:]
y_test = data_y[65000:]

from tensorflow.math import l2_normalize, reduce_sum, abs
def module_loss(y_true, y_pred):
    return -abs(reduce_sum(l2_normalize(y_true, axis=-1) * l2_normalize(y_pred, axis=-1), axis=-1))

def train():
    config_defaults = {
        'input_activation' : 'relu',
        'input_dropout' : 0.,
        'hidden_1_dense' : 1024,
        'hidden_1_activation' : 'tanh',
        'hidden_1_dropout' : 0.15,
        'middle_number' : 0,
        'middle_dense' : 1024,
        'middle_activation' : 'tanh',
        'middle_dropout' : 0.15,        
        'batch_size' : 100,
        'learning_rate' : 0.00005
    }

    wandb.init(config = config_defaults)
    wandb.config.epochs = 25
    
    keras.backend.clear_session()
    model = keras.models.Sequential(name="Retina")
    
    model.add(Dense(PP_K, name="input"))
    model.add(Activation(wandb.config.input_activation))
    model.add(Dropout(wandb.config.input_dropout))
    model.add(Dense(wandb.config.hidden_1_dense, name="hidden1"))
    model.add(Activation(wandb.config.hidden_1_activation))
    model.add(Dropout(wandb.config.hidden_1_dropout))
    for i in range(wandb.config.middle_number):
        model.add(Dense(wandb.config.middle_dense, name="middle"+str(i)))
        model.add(Activation(wandb.config.middle_activation))
        model.add(Dropout(wandb.config.middle_dropout))
    model.add(Dense(DIM, name="output"))
    
    model.compile(
        loss=module_loss,
        optimizer=keras.optimizers.RMSprop(learning_rate=wandb.config.learning_rate),
        metrics=['mae'],
    )
    
    _ = model.fit(
        x=X_train, 
        y=y_train,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[WandbCallback(save_model=False, save_graph=False)],
        verbose=2
    )

wandb.agent('6otu3wxa', function=train, project="BSF-ANN")