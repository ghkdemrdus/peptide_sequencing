#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import time
import tensorflow as tf

mz_max_len = 0
inten_max_len = 0
seq_max_len = 0

input_size = 200000
dmodel = 512
dff = 2048
num_layers = 4
num_heads = 4
dropout = 0.5
output_size = 30
EPOCHS = 10
BATCH_SIZE = 20
BUFFER_SIZE = 20000
input_array = np.array(input_array)
output_array = np.array(output_array)
intensity_array = np.array(intensity_array)
input_array = input_array.astype(np.int64)
output_array = output_array.astype(np.int64)
intensity_array = intensity_array.astype(np.float32)


dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()
model = transformer(vocab_size=input_size,
                        num_layers=num_layer,
                        dff=dff,
                        d_model=dmodel,
                        num_heads=num_head,
                        dropout=dropout)
learning_rate = CustomSchedule(dmodel)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model.fit(dataset,epochs=epoch)

