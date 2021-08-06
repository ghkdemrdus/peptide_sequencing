#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pyteomics import mgf, auxiliary


# In[12]:


def seq_to_int(sequence):
    int_array = []
    int_array.append(1)
    for i in sequence:
        int_array.append(ord(i)-ord('A') + 3)
    int_array.append(2)
    return int_array

def normalize(arr, t_min, t_max):
    norm_arr = []
    for value in arr:
        normalized_num = (value - min(arr)) * (t_max - t_min) / (max(arr) - min(arr))
        norm_arr.append(int(normalized_num))
    return norm_arr


# In[90]:


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

msms_dir = 'GA_mgf_info'
seq = {}
cnt = 0

for dir_name in os.listdir(msms_dir):
    msms_file = pd.read_csv(msms_dir + '/' + dir_name + '/' + 'msms.txt', delimiter = '\t')
    print(dir_name)
    for idx, row in msms_file.iterrows():
        title = row['Raw file']+'.'+ str(row['Scan number'])+'.'+ str(row['Scan number'])+'.'+ str(row['Charge'])
        sequence = row['Sequence']
        score = row['Score']
        output_array = seq_to_int(sequence)
        seq[title] = {'outputs' : output_array, 'score' : score}
        for prev_title in seq:
            if prev_title != title and seq[prev_title]['outputs'] == output_array:
                if seq[prev_title]['score'] < score:
                    del seq[prev_title]
                    break
                else:
                    del seq[title]
                    break


# In[92]:


cnt = 0
for raw_title in seq:
    cnt = cnt + 1
print(cnt)


# In[96]:


filename = 'mgf_data.tfrecords'
mgf_path = 'GA_mgf'
dataset_size=0
with tf.io.TFRecordWriter(filename) as writer:
    for file_name in os.listdir(mgf_path):
        reader = mgf.read(mgf_path + '/' + file_name)
        print(file_name)
        for spectrum in reader:
            title = spectrum['params']['title'].split()[0]
            mz = np.array(100*spectrum['m/z array'], dtype=np.int)
            intensity = np.array(spectrum['intensity array'], dtype=np.float)
            #integer nomalization
            intensity = normalize(intensity, 0, 100)
            if title in seq:
                dataset_size = dataset_size + 1
                sequence = seq[title]['outputs']
                feature = {
                    'inputs': _int64_feature(mz),
                    'intensity': _int64_feature(intensity),
                    'outputs': _int64_feature(sequence),
                }
                serialized_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(serialized_example.SerializeToString())


# In[97]:


print(dataset_size)


# In[98]:


feature_description = {
    'inputs': tf.io.FixedLenSequenceFeature ([], tf.int64, allow_missing=True),
    'intensity': tf.io.FixedLenSequenceFeature ([], tf.int64, allow_missing=True),
    'outputs': tf.io.FixedLenSequenceFeature ([], tf.int64, allow_missing=True)
}

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def input_fn(record_file):
    parsed_data=_parse_function
    dataset = tf.data.TFRecordDataset(record_file)   
    parsed_dataset = dataset.map(_parse_function)
#     train_dataset = parsed_spec_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
#     train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return parsed_dataset


# In[108]:


full_dataset = input_fn(filename)

# train_size = int(0.7 * dataset_size)
# test_size = int(0.3 * dataset_size)
train_size = 1000
# full_dataset = full_dataset.shuffle(buffer_size=1000000)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size).take(2000)

train_file = "train.tfrecords"
test_file = "test.tfrecords"


# In[102]:


cnt1 = 0
for (batch, spec) in enumerate(full_dataset):
    if cnt1 == 100:
        break
    input_raw = spec['inputs']
    intensity_raw = spec['intensity']
    output_raw = spec['outputs']
    train_data[batch] = {'inputs' : input_raw, 'intensity' : intensity_raw, 'outputs' : output_raw}
    print(train_data[batch])
    cnt1 = cnt1 + 1


# In[109]:


train_data = {}
cnt3=0
cnt4=0
for (batch, spec) in enumerate(train_dataset):
    input_raw = spec['inputs']
    intensity_raw = spec['intensity']
    output_raw = spec['outputs']
    train_data[batch] = {'inputs' : input_raw, 'intensity' : intensity_raw, 'outputs' : output_raw}
    cnt3 = cnt3 + 1
print(cnt3)
test_data = {}
for (batch, spec) in enumerate(test_dataset):
    input_raw = spec['inputs']
    intensity_raw = spec['intensity']
    output_raw = spec['outputs']
    test_data[batch] = {'inputs' : input_raw, 'intensity' : intensity_raw, 'outputs' : output_raw}
    cnt4 = cnt4 +1
print(cnt4)


# In[29]:


print(test_data[batch])

