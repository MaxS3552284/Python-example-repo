# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:44:06 2021

@author: Max
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
print(tf.version)

# %%

# BASIC SHIT START:

# rank/degree (dimensions involved) zero tensor
string = tf.Variable("this is a string", tf.string) # create string tensor (value, type)
number = tf.Variable(324, tf.int16)                 # int tensor
floating = tf.Variable(3.567, tf.float64)           # float tensor


# rank 1 (array)
rank1_tensor = tf.Variable(["test"], tf.string)
# rank 2 (2d array)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)


print(tf.rank(rank2_tensor))
d = rank2_tensor.shape 
print(d) # get a number as output per rank


# Changing Shape
# The number of elements of a tensor is the product of the sizes of all its shapes. 
# There are often many shapes that have the same number of elements, 
# making it convient to be able to change the shape of a tensor.
# The example below shows how to change the shape of a tensor.

tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,3]
                                                                             
# The numer of elements in the reshaped tensor MUST match the number in the original
print(tensor1)
print(tensor2)
print(tensor3)
# Notice the changes in shape


# Types of Tensors
# Before we go to far, I will mention that there are diffent types of tensors. 
# These are the most used and we will talk more in depth about each as they are used.

# Variable
# Constant
# Placeholder
# SparseTensor
# With the execption of Variable all these tensors are immuttable, 
# meaning their value may not change during execution.

# For now, it is enough to understand that we use the Variable tensor 
# when we want to potentially change the value of our tensor.


# BASIC SHIT END:
    
# %%

# Load dataset:
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())

print(dftrain.describe())

# kj = dftrain.shape()
# print(kj)

print(y_train.head())

dftrain.age.hist(bins=20)

dftrain.sex.value_counts().plot(kind='barh')

dftrain['class'].value_counts().plot(kind='barh')

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# feature columns:

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# input function:
    # The TensorFlow model we are going to use requires that the data we pass it comes in as a 
    # tf.data.Dataset object. This means we must create a input function that 
    # can convert our current pandas dataframe into that object.
    
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


# Creating the Model:

# In this tutorial we are going to use a linear estimator to utilize the linear regression algorithm.
# Creating one is pretty easy! Have a look below.

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

# train model:

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

# predict

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

# %%











