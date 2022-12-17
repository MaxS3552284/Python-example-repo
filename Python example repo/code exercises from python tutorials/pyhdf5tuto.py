# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:52:06 2021

@author: Max
"""
## https://www.youtube.com/playlist?list=PLea0WJq13cnB_ORdGzEkPlZEN20TSt6Lx

import pandas as pd
import h5py
import numpy as np


#%%
# video3:

matrix1 = np.random.random(size = (1000,1000))
matrix2 = np.random.random(size = (10000,100))

# create hdf5 file
with h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data.h5', 'w') as hdf:
    hdf.create_dataset('dataset1', data=matrix1)
    hdf.create_dataset('dataset2', data=matrix2)
#%%
# video4:
# read files
with h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data.h5', 'r') as hdf:
    ls= list(hdf.keys())
    print('List of Datasets in this File \n ', ls)
    data=hdf.get('dataset1') # get data
    dataset1=np.array(data) # and transform it back to numpy array
    print('Shape of Dataset: \n ', dataset1.shape)

#alternative (loads all)    
f = h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data.h5', 'r')
ls2= list(f.keys())
f.close()

#%%
# video5
# create groupsand subgroups

matrix1 = np.random.random(size = (1000,1000))
matrix2 = np.random.random(size = (1000,1000))
matrix3 = np.random.random(size = (1000,1000))
matrix4 = np.random.random(size = (1000,1000))

with h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data2.h5', 'w') as hdf:
    G1 = hdf.create_group('Group1')
    G1.create_dataset('dataset1', data=matrix1)
    G1.create_dataset('dataset4', data=matrix4)

    G2 = hdf.create_group('Group2/SubGroup1')
    G2.create_dataset('dataset3', data=matrix3)

    G3 = hdf.create_group('Group1/SubGroup2')
    G3.create_dataset('dataset2', data=matrix2)

#%%
# video 6
# read groups and subgroups

with h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data2.h5', 'r') as hdf:
    base_items= list(hdf.items())
    print('Items in Base Directory: \n ', base_items)
    G1=hdf.get('Group1') 
    G1_items=list(G1.items()) 
    print('Items in Group 1: \n ', G1_items)
    dataset4 = np.array(G1.get('dataset4'))
    print('Shape of Dataset: \n ', dataset4.shape)
    G2=hdf.get('Group2') 
    G2_items=list(G2.items()) 
    print('Items in Group 2: \n ', G2_items)
    G21 = G2.get('/Group2/SubGroup1')
    G21_items=list(G21.items()) 
    G21_items=list(G21.items()) 
    print('Items in Group 21: \n ', G21_items)
    dataset3 = np.array(G21.get('dataset3'))
    print('Shape of Dataset: \n ', dataset3.shape)
    
    

#%%
# video 7
# copression 
matrix1 = np.random.random(size = (1000,1000))
matrix2 = np.random.random(size = (1000,1000))
matrix3 = np.random.random(size = (1000,1000))
matrix4 = np.random.random(size = (1000,1000))

with h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data3.h5', 'w') as hdf:
    G1 = hdf.create_group('Group1')
    G1.create_dataset('dataset1', data=matrix1, compression="gzip", compression_opts=9) # opts = 0-9
    G1.create_dataset('dataset4', data=matrix4, compression="gzip", compression_opts=9)

    G2 = hdf.create_group('Group2/SubGroup1')
    G2.create_dataset('dataset3', data=matrix3, compression="gzip", compression_opts=9)

    G3 = hdf.create_group('Group1/SubGroup2')
    G3.create_dataset('dataset2', data=matrix2, compression="gzip", compression_opts=9)

#%%
# video 8
# set and read attributes

matrix1 = np.random.random(size = (1000,1000))
matrix2 = np.random.random(size = (10000,100))

# create file
hdf= h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data4.h5', 'w')
#create dataset
dataset1 = hdf.create_dataset('dataset1', data=matrix1)
dataset2 = hdf.create_dataset('dataset2', data=matrix2)
#set attributes
dataset1.attrs['CLASS'] = 'DATA MATRIX'
dataset1.attrs['VERSION'] = '1.1'
hdf.close()


# read the hdf5 file
f2 = h5py.File('C:/Users/Max/Desktop/HTW Studium/python-stuff/hdf5_data4.h5', 'r')
ls3= list(f2.keys())

print('List of Datasets in this File \n ', ls3)
data=f2.get('dataset1') # get data
dataset1=np.array(data) # and transform it back to numpy array (get attibuts BEFORE transforming into numpy array)
print('Shape of Dataset: \n ', dataset1.shape)
#read the attributes
k = list(data.attrs.keys())
v = list(data.attrs.values())
print(k[0])
print(v[0])
print(data.attrs[k[0]])

f2.close()

#%%
# video 9
# create hdf5 file using pandas



























