# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:27:33 2021

@author: Max
"""
# tensor flow tuto playlist:
# https://www.youtube.com/playlist?list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj


# video 2: loading and looking at data

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

data = keras.datasets.fashion_mnist

(train_images, train_lables), (test_images, test_lables) = data.load_data()
# .load_data() of keras loads and splits data into test and training data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# lable name list
# print(train_lables[insert index]), only gives index of lables, names from tesorflow page

# plt.imshow(train_images[7])
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

train_images = train_images/255.0 # 255 for shrinking picture size 28x28pixel
test_images = test_images/255.0

# %%

# video 3: creating a model

# Now time to create our first neural network model! We will do this by using 
# the Sequential object from keras. A Sequential model simply defines a sequence of layers 
# starting with the input layer and ending with the output layer. 
# Our model will have 3 layers, and input layer of 784 neurons 
# (representing all of the 28x28 pixels in a picture) a hidden layer of an arbitrary 
# 128 neurons and an output layer of 10 neurons representing the probability of the picture 
# being each of the 10 classes.

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")
                          ])
# flatten input_layer data so its passable to the different neurons
# dense layers are fully connected to following layers
# "relu" = ativates by rectifi linear unit (works fast)
# "softmax" = picks values for each neuron to add up to 1, probability of the network to think its certain value

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images, train_lables, epochs = 10)
# epoch = how many times the model sees this information

# test model
test_loss, test_acc = model.evaluate(test_images, test_lables)

print('\nTest accuracy:', test_acc)


# %%

# video 4: using the model to make predictions

predictions = model.predict(test_images) #put your input shape into a list or np.array 
print(class_names[np.argmax(predictions[0])])

# display first 5 images and their prediction
plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_lables[i]])
    plt.title("Prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()


# %%

# video 5,6: tesxt classification part 1,2  (might bugg out due to different numpy version)

# import tensorflow as td
# from tensorflow import keras
# import numpy as np

# Loading Data
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# Integer Encoded Data

# A dictionary mapping words to an integer index
word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()} # add 3 to assigne special custom stuff like padding, unknown etc.
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])
# .get() if stuff cant be found put questionmark, prevents crash

# this function will return the decoded (human readable) reviews  

# Preprocessing Data

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Defining the Mode

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model


# %%

# video 7: tesxt classification part 3

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# validation data

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# train model

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
# batch_size= how much data is given at once
# test model

results = model.evaluate(test_data, test_labels)
print(results)


# %%

# video 8: tesxt classification part 4

# Saving the Model

model.save("model.h5")  # name it whatever you want but end with .h5

# load model

model = keras.models.load_model("model.h5")

# Transforming our Data

def review_encode(s):
	encoded = [1]

	for word in s: # check if word is in dictionary, if not denote it as unknown
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])












