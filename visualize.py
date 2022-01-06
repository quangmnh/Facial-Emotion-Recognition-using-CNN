import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load trained model
model = load_model('new_dataset_model.h5')

# Visualize the first convolutional layer of CNN
layer = model.layers
filters, bias = model.layers[0].get_weights()

# Print layer name and output size
print(layer[0].name, filters.shape)

# Print 32 kernel filter map using for 1st convolutional layer
fig1 = plt.figure(figsize=(4, 4))
columns = 8
rows = 4
n_filters = columns * rows
for i in range(1, n_filters + 1):
    f = filters[:, :, :, i-1]
    fig1 = plt.subplot(rows, columns, i)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray')

plt.show()

# Visualize all convolutional layer
conv_layer_index = [0, 2, 6, 8, 12, 14, 18, 20]
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

# Load image from file
img = load_img('test_visualize/happy.jpg', target_size=(48, 48))
img = img_to_array(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.expand_dims(img, axis=0)

# Get feature output from each layer
feature_output = model_short.predict(img)

# Plot the feature maps of each layer
cnt = 0
for ftr in feature_output:
    if cnt >= 6:
      columns = 16
      rows = 16
    elif cnt >= 4:
      columns = 16
      rows = 8
    elif cnt >= 2:
      columns = 8
      rows = 8
    else:
      columns = 8
      rows = 4

    cnt += 1
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, rows*columns+1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')

    plt.show()

