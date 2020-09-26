# import all packages necessary


import matplotlib.pyplot as plt
import random
import numpy as np

from os import listdir
from matplotlib import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split




# load all images in a directory
mine_images = list()
directory = listdir('images/50sqkm_imagery')
for i in range(800):
    # load image
    filename = directory[i]
    img_data = image.imread('images/50sqkm_imagery/' + filename)
    # store loaded image
    mine_images.append(img_data)

nonmine_images = list()
directory = listdir('images/nonmine_imagery')
for i in range(800):
    # load image
    filename = directory[i]
    if filename.endswith(".png"):
        img_data = image.imread('images/nonmine_imagery/' + filename)
        # store loaded image
        nonmine_images.append(img_data)

images = np.array(mine_images[:] + nonmine_images[:])
labels = np.array([1]*len(mine_images) + [0]*len(nonmine_images))

random = np.arange(len(images))
np.random.shuffle(random)
images = images[random]
labels = labels[random]

train_images = images[:700]
train_labels = labels[:700]

test_images = images[700:]
test_labels = labels[700:]


X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size = 0.20, random_state = 853)

print("Training image shape is", X_train.shape)
print("Training label shape is", Y_train.shape)

print("Validation image shape is", X_val.shape)
print("Validation label shape is", Y_val.shape)

# Normalize the images.
X_train = (X_train / 255) - 0.5
X_val = (X_val / 255) - 0.5

# build the model
model = Sequential()
# Create convolutional layer. There are 3 dimensions for input shape
model.add(Conv2D(100, (3, 3), activation = 'relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
# Adding a second convolutional layer with 64 filters
model.add(Conv2D(200, (3, 3), activation = 'relu', input_shape = (200, 200, 3)))
# Second pooling layer
model.add(MaxPooling2D((2, 2)))
# Adding a third convolutional layer with 128 filters
model.add(Conv2D(400, (3, 3), activation = 'relu', input_shape = (200, 200, 3)))
# Third pooling layer
model.add(MaxPooling2D((2, 2)))

# Flattening
model.add(Flatten())
# Full connection
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])
              

# Define the callbacks for early stopping of model based on val loss change.
early_stopping = [EarlyStopping(monitor = 'val_loss', min_delta =  0.01, patience = 3)]
# Fitting the CNN
history = model.fit(X_train,
            Y_train,
            steps_per_epoch = 500, epochs = 10,
            callbacks = early_stopping,
            validation_data = (X_val, Y_val))
            
            
# Print out test loss and accuracy
results_test = model.evaluate(test_images, test_labels)
print(results_test)