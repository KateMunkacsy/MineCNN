# import all packages necessary


import matplotlib.pyplot as plt
import random
import numpy as np
from os import listdir
from matplotlib import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ---load all images from a directory---
def load_images(directory):
    images = list()
    dir = listdir(directory)
    for i in range(800):
      # load image
      filename = dir[i]
      if filename.endswith(".png"):
        img_data = image.imread(directory + filename)
        images.append(img_data)
      
    return images
        
mine_images = load_images('images/50sqkm_imagery/')  
nonmine_images = load_images('images/nonmine_imagery/')
# ----------------------------------------


# ---combine, shuffle, split and transform data---

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


X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels,
                                                  test_size = 0.20, random_state = 853)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow(X_train, Y_train, batch_size = 35)
val_generator = val_datagen.flow(X_val, Y_val, batch_size = 35)

# -------------------------------------------------

# ---create a model---
model = Sequential()

model.add(Conv2D(100, (3, 3), activation = 'relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(200, (3, 3), activation = 'relu', input_shape = (200, 200, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(400, (3, 3), activation = 'relu', input_shape = (200, 200, 3)))
model.add(MaxPooling2D((2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
# ------------------------


# ---Compiling the CNN---
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])
              
              
# ---include early stopping---
early_stopping = [EarlyStopping(monitor = 'val_loss', min_delta =  0.01, patience = 3)]


# ---Fit the CNN---
history = model.fit(train_generator,
            steps_per_epoch = 2, epochs = 10,
            callbacks = early_stopping,
            validation_data = val_generator)
          

# Print out test loss and accuracy
results_test = model.evaluate(test_images, test_labels)
print(results_test)

'''# ---Save the model---
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')'''




'''# ---Visualize the process---
accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']
loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'b', label = 'Training Accuracy')
plt.plot(epochs, validation_accuracy, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()


plt.plot(epochs, loss, 'b', label = 'Training Loss')
plt.plot(epochs, validation_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()'''