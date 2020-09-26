#!pip install -q tensorflow-hub
#!pip install -q tensorflow-datasets
!pip install tensorflow==2.2

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
import data_preprocessing as dp
from keras.preprocessing.image import ImageDataGenerator
import numpy as np



train_imgs, train_labels = dp.create_lists('train')
validation_imgs, validation_labels = dp.create_lists('validation')

train_labels_enc, validation_labels_enc = dp.label_encode(train_labels,
                                                          validation_labels)



# perform transformations on the images
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3,
                                   rotation_range=50, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow(np.array(train_imgs), np.array(train_labels_enc), batch_size=50)
val_data = val_datagen.flow(np.array(validation_imgs), np.array(validation_labels_enc), batch_size=50)

for image_batch, label_batch in train_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break


'''feature_extractor_url = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         tags = {"train"},
                                         trainable = True,
                                         input_shape=(200,200,3))'''

# old way of loading in the model - requires tensorflow 1.15
'''module = hub.Module("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1",
                   tags = {'train'}, trainable = False)
images = image_batch  # A batch of images with shape [batch_size, height, width, 3].
output = module(images)  # Features with shape [batch_size, num_features] 
'''

# using efficientnet trained on Imagenet
'''feature_extractor_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1" 
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         trainable = True,
                                         input_shape=(200,200,3))'''
# using Resnet trained on Imagenet
feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4" 
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         trainable = True,
                                         input_shape=(350,350,3))


IMAGE_SIZE = (350, 350)
# layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),

model = tf.keras.Sequential([
  layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
  feature_extractor_layer,
  layers.Dense(512, activation='tanh'),
  layers.Dropout(0.2),
  layers.Dense(128, activation = 'tanh'),
  layers.Dropout(0.2),
  layers.Dense(32, activation = 'tanh'),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')
])

model.summary()

# edit to_file path
tf.keras.utils.plot_model(model, to_file=basePath + "/figures/modelArchitecture.png", show_shapes=False)

model.compile(
  optimizer=tf.keras.optimizers.Adam(1e-5), # change to 1e-4
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=['acc'])


steps_per_epoch = 100
validation_steps = 20


hist = model.fit(train_data, epochs=30,
                    steps_per_epoch=steps_per_epoch,
                    validation_data = val_data,
                    validation_steps=validation_steps).history

# results of final activiation as softmax


plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
x = plt.plot(hist["loss"])
y = plt.plot(hist["val_loss"])
plt.show()

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])


#-----------------------------------------------------------
#saving the model
from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# to load and read the previously saved model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=['acc'])
score = loaded_model.evaluate(train_data, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# -----------------------------------------------------------
