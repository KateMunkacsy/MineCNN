import glob
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder 






IMG_WIDTH=700
IMG_HEIGHT=700
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)


def create_lists(set):
  '''create arrays with image data and label data 
      for training and validation sets'''
  if set == 'train':
    mine_files = glob.glob('/home/cdsw/datasets/train/mines/*.png')
    nonmine_files = glob.glob('/home/cdsw/datasets/train/non_mines/*.png')
  
  elif set == 'validation':
    mine_files = glob.glob('/home/cdsw/datasets/validation/mines/*.png')
    nonmine_files = glob.glob('/home/cdsw/datasets/validation/non_mines/*.png')
  
  mine_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in mine_files]
  nonmine_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in nonmine_files]
  imgs_org = np.array(mine_imgs + nonmine_imgs)
  #print(imgs_org[:6])
  labels_org = ['mine' for fn in mine_files] + ['nonmine' for fn in nonmine_files]

  # Shuffle lists with same order 
  # Using zip() + * operator + shuffle() 
  temp = list(zip(imgs_org,labels_org))
  #print(temp)
  random.shuffle(temp) 
  images, labels = zip(*temp) 
  
  return images, labels
  

def label_encode(train_labels, validation_labels):
  '''encode text category labels''' 
  le = LabelEncoder()
  le.fit(train_labels) 
  train_labels_enc = le.transform(train_labels) 
  validation_labels_enc = le.transform(validation_labels)
  
  return train_labels_enc, validation_labels_enc

  



def main():
  trains_imgs, train_labels = create_lists('train')
  validation_imgs, validation_labels = create_lists('validation')

  # check the shapes of the training and validation sets
  # should be 5000 and 1000 200 X 200
  print('Train dataset shape:', train_imgs.shape, 
   '\tValidation dataset shape:', validation_imgs.shape)


  print(train_labels[5023:5024], train_labels_enc[5023:5024])