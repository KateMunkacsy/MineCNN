# creating training, validation, and test sets

import os, shutil
from os import listdir
import random


original_dataset_dir = 'images/'

base_dir = 'datasets/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Create directories
train_dir = os.path.join(base_dir,'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

    
    
    
train_mines_dir = os.path.join(train_dir,'mines')
if not os.path.exists(train_mines_dir):
    os.mkdir(train_mines_dir)

train_nonmines_dir = os.path.join(train_dir,'non_mines')
if not os.path.exists(train_nonmines_dir):
    os.mkdir(train_nonmines_dir)

validation_mines_dir = os.path.join(validation_dir,'mines')
if not os.path.exists(validation_mines_dir):
    os.mkdir(validation_mines_dir)

validation_nonmines_dir = os.path.join(validation_dir, 'non_mines')
if not os.path.exists(validation_nonmines_dir):
    os.mkdir(validation_nonmines_dir)

test_mines_dir = os.path.join(test_dir, 'mines')     
if not os.path.exists(test_mines_dir):
    os.mkdir(test_mines_dir)

test_nonmines_dir = os.path.join(test_dir, 'non_mines')
if not os.path.exists(test_nonmines_dir):
    os.mkdir(test_nonmines_dir)

    
    
    
# pulling mine imagery into appropriate folders
original_dir = 'images/mine_imagery_new'
directory = listdir(original_dir)
mine_files = []
for file in directory:
  mine_files.append(file)
random.shuffle(mine_files)

# Copy first 5000 mine images to train_mines_dir
for i in range(5000):
    src = os.path.join(original_dir, mine_files[i])
    dst = os.path.join(train_mines_dir, mine_files[i])
    shutil.copyfile(src, dst)
    
# Copy next 1000 mine images to validation_mines_dir
for i in range(5000,6000):
    src = os.path.join(original_dir, mine_files[i])
    dst = os.path.join(validation_mines_dir, mine_files[i])
    shutil.copyfile(src, dst)
    
# Copy next 1000 mine images to test_mines_dir
for i in range(6000,7000):
    src = os.path.join(original_dir, mine_files[i])
    dst = os.path.join(test_mines_dir, mine_files[i])
    shutil.copyfile(src, dst)

  
  
# pulling non-mine imagery into appropriate folders  
original_dir = 'images/nonmine_imagery_092020'
directory = listdir(original_dir)
nonmine_files = []
for file in directory:
  nonmine_files.append(file)
random.shuffle(nonmine_files)

# Copy first 5000 non-mine images to train_nonmines_dir
for i in range(5000):
    src = os.path.join(original_dir, nonmine_files[i])
    dst = os.path.join(train_nonmines_dir, nonmine_files[i])
    shutil.copyfile(src, dst)
                                                
# Copy next 1000 non-mine images to validation_nonmines_dir
for i in range(5000,6000):
    src = os.path.join(original_dir, nonmine_files[i])
    dst = os.path.join(validation_nonmines_dir, nonmine_files[i])
    shutil.copyfile(src, dst)

# Copy next 1000 non-mine images to test_nonmines_dir
for i in range(6000,7000):
    src = os.path.join(original_dir, nonmine_files[i])
    dst = os.path.join(test_nonmines_dir, nonmine_files[i])
    shutil.copyfile(src, dst)
    

    
# Sanity checks
print('total training mine images:', len(os.listdir(train_mines_dir)))
print('total training non-mine images:', len(os.listdir(train_nonmines_dir)))
print('total validation mine images:', len(os.listdir(validation_mines_dir)))
print('total validation non-mine images:', len(os.listdir(validation_nonmines_dir)))
print('total test mine images:', len(os.listdir(test_mines_dir)))
print('total test non-mine images:', len(os.listdir(test_nonmines_dir)))