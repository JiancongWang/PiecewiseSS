#%% This class is dataloader for the MRI data

import os
import os.path as osp
import numpy as np
import numpy.ma as ma
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib

from sklearn.model_selection import train_test_split

import csv
from itertools import product

# The mean and variance normalization
def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = image[mask>0]
    mean = masked_img.mean()
    std = masked_img.std()
    image = (image - mean) / max(std, 1e-5)
    
    return image, mean, std


# Create and store simple split
# 780 training, 111 validation, 111 evaluation and 111 test samples by subject    
    
def create_split(imagelist, split_dir):
    train, val_eval_test = train_test_split(imagelist, train_size = 780)
    val, eval_test = train_test_split(val_eval_test, train_size = 111)
    evaluation, test = train_test_split(eval_test, train_size = 111)
    
    with open(split_dir, 'w') as f:
        writer = csv.writer(f)
        for im in train:
            writer.writerow([im, 'train'])
        for im in val:
            writer.writerow([im, 'val'])
        for im in evaluation:
            writer.writerow([im, 'evaluation'])
        for im in test:
            writer.writerow([im, 'test'])
            
    return train, val, evaluation, test


def load_split(split_dir):
    train, val, evaluation, test = [], [], [], []
    with open(split_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == 'train':
                train.append(row[0])
            elif row[1] == 'val':
                val.append(row[0])
            elif row[1] == 'evaluation':
                evaluation.append(row[0])
            elif row[1] == 'test':
                test.append(row[0])
            else:
                raise ValueError("Unknown split")
    return train, val, evaluation, test

# Function fetched and adapted from this thread
# https://stackoverflow.com/questions/43922198/how-to-rotate-a-3d-image-by-a-random-angle-in-python
def random_rotation_3d(image, mask, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    A randomly rotated 3D image and the mask
    """
    # Consider this function being used in multithreading in pytorch's dataloader,
    # if one don't reseed each time this thing is run, the couple worker in pytorch's
    # data worker will produce exactly the same random number and that's no good.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # image_raw = image.copy()
    # rotate along z-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(0, 1), reshape=False, order = 1)
    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(0, 1), reshape=False, order = 0)
    # mask = scipy.ndimage.interpolation.rotate(image_raw, angle, mode='nearest', axes=(0, 1), reshape=False, order = 3)
     
    # rotate along y-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(0, 2), reshape=False, order = 1)
    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(0, 2), reshape=False, order = 0)
    # mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='nearest', axes=(0, 2), reshape=False, order = 3)

    # rotate along x-axis
    angle = np.random.uniform(-max_angle, max_angle)
    image = scipy.ndimage.interpolation.rotate(image, angle, mode='constant', axes=(1, 2), reshape=False, order = 1)
    mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='constant', axes=(1, 2), reshape=False, order = 0)
    # mask = scipy.ndimage.interpolation.rotate(mask, angle, mode='nearest', axes=(1, 2), reshape=False, order = 3)

    return image, mask

# This dataset loads whole image and its binary mask, does image level normalization and augmentation
class PairedImageDataset(Dataset):
    # imagelist: directories of all the nii brain images
    def __init__(self, data_dir, 
                 istrain = True, whitening = True, mask_type = 'mean_plus', augment_type = ['flip', 'rotate'], 
                 augment_param = {'max_angle' : 10}
                 ):
        
        self.data_dir = data_dir
        self.subjectlist = os.listdir(self.data_dir)
        
        self.pairlist = [p for p in list(product(self.subjectlist, self.subjectlist)) if p[0]!=p[1]]
        
        self.istrain = istrain
        self.augment_type = augment_type
        self.augment_param = augment_param
        self.whitening = whitening
        
    def __len__(self):
        return len(self.pairlist)
    
    def __getitem__(self, idx):
        # Image level normalization and augmentation put here
        fixed_dir, moving_dir = self.pairlist[idx]
        
        # Low resolution image
        fixed = nib.load(osp.join(self.data_dir, fixed_dir, "norm.nii.gz"))
        affine = fixed.affine # The affine matrix that convert the pixel space to world space
        fixed_np = fixed.get_fdata()
        
        fmask = nib.load(osp.join(self.data_dir, fixed_dir, "norm_aseg.nii.gz"))
        fmask_np = fmask.get_fdata()
        
        moving = nib.load(osp.join(self.data_dir, moving_dir, "norm.nii.gz"))
        affine = moving.affine # The affine matrix that convert the pixel space to world space
        moving_np = moving.get_fdata()
        
        mmask = nib.load(osp.join(self.data_dir, moving_dir, "norm_aseg.nii.gz"))
        mmask_np = mmask.get_fdata()
        
        # Stack them together  so that the spatial augmentation is aligned.
        images = np.stack([fixed_np,  moving_np], axis = -1)
        masks = np.stack([fmask_np, mmask_np], axis = -1)
        
        # Augmentation
        # The augmentation is inspired by lfz/deeplung's data.py
        # If train, do augmentation. If test, no augmentation is needed
        if self.istrain:
            # left right flip
            if 'flip' in self.augment_type:
                flip = np.random.random() > 0.5
                if flip:
                    images = images[::-1] # I remember the left right axis is the first one
                    masks = masks[::-1] 
            
            # Random rotate
            if 'rotate' in self.augment_type:
                rotate = np.random.random() > 0.5
                if rotate:
                    images, masks = random_rotation_3d(images, masks, self.augment_param['max_angle'])
        
        fixed_np,  moving_np = images[..., 0], images[..., 1]
        fmask_np, mmask_np = masks[..., 0], masks[..., 1]
        
        # Mean and var normalization
        if self.whitening:
            fixed_np, fixed_mean, fixed_std = whitening_transformation(fixed_np, fmask_np)
            moving_np, moving_mean, moving_std = whitening_transformation(moving_np, mmask_np)
                    
        return {'fixed': torch.from_numpy(fixed_np.astype(np.float32)), 
                'fmask': torch.from_numpy(fmask_np.astype(np.int32)), 
                'fixed_mean': fixed_mean,
                'fixed_std': fixed_std,
                
                'moving': torch.from_numpy(moving_np.astype(np.float32)), 
                'mmask': torch.from_numpy(mmask_np.astype(np.int32)), 
                'moving_mean': moving_mean,
                'moving_std': moving_std,
                
                'affine': torch.from_numpy(affine),
                
                'fixed_dir': fixed_dir, 
                'moving_dir': moving_dir
                }
            