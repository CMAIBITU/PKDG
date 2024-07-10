
import os
import torch
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import random
from scipy import ndimage
image_list = glob('/home/zll/fedDG/dataset/Fundus/client1/data_npy/*')
raw_file = image_list[1]
raw_image = np.load(raw_file)
raw_file = raw_file.replace("data_npy", "data_label")
raw_mask = np.load(raw_file)
plt.show(raw_image)
cv2.waitKey(1)