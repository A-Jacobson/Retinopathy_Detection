import glob
import pandas as pd
import os
from scipy import misc
import numpy as np

def resize_directory(input_dir, output_dir, size=(512, 512, 3)):
    files = os.listdir(input_dir)
    for f in files:
        inpath = os.path.join(input_dir, f)
        outpath = os.path.join(output_dir, f)
        img = misc.imread(inpath)
        img = misc.imresize(img, size=size)
        misc.imsave(outpath, img)

def matrix_from_dir(directory, name='X_train'):
    images = glob.glob(os.path.join(output_dir, "*"))
    X = np.array([misc.imread(i) for i in images])
    X = X.reshape(X.shape[0], 3, 512, 512)
    np.save(os.path.join('..', 'data', 'X_train'), X)
