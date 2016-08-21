import pandas as pd
import shutil
import os

def train_test_val_split(y, split=0.8, random_state=1337):
    y_new = y.sample(frac=split,random_state=random_state)
    test = y.drop(y_new.index)
    train= y_new.sample(frac=split,random_state=random_state)
    val = y_new.drop(train.index)
    return train, test, val

def find_class(df, label):
    return df[df.level == label].image.values

def move_file(name, split, label, src=os.path.join("E:", "DR_Data"), dst=os.path.join("E:", "DR_Data"), ending='.jpeg'):
    shutil.move(os.path.join(src, 'train_512', name+ending), os.path.join(dst, split, str(label), name+ending))

def arrange_directories(df, split, labels=[0, 1, 2, 3, 4], src=os.path.join("E:", "DR_Data"), dst=os.path.join("E:", "DR_Data"), ending='.jpeg'):
    for label in labels:
        names = find_class(df, label)
        for name in names:
            move_file(name, split, label, src, dst, ending)
