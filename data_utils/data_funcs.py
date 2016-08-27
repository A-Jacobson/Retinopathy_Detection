import pandas as pd
import shutil
import os
import fnmatch

def train_test_val_split(y, split=0.8, random_state=1337):
    y_new = y.sample(frac=split,random_state=random_state)
    test = y.drop(y_new.index),mm,m,tyhjyhyhjj
    val = y_new.drop(train.index)
    return train, test, val

def find_class(df, label):
    return df[df.level == label].image.values

def move_file(name, split, label, src=os.path.join("E:", "DR_Data"), dst=os.path.join("E:", "DR_Data"), ending='.jpeg'):
    shutil.move(os.path.join(src, 'train_256', name+ending), os.path.join(dst, split, str(label), name+ending))

def arrange_directories(df, split, labels=[0, 1, 2, 3, 4], src=os.path.join("E:", "DR_Data"), dst=os.path.join("E:", "DR_Data"), ending='.jpeg'):
    for label in labels:
        names = find_class(df, label)
        for name in names:
            move_file(name, split, label, src, dst, ending)

def get_labels(directory):
    return [image.strip('.jpeg') for image in os.listdir(directory)]

def fix_preds(x):
    if x <= 0:
        return 0.0
    elif x >= 4:
        return 4.0
    else:
        return x

def write_answer(preds, directory):
    images = get_labels(directory)
    preds['image'] = images
    preds['level'] = preds['0']
    preds['level'] = preds['0'].apply(lambda x: np.round(x, 0))
    preds['level'] = preds['level'].apply(fix_preds)
    preds = preds.fillna(0)
    preds['level'] = preds['level'].astype(int)
    preds.to_csv('answers.csv', columns=['image', 'level'], index=False)
