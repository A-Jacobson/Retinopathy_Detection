import pandas as pd
import os
path = os.path.join('data', 'samples', 'trainLabels.csv')
df = pd.read_csv(path, index_col='image')

sample_labels = df[0:10] # 10 eyes in the sample set
train_labels = df[10:] # there are 35116 eyes, representing 17558 patients
