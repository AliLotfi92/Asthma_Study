
import numpy as np
from datetime import date
import datetime
from sklearn import preprocessing


feature_size = 15
total_num_samples = 1173

dataset = np.load('here.npy')
dataset_fixed = np.zeros([feature_size, total_num_samples])

for i in range(10):
    if i == 0 or 1:
        dataset_fixed[i] = dataset[i, 1:]
    else:
        dataset_fixed[i] = preprocessing.scale(dataset[i, 1:])

for j in range(total_num_samples):
    if np.isnan(dataset_fixed[9, j]):
        dataset_fixed[9, j] = 28.

for j in range(total_num_samples):
    dataset_fixed[10, j] = 1. if dataset[11, j+1]=='M' else 0.


for j in range(total_num_samples):
    birth_date = str(dataset[12, j+1])
    birth_date = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
    today = datetime.datetime.strptime('2020-03-18', '%Y-%m-%d')
    delta = today - birth_date
    dataset_fixed[11, j] = delta.days

dataset_fixed[11, :] = preprocessing.scale(dataset_fixed[11, :])

for j in range(total_num_samples):
    dataset_fixed[12, j] = 1.0 if dataset[13, j] == 'hispanic' else 0.

for j in range(total_num_samples):
    dataset_fixed[13, j] = 1.0 if dataset[14, j] == 'M' else 0.

for j in range(total_num_samples):
    if dataset[15, j] == 'white':
        dataset_fixed[14, j] = 0.

    elif dataset[15, j] == 'asian':
        dataset_fixed[14, j] = 1.

    elif dataset[15, j] == 'black':
        dataset_fixed[14, j] = 2.

    else:
        dataset_fixed[14, j] = 0.0

training_dataset = dataset_fixed[:, :1000]
test_dataset = dataset_fixed[:, 1000:]
np.save('training_dataset', dataset_fixed)
np.save('test_dataset', test_dataset)


