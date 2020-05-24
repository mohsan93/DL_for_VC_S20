from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

import numpy as np
from dlvc.ops import *
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# TODO implement steps 1-2

data_path = "" #something ending with "...\\cifar-10-batches.py"
trainingDataset = PetsDataset(data_path, Subset.TRAINING)
validationDataset = PetsDataset(data_path, Subset.VALIDATION)
testDataset = PetsDataset(data_path, Subset.TEST)

op = chain([
    vectorize(),
    type_cast(np.float32),
    add(-127.5),
    mul(1/127.5),
])


bg_training = BatchGenerator(dataset=trainingDataset, num=32, shuffle=True, op=op)
bg_validation = BatchGenerator(dataset=validationDataset, num=32, shuffle=True, op=op)
bg_test = BatchGenerator(dataset=testDataset, num=32, shuffle=True, op=op)


def random_search(lr_max=1, lr_min=0.9, momentum_max=1, momentum_min=0.9):
    random_lr = (lr_max - lr_min) * np.random.random_sample() + lr_min
    random_momentum = (momentum_max-momentum_min) * np.random.random_sample() + momentum_min
    return (random_lr, random_momentum)

def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # TODO implement step 3
    clf = LinearClassifier(input_dim=3072, num_classes= 2, lr=lr, momentum= momentum, nesterov=True)

    n_epochs = 10
    for i in range(n_epochs):
        train_batches = iter(bg_training)
        i=1
        for batch in train_batches:
            
            if(i<len(bg_training)):
                clf.train(data= batch.data, labels = batch.label)
                i += 1
            else:
                break

    accuracy = Accuracy()
    
    val_batches = iter(bg_validation)
    j = 1
    for batch in val_batches:
        
        if(j < len(bg_validation)):
            predictions = clf.predict(data=batch.data)
            target = batch.label
            accuracy.update(predictions, target)
            j += 1
            
        else:
            break

    return TrainedModel(clf, accuracy)

# TODO implement steps 4-7

#random baseline accuracy:

lr, m = random_search()
baseline = train_model(lr = lr, momentum = m)
    
all_models = list()
best_model = baseline
all_models.append(baseline)
acc_threshold = 0.9
max_iter = 20

for i in np.arange(max_iter):
    lr, m = random_search()
    print(f"training model {i+1} of {max_iter}...")
    clfAccuracy = train_model(lr = lr, momentum = m)
    all_models.append(clfAccuracy)
    if clfAccuracy.accuracy > best_model.accuracy:
        best_model = clfAccuracy
    if clfAccuracy.accuracy.accuracy() >= 0.9:
        best_model = clfAccuracy
        break
print(f"accuracy of the best model: {best_model.accuracy}")
print("other accuracies:\n")
for model in all_models:
    print(f"model with learning rate {model.model.lr} and momentum {model.model.momentum}: {model.accuracy} \n")

fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection="3d")

#making a 3d plot for the hyperparameters
num_bars = len(all_models)
x_pos = list()
y_pos = list()
z_size = list()

for model in all_models:
    x_pos.append(model.model.lr)
    y_pos.append(model.model.momentum)
    z_size.append(model.accuracy.accuracy())

#print(x_pos)
z_pos = np.zeros(num_bars)
x_size = np.repeat(0.02, num_bars)
y_size = np.repeat(0.02, num_bars)

ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='red')
ax.set_xlabel('learning rate')
ax.set_ylabel('momentum')
ax.set_zlabel('accuracy')
plt.show()

#test accuracy:


accuracy = Accuracy()
    
test_batches = iter(bg_test)
j = 1
for batch in test_batches:
    
    if(j < len(bg_test)):
        predictions = best_model.model.predict(data=batch.data)
        target = batch.label
        accuracy.update(predictions, target)
        j += 1
        
    else:
        break

print(f"Accuracy on test set of best model: {accuracy}")