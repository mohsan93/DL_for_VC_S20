from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# TODO implement steps 1-2


#trainingDataset = PetsDataset("D:\\fuck\\dlvc2020\\assignments\\cifar-10-batches-py", Subset.TRAINING)
#validationDataset = PetsDataset("D:\\fuck\\dlvc2020\\assignments\\cifar-10-batches-py", Subset.VALIDATION)
#testDataset = PetsDataset("D:\\fuck\\dlvc2020\\assignments\\cifar-10-batches-py", Subset.TEST)

#bg_training = BatchGenerator(dataset=trainingDataset, num=500, shuffle=True, op=op)
#bg_validation = BatchGenerator(dataset=validationDataset, num=500, shuffle=True, op=op)
#bg_test = BatchGenerator(dataset=testDataset, num=500, shuffle=True, op=op)


def random_search(lr_max=1, lr_min=0.5, momentum_max=1, momentum_min=0):
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
        i=0
        for batch in train_batches:
            if(i<len(bg_training)):
                clf.train(data= batch.data, labels = batch.label)
                i += 1
            else:
                break

    accuracy = Accuracy()
    
    val_batches = iter(bg_validation)
    j = 0
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