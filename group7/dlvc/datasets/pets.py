
from ..dataset import Sample, Subset, ClassificationDataset
import pickle
from os import listdir
import os
import numpy as np

from os.path import isfile, join


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files

        self.data = None
        self.labels = None

        # check if directory exists
        is_directory = os.path.isdir(fdir)
        if is_directory:
            pass
        else:
            raise Exception(fdir + " is not a valid directory.")

        # get filenames of files in directory
        files = [f for f in listdir(fdir)]

        if subset == Subset.TRAINING:
            files_needed = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
            self.missing_files(files, files_needed)
            full_file_paths = [os.path.join(fdir, i) for i in files_needed]
            self.data, self.labels = self.read_and_process(full_file_paths)

        if subset == Subset.VALIDATION:
            files_needed = ["data_batch_5"]
            self.missing_files(files, files_needed)
            full_file_paths = [os.path.join(fdir, i) for i in files_needed]
            self.data, self.labels = self.read_and_process(full_file_paths)

        if subset == Subset.TEST:
            files_needed = ["test_batch"]
            self.missing_files(files, files_needed)
            full_file_paths = [os.path.join(fdir, i) for i in files_needed]
            self.data, self.labels = self.read_and_process(full_file_paths)

    def read_and_process(self, files):
        labels = []
        data = np.array([])

        for file in files:
            with open(file, 'rb') as fo:
                raw_dict = pickle.load(fo, encoding='bytes')
                # get the indices of the list with cat and dog (0,1) labels
                indices_to_keep = [i for i, e in enumerate(raw_dict[b'labels']) if e == 3 or e == 5]

                # use those indices to get the desired data and labels
                label_encoded = [0 if j == 3 else 1 for j in [raw_dict[b'labels'][i] for i in indices_to_keep]]
                labels.extend(label_encoded)

                # transforming the array to make it readble for plotting and inverting RGB to GBR:
                raw_images = raw_dict[b'data'][indices_to_keep, :]
                raw_images = raw_images.reshape((len(raw_images), 3, 32, 32)).transpose(0, 2, 3, 1)
                # RGB to BGR
                if data.size != 0:
                    data = np.concatenate((data, raw_images[:, :, :, ::-1]), axis=0)
                else:
                    data = raw_images[:, :, :, ::-1]
        return data, labels

    def missing_files(self, filenames, expected_files):
        result = all(elem in filenames for elem in expected_files)
        if result:
            pass
        else:
            raise Exception('some files are missing from directory ' + fdir)

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''
        if idx > len(self.data) - 1:
            raise IndexError()
        return Sample(idx, self.data[idx, :, :, :], self.labels[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return len(set(self.labels))


