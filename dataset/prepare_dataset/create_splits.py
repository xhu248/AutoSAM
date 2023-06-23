import pickle

import os
import random
import numpy as np


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y   # lambda is another simplified way of defining a function
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def create_splits(output_dir, image_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)

    trainset_size = len(npy_files)*60//100
    valset_size = len(npy_files)*20//100
    testset_size = len(npy_files)*20//100

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        trainset = []
        valset = []
        testset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient)
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient)
        for i in range(0, testset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            testset.append(patient)
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = testset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def create_folds(output_dir, image_dir, fold_num=5):
    npy_files = os.listdir(image_dir)
    npy_files.sort()
    folds = []

    fold_size = len(npy_files) // fold_num
    for i in range(fold_num):
        if i < fold_num - 1:
            fold = npy_files[i*fold_size:(i+1)*fold_size]
        else:
            fold = npy_files[i*fold_size:]

        folds.append(fold)

    splits = []
    for i in range(fold_num):
        image_list = folds.copy()
        image_list[0], image_list[i] = image_list[i], image_list[0]
        if i < fold_num - 1:
            image_list[1], image_list[i+1] = image_list[i+1], image_list[1]
        else:
            image_list[1], image_list[i] = image_list[i], image_list[1]
        split_dict = dict()
        split_dict['val'] = image_list[1]
        split_dict['test'] = image_list[0]

        image_list.remove(image_list[0])
        image_list.remove(image_list[0])
        train_list = [item for sublist in image_list for item in sublist]
        random.shuffle(train_list)
        split_dict['train'] = train_list
        splits.append(split_dict)

        print(split_dict['train'])

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def create_acdc_folds(output_dir, image_dir, fold_num=5):
    ppl_files = os.listdir(image_dir)
    trainset_size = len(ppl_files) * 90 // 100
    valset_size = len(ppl_files) * 5 // 100
    testset_size = len(ppl_files) * 5 // 100

    splits = []
    for i in range(fold_num):
        random.shuffle(ppl_files)
        split_dict = {}
        split_dict['train'] = ppl_files[0:trainset_size]
        split_dict['val'] = ppl_files[trainset_size: trainset_size + valset_size]
        split_dict['test'] = ppl_files[trainset_size + valset_size:]
        splits.append(split_dict)
        print(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)



# some dataset may include an independent test set
def create_splits_1(output_dir, image_dir, test_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    test_files = subfiles(test_dir, suffix=".npy", join=False)

    trainset_size = len(npy_files) * 3 // 4
    valset_size = len(npy_files) - trainset_size

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        trainset = []
        valset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient)
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient)
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = test_files

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


if __name__ == "__main__":
    root_dir = "../../../DATA/brats"
    image_dir = "../../../DATA/brats/imgs"
    create_acdc_folds(root_dir, image_dir)


