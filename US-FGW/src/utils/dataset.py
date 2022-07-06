#!/usr/bin/python3

import numpy as np
# import random
import os
from torch.utils import data
from collections import Counter

# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video  : ''change from index to label''
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, label2index, shuffle=False):
        self.features = dict()      # element type: np.darray
        self.transcript = dict()    # element type: list
        self.gt_label = dict()      # element type: list
        self.shuffle = shuffle
        self.idx = 0

        # read features for each video
        base_path = base_path.rstrip('/')
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # transcript
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
                # self.transcript[video] = [ line for line in f.read().split('\n')[0:-1] ]
            # gt_label
            with open(base_path + '/groundTruth/' + video + '.txt') as f:
                self.gt_label[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
                # self.gt_label[video] = [ line for line in f.read().split('\n')[0:-1] ]

        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        if self.shuffle:
            np.random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)
        print("shuffle data: ", self.shuffle)

    def videos(self):
        return list(self.features.keys())

    def __getitem__(self, video):
        return self.features[video], self.transcript[video], self.gt_label[video]

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            return video, self.features[video], self.transcript[video], self.gt_label[video]

    def get(self):
        try:
            return self.__next__()
        except StopIteration:
            return self.get()
