#!/usr/bin/python3

import numpy as np


class LengthModel(object):
    
    def n_classes(self):
        return 0

    def score(self, length, label):
        return 0.0

    def max_length(self):
        return np.inf


class PoissonModel(LengthModel):
    
    def __init__(self, model, 
                max_length = 2000, renormalize = True, bg_limit=0):

        super(PoissonModel, self).__init__()
        if type(model) == str:
            self.mean_lengths = np.loadtxt(model)
        else:
            self.mean_lengths = model
        self.num_classes = self.mean_lengths.shape[0]

        self.max_len = max_length

        self.renormalize = renormalize
        self.bg_limit = bg_limit
        self.precompute_prob(self.mean_lengths, self.max_len)

    def precompute_prob(self, mean_lengths, max_len):
        self.poisson = np.zeros((max_len, self.num_classes))

        # precompute normalizations for mean length model
        self.norms = np.zeros(mean_lengths.shape)
        assert (mean_lengths > 0).all()
        if self.renormalize:
            self.norms = np.round(mean_lengths) * np.log(np.round(mean_lengths)) - np.round(mean_lengths)
            for c in range(len(mean_lengths)):
                logFak = 0
                for k in range(2, int(mean_lengths[c])+1):
                    logFak += np.log(k)
                self.norms[c] = self.norms[c] - logFak

        # precompute Poisson distribution
        self.poisson[0, :] = -np.inf # length zero can not happen
        logFak = 0
        for l in range(1, max_len):
            logFak += np.log(l)
            self.poisson[l, :] = l * np.log(mean_lengths) - mean_lengths - logFak - self.norms

    def n_classes(self):
        return self.num_classes

    def score(self, length, label):
        if length >= self.max_len:
            # return -np.inf
            return -10000
        else:
            return self.poisson[length, label]

    def max_length(self):
        return self.max_len

    def update_mean_lengths(self, buffer):

        self.mean_lengths = np.zeros( (self.num_classes), dtype=np.float32 )
        for i, label_count in enumerate(buffer.label_counts):
            self.mean_lengths += label_count 

        instances = np.zeros((self.num_classes), dtype=np.float32)
        for instance_count in buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 and  self.mean_lengths[i] > 0 
                        else sum(self.mean_lengths) / sum(instances) for i in range(self.num_classes) ] )

        if self.bg_limit > 0:
            self.mean_lengths[0] = min(self.mean_lengths[0], self.bg_limit)

        self.precompute_prob(self.mean_lengths, self.max_len)

