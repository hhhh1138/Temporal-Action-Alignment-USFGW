#!/usr/bin/python3

# import random
import numpy as np
import torch.utils.data
import pickle
from collections import Counter

def to_np(tensor):
    return tensor.detach().cpu().numpy()


# buffer for old sequences (robustness enhancement: old frames are sampled from the buffer during training)
class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.vfnames = []
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.gt_labels = []
        self.softlabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, vfname, features, transcript, framelabels, gt_labels, softlabels=None):
        assert features.shape[1] == len(framelabels)
        # assert features.shape[1] == len(gt_labels)
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.vfnames.append(vfname)
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            self.gt_labels.append(gt_labels)
            if softlabels is not None:
                self.softlabels.append(softlabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.vfnames[self.next_position] = vfname
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            self.gt_labels[self.next_position] = gt_labels
            if softlabels is not None:
                self.softlabels[self.next_position] = softlabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # update frame selectors
        self.frame_selectors = []
        for seq_idx in range(len(self.features)):
            self.frame_selectors.extend([ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ])

    def count(self, labels):
        ct = Counter(labels)
        counts = np.zeros(self.n_classes, dtype=np.int)
        for i, k in ct.items():
            counts[i] = k
        return counts

    def random(self):
        i = np.random.choice(len(self.frame_selectors))
        return self.frame_selectors[i] # return sequence_idx and frame_idx within the sequence

    def n_frames(self):
        return len(self.frame_selectors)

    def load(self, buffer_save, dataset=None):

        if isinstance(buffer_save, str):
            with open(buffer_save, 'rb') as fp:
                buffer_save = pickle.load(fp)

        for v in buffer_save:
            transcript, framelabels, gt = buffer_save[v]
            self.vfnames.append(v)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            self.gt_labels.append(gt)
            if dataset is not None:
                self.features.append(dataset.features[v])

            # statistics for prior and mean lengths
            self.instance_counts.append( self.count(transcript) )
                    # np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( self.count(framelabels) )
                    # np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )

        if dataset is not None:
            # update frame selectors
            self.frame_selectors = []
            for seq_idx in range(len(self.features)):
                self.frame_selectors.extend([ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ])

    def save(self, buffer_save_file):
        buffer_save = {}
        for t in range(len(self.vfnames)):
            v = self.vfnames[t]
            transcript = self.transcript[t]
            label = self.framelabels[t]
            gt   = self.gt_labels[t]
            buffer_save[v] = [ transcript, label, gt ]
        with open(buffer_save_file, 'wb') as fp:
            pickle.dump(buffer_save, fp)
