from sklearn.neighbors import KDTree as KDT
import pickle
import numpy as np
import torch
import json
from torch.autograd import Variable
import sys
import os




def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm



def create_KDtree(semantic_vecs, KDtree_path):
    print('***Generating KDtree and saving it to file...')
    semantic_vecs = np.asarray(semantic_vecs)
    KDtree = KDT(semantic_vecs, metric = 'euclidean')
    with open(KDtree_path, 'wb') as file:
        pickle.dump(KDtree,file)
    print('Done!')


def load_KDtree(path):
    if os.path.exists(path):
        print('***Loading KDtree...')
        file = open(path, 'rb')
        KDtree = pickle.load(file)
        file.close()
        return KDtree
    else:
        sys.exit('!!! KDtree file needed, create one setting to "True" the "generate_KDtree" variable! Program ending...')


def compute_K_elements_matrix(KDtree, K, vecs, k_elements_file):
    print('***Generating k_neighbors dictionary and saving it to file...')

    vecs = np.asarray(vecs)
    #dist, k_neighbors_idx = KDtree.query(vecs[:len(vecs)], k=K + 1)
    dist, k_neighbors_idx = KDtree.query(vecs[:len(vecs)], len(vecs))

    positives = []
    negatives = []
    for idx, vec in enumerate(vecs):
        #indexes.append(k_neighbors_idx[idx])
        positives.append(k_neighbors_idx[idx][:K].tolist())
        negatives.append(k_neighbors_idx[idx][len(vecs)-K:].tolist())

    with open(k_elements_file, 'w') as file:
        indexes = {'positives': positives, 'negatives': negatives}
        json.dump(indexes, file)
    #np.save(k_neighbors_file, indexes)
    print('Done!')
    return



def load_k_elements(file):
    if os.path.exists(file):
        print('***Loading k_neighbors dictionary...')
        #k_neighbors = np.load(neigh_file)
        k_elements = json.load(open(file))
        return k_elements
    else:
        sys.exit('!!! k_neighbors dictionary needed, create one setting to "True" the "generate_k_neighbors" variable! Program ending...')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_filename(path):
    '''
    Return the root filename of `path` without file extension
    '''
    return os.path.splitext(os.path.basename(path))[0]


