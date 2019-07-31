from Network.utils import get_filename
import random
import torch
import torch.utils.data as data
from scipy.stats import multivariate_normal as multi_norm




class SemanticSpaceDataset(data.Dataset):
    def __init__(self, json, k_elements, k):
        self.filenames = json['filenames']
        self.topic_probs = json['topic_probs']
        #self.neighbor_matrix = k_neighbor
        self.neighbors_matrix = k_elements['positives']
        self.farthest_matrix = k_elements['negatives']
        self.K = k
        self.lenght = len(self.filenames)
        self.stage = 0
        self.constraints_weights = []

    def __getitem__(self, index):
        filename = get_filename(self.filenames[index])

        current = torch.Tensor(self.topic_probs[index])
        #current_n = 'arr_'+str(index)
        neighbors = self.neighbors_matrix[index]

        positive_neighbor_idx = random.randint(1, self.K-1)
        positive_idx = int(neighbors[positive_neighbor_idx])
        positive = torch.Tensor(self.topic_probs[positive_idx])

        if self.stage == 0:
            negative = self.stageZero(index)
        elif self.stage == 1:
            negative = self.stageOne(index)
        #farthests = self.farthest_matrix[index]

        #negative_neighbor_idx = random.randint(1, self.K-1)
        #negative_idx = int(farthests[negative_neighbor_idx])
        #negative = torch.Tensor(self.topic_probs[negative_idx])

        #negative_idx = random.randint(self.K, len(self.topic_probs)-1)
        #negative = torch.Tensor(self.topic_probs[negative_idx])

        anchor_weights = []
        if len(self.constraints_weights) > 0:
            for weights in self.constraints_weights:
                anchor_weights.append(weights[index])
        return filename, current, positive, negative, anchor_weights



    def gen_anchor_weights(self, constr_sem_pos):
        weights = []
        print("Generating weights for the new Loss, to satisfy new set anchor....")
        std = 0.16 * np.ones(len(constr_sem_pos))
        for el in self.topic_probs:
            #weights.append(sigmoid(multi_norm(constr_sem_pos, std).pdf(el)))
            weights.append(multi_norm(constr_sem_pos, std).pdf(el))
        print("Weights calculated: "+ str(weights))
        print ("Done!")
        self.constraints_weights.append(weights)
        print (len(self.constraints_weights))



    def stageZero(self, index):
        farthests = self.farthest_matrix[index]

        negative_neighbor_idx = random.randint(1, self.K-1)
        negative_idx = int(farthests[negative_neighbor_idx])
        negative = torch.Tensor(self.topic_probs[negative_idx])

        return negative

    def stageOne(self, index):
        negative_idx = random.randint(self.K, len(self.topic_probs) - 1)
        negative = torch.Tensor(self.topic_probs[negative_idx])

        return negative

    def set_stage(self, stage):
        if stage == 0:
            self.stage = 0
        elif stage > 0:
            self.stage = 1

    def __len__(self):
        return self.lenght


def get_loader(json, k_elements, k, batch_size, shuffle, num_workers):
    dataset = SemanticSpaceDataset(json=json, k_elements=k_elements, k=k)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader