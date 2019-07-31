import torch.nn as nn
import torch.utils.data as data
import torch
from torch.nn import functional as F
import sys

sys.path.insert(0, '../PixPlot')
sys.path.insert(0, '../LDA')
import PixPlot.utils.process_images as PP
import requests
import LDA.generate_semantic_vector as SC
from Network.utils import *
from Network.SemanticSpaceDataset import get_loader
from Network.CustomLoss import add_customLoss
from Network.LinearModel import *

dictionary_path = '/home/jacopo/Desktop/TTN_Pytorch/src/outputDictionary.json'
k_elements_file = './k_elements.json'
KDtree_path = './KDtree.pickle'

semantic_dim = 40
mapping_dim = 2

#batch_size = 10
batch_size = 32
learning_rate = 1e-3


########################################################################################################################

def main():
    generate_KDtree = False
    generate_k_neighbors = False

    # number of neighbors present in the each line of neighbors matrix (plus the element itself, total k+1 elements per row)
    k = 200
    #k = 3

    num_epoch = 500
    log_step = 20

    data_dictionary = json.load(open(dictionary_path))

    if generate_KDtree:
        topic_probs = data_dictionary['topic_probs']
        create_KDtree(topic_probs, KDtree_path)

    if generate_k_neighbors:
        topic_probs = data_dictionary['topic_probs']
        compute_K_elements_matrix(load_KDtree(KDtree_path), k, topic_probs, k_elements_file)

    k_elements = load_k_elements(k_elements_file)

    print('***Getting DataLoader...')
    data_loader = get_loader(data_dictionary, k_elements, k, batch_size, shuffle=False, num_workers=4)

    print('***Building Neuron and initializing weights...')
    model = get_model(dimIn=40, dimOut=2)
    model.apply(init_weights)
    if torch.cuda.is_available():
        model.cuda()

    print('***Loss, optimizer, etc...')
    criterion = nn.TripletMarginLoss(margin=2.0)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    total_step = len(data_loader)

    positions = []

    constraints = []
    constraints_criterions = []

    print('Starting training')
    model.train()

    post_positions = False

    for epoch in range(num_epoch): #USE xrange IF USING PYTHON 2.7
        count = 0

        for i, (filename, current, positive, negative, anchor_weight) in enumerate(data_loader):

            current = to_var(current)
            positive = to_var(positive)
            negative = to_var(negative)

            mapping_current = model(current)
            mapping_positive = model(positive)
            mapping_negative = model(negative)

            loss = criterion(mapping_current,mapping_positive,mapping_negative)

            if len(anchor_weight) > 0:
                losses = []
                for i in range(len(constraints)):
                    anchor_positions = []
                    weights = []
                    for el in anchor_weight:
                        anchor_positions.append(constraints[i]['map'])
                        weights.append(el[i])
                    anchor_positions = to_var(torch.FloatTensor(anchor_positions))
                    weights = to_var(torch.FloatTensor(weights))
                    #losses.append(constraints_criterions[i](weights, anchor_positions, mapping_current))
                    losses = constraints_criterions[i](weights, anchor_positions, mapping_current)
                    loss += losses[0]



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %f, LR: %f'
                      % (epoch, num_epoch, i, total_step, loss.item(), learning_rate))
                #print(model(anchor))
            if epoch == 1:
                tmp_clone = mapping_current
                positions.extend(tmp_clone.cpu().detach().numpy())

            if post_positions == True:
                tmp_clone = mapping_current
                tmp_clone = tmp_clone.cpu().detach().numpy().tolist()

                #data to send to the server. "type" specifies the data is consisting of new positions to update the webgl visualization
                data = {}
                data['type'] = 'new_positions'
                for i, fn in enumerate(filename):
                    tmp_clone[i].append(count)
                    data[fn] = tmp_clone[i]
                    count += 1
                #print (data)

                try:
                    page = requests.post('http://127.0.0.1:8051', json=data)
                    if page.text == 'new_constraint':

                        try:
                            constraint_resp = requests.get('http://127.0.0.1:8051/?constraint')
                            constraint = json.loads(constraint_resp.text.replace(r'\u',''))
                            print('New anchor defined in visualization, generating LDA representation of it......')
                            #This only works with this 40 dimension semantic space, since LDA is trained specifically for Wikipedia dataset. In other case, other embedding must be used
                            LDA_extract = SC.SemanticSpaceConverter()
                            constraint_vec = LDA_extract.generate_constraint(constraint['text'])
                            constraint_pos = constraint['pos']
                            print('Constraint semantic representation:'+ str(constraint_vec))
                            data_loader.dataset.gen_anchor_weights(constraint_vec)
                            constraints.append({'map': constraint_pos, 'semantic': constraint_vec})

                            #Adds the new loss component to the loss function, to satisfy the new constraint (anchor)
                            constraints_criterions = add_customLoss(constraints_criterions)

                            #free the memory
                            LDA_extract = None
                            constraint = None

                        except:
                            print('Problem getting new constraint!')

                except:
                    print ('Problem with POST request:'+ str(page.status_code))

        if epoch == 1:
            os.chdir('../PixPlot')
            PP.RealTimePixPlot(data_dictionary['filenames'], data_dictionary['topic_probs'], positions)
            positions = None
            post_positions = True



main()