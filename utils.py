from tqdm import tqdm
import torch
import math
from torch.autograd import Variable
from math import floor
from torch import multiprocessing as mp

def get_labels(dataset):
    return torch.Tensor(list(map(lambda x: x[-1], dataset))).view(-1,1)

def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    if label==-1:
        target[0,0] = 1
    else:
        target[0,1] = 1
    return target

def map_labels_to_targets(labels , num_classes):
    targets = torch.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        targets[idx] = map_label_to_target(label, num_classes)
    return targets

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            source_function,  target_function,  label = dataset[indices[idx]]
            ltree = source_function[-1]
            rtree = target_function[-1]
            # ltree = source_function
            # rtree = target_function
            target = map_label_to_target(label, self.args.num_classes)
            target = target.to(self.device)
            output = self.model(ltree, rtree)
            output = output.view(1,-1)
            #To
            assert output.size() == target.size()
            loss = self.criterion(output, target)
            #print(output)
            del output
            total_loss += loss.item()
            loss.backward()
            del loss
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()#free memory

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset, threshold=0.0):
        '''
        :return:
        '''
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros((len(dataset),2), dtype=torch.float, device='cpu')
            #indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                source_function, target_function, label = dataset[idx]
                #ltree = source_function[-1]
                #rtree = target_function[-1]
                ltree = source_function
                rtree = target_function

                target = map_label_to_target(label, self.args.num_classes)
                target = target.to(self.device)
                output = self.model(ltree, rtree)
                output = output.view(1, -1)
                # Ensure that the dimensions of the calculation loss must be consistent
                assert output.size() == target.size()
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu') #
                predictions[idx] = output
            # predictions = torch.Tensor(list(map(lambda x: 1 if x > threshold else 0, predictions)))
        return total_loss / len(dataset), predictions


