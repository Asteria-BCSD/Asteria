from copy import deepcopy
import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._l1loss = torch.nn.L1Loss()
    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)

    def accuracy(self, pred, labels):
        x = deepcopy(pred)
        y = deepcopy(labels)

        return 1-torch.div(self._l1loss(x,y),torch.FloatTensor([2]))

if __name__ == '__main__':
    #test accuracy
    pred = torch.Tensor([-1,1,1,1,-1])
    labels = torch.Tensor([1,1,1,1,1])
    m = Metrics(1)
    print(m.accuracy(pred, labels))
    print(m.mse(pred,labels))
    ml = torch.nn.MSELoss()
    print(ml(pred, labels))
