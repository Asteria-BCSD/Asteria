import os, sys

root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root)
from datahelper import DataHelper
from config import parse_args
import torch
import utils
from model import SimilarityTreeLSTM
import torch.nn as nn
import numpy as np
from metrics import Metrics
import torch.optim as optim
from sklearn.metrics import auc, roc_curve
from utils import Trainer
import logging
import json

logger = logging.getLogger("train.py")
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("T_train.log"))
logger.setLevel(logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #
args = parse_args()
dataset = torch.load(os.path.join(root, "./data/train_dataset_buildroot_demo.pth"))
all = len(dataset)
np.random.shuffle(dataset)
proportion = 0.8  #
split_idx = int(all * proportion)
train_dataset, test_dataset = dataset[0:split_idx], dataset[split_idx:]
logger.info("==> Size of train data \t : %d" % len(train_dataset))
logger.info("==> Size of test data \t : %d" % len(test_dataset))
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
model = SimilarityTreeLSTM(
    args.vocab_size,
    args.input_dim,
    args.mem_dim,
    args.hidden_dim,
    args.num_classes,
    device
)
criterion = nn.BCELoss()
logger.info("[CUDA] available " + str(args.cuda))
logger.info("args" + str(args))
if args.optim == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.lr, weight_decay=args.wd)
model.to(device), criterion.to(device)
trainer = Trainer(args, model, criterion, optimizer, device)
best = - float("inf")
metrics = Metrics(8)
for epoch in range(args.epochs):
    train_loss = trainer.train(train_dataset)
    train_loss, train_pred = trainer.test(train_dataset)
    test_loss, test_preds = trainer.test(test_dataset)
    train_pred = train_pred[:, 1]
    test_preds = test_preds[:, 1]
    train_labels = utils.get_labels(train_dataset).squeeze(1)
    train_targets = utils.map_labels_to_targets(train_labels, args.num_classes)
    train_mse = metrics.mse(train_pred, train_labels)
    acc = metrics.accuracy(train_pred, train_labels)  # label！！！
    fpr, tpr, threshold = roc_curve(train_labels, train_pred)
    train_auc = auc(fpr, tpr)
    logger.info("==> Epoch {}, Train \t Loss: {}\t Auc: {}\tMSE{} \t Accuracy{}".format(
        epoch, train_loss, train_auc, train_mse, acc
    ))

    test_lables = utils.get_labels(test_dataset).squeeze(1)
    test_targets = utils.map_labels_to_targets(test_lables, args.num_classes)
    test_mse = metrics.mse(test_preds, test_lables)
    test_acc = metrics.accuracy(test_preds, test_lables)
    fpr, tpr, t = roc_curve(test_lables, test_preds)
    test_auc = auc(fpr, tpr)
    logger.info("==> Epoch {}, Test \t Loss: {}\tAuc: {}\tMSE{} \t Accuracy{} \t".format(
        epoch, test_loss, test_auc, test_mse, test_acc
    ))

    if best < test_auc:
        best = test_auc
        checkpoint = {
            'model': trainer.model.state_dict(),
            'optim': trainer.optimizer,
            'auc': test_auc, 'mse': test_mse,
            "args": args, 'epoch': epoch
        }
        logger.warning("==> New optimum found, checkpoint everything now...")
        logger.info("auc is %f" % test_auc)
        logger.info("fpr=%s" % json.dumps(fpr.tolist()))
        logger.info("tpr=%s" % json.dumps(tpr.tolist()))
        torch.save(checkpoint, "%s.pt" % (os.path.join(args.save, args.expname)))