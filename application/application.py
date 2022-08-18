# encoding=utf-8
'''
yangshouguo@iie.ac.cn
'''
import os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(ROOT)
import logging
import torch, numpy
import gevent, datetime
from math import exp
from config import parse_args
from Tree import Tree
import time
from sklearn.metrics.pairwise import cosine_similarity
from model_arch import Sieamens
from model import get_tree_flat_nodes
from tqdm import tqdm

logger = logging.getLogger("application.py")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
from multiprocessing import Pool, cpu_count
import json

DB_CONNECTION = None



class Application():
    '''
    Computing similarities between asts based on trained model.
    Functionality:
    1. load model
    2. encode ast
    3.calculate similarity between two asts
    '''

    def __init__(self, load_path, threshold=0.5, cuda=False, model_name="treelstm", model=None):
        '''
        :param load_path: path to saved model
        :param cuda: bool : Whether to use GPU, it must be False when multi-process encoding of ast in the database
        :param model_name: [treelstm, treelstm_boosted, binarytreelstm]
        :param threshold: Threshold, value range (0~1), it is recommended to be around 0.8
        :param model: A network variant of Tree-LSTM
        '''
        self.args = parse_args(["--cuda"])  # TODO argparse
        self.model = None
        self.embmodel = None  # The model used for computational coding
        self.model_name = model_name
        self.db = None
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        if model is not None:
            self.model = model
        else:
            if model_name == "treelstm":
                from model import SimilarityTreeLSTM
                self.model = SimilarityTreeLSTM(
                    self.args.vocab_size,
                    self.args.input_dim,
                    self.args.mem_dim,
                    self.args.hidden_dim,
                    self.args.num_classes,
                    device=self.device
                )

            else:
                self.model = Sieamens(
                    self.args.vocab_size,
                    self.args.input_dim,
                    self.args.mem_dim,
                    self.args.hidden_dim,
                    self.args.num_classes,
                    modelstr=self.args.model_selector)

        self.load_model(load_path, self.model)

    def load_model(self, path, model):
        '''
        :param path: path to model
        :param model: model name
        '''
        if not os.path.isfile(path):
            print("model path {} non-exists".format(path))
            raise Exception
        checkpoint = torch.load(path, map_location=self.device)
        if "auc" in checkpoint:
            logger.info("checkpoint loaded: auc %f , mse: %f \n args %s" % (
            checkpoint['auc'], checkpoint['mse'], checkpoint['args']))
        
        model_dic = checkpoint['model']
        
        model.load_state_dict(model_dic)
        model.eval()
        # self.model.to(self.device)
        self.embmodel = model.embmodel
        self.embmodel.to(self.device)

    def encode_ast(self, tree):
        '''
        :param tree: Tree object
        :return: numpy vector
        '''
        with torch.no_grad():
            state, hidden = self.embmodel(tree, get_tree_flat_nodes(tree).to(self.device))
            return state.detach().squeeze(0).cpu().numpy()

    def similarity_tree_with_correction(self, ltree, rtree, lcallee, rcallee):
        '''
        :param ltree: AST1
        :param rtree: AST2
        :param lcallee: No. of callees in AST1 (caller, callee)
        :param rcallee: No. of callees in AST2 (caller, callee)
        :return:
        '''

        sim_tree = self.similarity_tree(ltree, rtree)
        # return sim_tree
        # scale lcallee and rcallee in case that zero vector
        lcallee = list(map(lambda x: x + 1, lcallee))
        rcallee = list(map(lambda x: x + 1, rcallee))

        cs = cosine_similarity([lcallee], [rcallee])[0][0]  # （caller,callee）cosine distance
        scale = exp(0 - abs(lcallee[-1] - rcallee[-1]))
        return sim_tree * scale * cs

    def similarity_tree(self, ltree, rtree):
        '''
        Use the neural network treelstm to calculate the similarity between two abstract syntax trees, there is no sequential relationship between the trees
        :param ltree:  AST
        :param rtree:  AST
        '''
        with torch.no_grad():
            if self.model_name == 'treelstm' or self.model_name == "binarytreelstm":
                res = self.model(ltree, rtree)[0][1].item()
            else:
                res = self.model(ltree, rtree)[1].item()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return res

    def similarity_vec(self, lvec, rvec):
        '''
        calculate similarity between encoded vectors
        :param lvec: numpy.ndarray or torch.tensor
        :param rvec:
        :return: similarity 0~1
        '''
        if type(lvec) is list:
            lvec = numpy.array(lvec)
            rvec = numpy.array(rvec)
        if type(lvec) is numpy.ndarray:
            lvec = torch.from_numpy(lvec).to(self.device).float()
            rvec = torch.from_numpy(rvec).to(self.device).float()
            lvec = lvec.unsqueeze(0)
            rvec = rvec.unsqueeze(0)
        with torch.no_grad():
            if self.model_name in ['treelstm', 'treelstm_boosted']:
                res = self.model.similarity(lvec, rvec)[0][1].float().item()
            else:
                res = self.model.similarity(lvec, rvec)[1].float().item()
            return res

    def __del__(self):
        if self.db:
            self.db.close()

    def func_wrapper(self, func, *args):
        self.Timeout = 350  # 8 minutes
        gevent.Timeout(self.Timeout, Exception).start()
        s = datetime.datetime.now()
        try:
            g = gevent.spawn(func, args[0])
            g.join()
            return g.get()
        except Exception as e:
            # print("Timeout Error. ")
            e = datetime.datetime.now()
            # print("Real run time is %d" % (e - s).seconds)
            r = numpy.zeros((1, 150))
            return r

if __name__ == '__main__':
    # end_to_end_evaluation()
    # time_consumption_statistics()
    # db_names = [""]
    # app.encode_ast_in_db("/root/data/firmwares/vul.sqlite")
    # app.encode_ast_in_db("/root/data/firmwares/Dlinkfirmwares.sqlite")
    db_path = "/root/data/firmwares/Netgearfirmwares.sqlite"
    # app.encode_ast_in_db(db_path, table_name='NormalTreeLSTM', where_suffix=" where elf_file_name like 'libcrypto%' ")
