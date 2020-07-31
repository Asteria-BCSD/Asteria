#encoding=utf-8
'''
20191206
yangshouguo
yangshouguo@iie.ac.cn
'''
import os,sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
import logging
import torch, numpy
import gevent, datetime
from math import exp
from config import parse_args
from Tree import Tree
import time
from sklearn.metrics.pairwise import cosine_similarity
from model import get_tree_flat_nodes
from datahelper import DataHelper as old_DataHelper
from tqdm import tqdm
logger = logging.getLogger("application.py")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)
from multiprocessing import Pool, cpu_count
import json
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
DB_CONNECTION = None # global database connection object
old_datahelper = old_DataHelper()
datahelper = old_datahelper
class Application():
    '''
    This class loads the trained model and use the model to encode an ast into a vector and calculate the similarity between asts.
    '''
    def __init__(self, load_path, threshold = 0.5, cuda = False, model_name = "treelstm" , model = None):
        '''
        :param load_path: the path to saved model. e.g. "saved_model.pt"
        :param cuda: bool : whether GPU is utilized when model calculation. Notice that you need install the torch gpu version.
        :param model_name: the model type used. Only the "treelstm" is supported in this script.
        :param threshold: decided what similarity scores are outputted.
        :param model: A network variant of Tree-LSTM
        '''
        self.args = parse_args(["--cuda"]) #TODO argparse
        self.model =None
        self.embmodel = None #真正用于计算编码的模型
        self.model_name = model_name
        self.db = None
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        if model is not None:
            self.model = model
        else:
            if model_name=="treelstm":
                from model import SimilarityTreeLSTM
                self.model = SimilarityTreeLSTM(
                    self.args.vocab_size,
                    self.args.input_dim,
                    self.args.mem_dim,
                    self.args.hidden_dim,
                    self.args.num_classes,
                    device=self.device
                )
        self.load_model(load_path, self.model)

    def load_model(self, path, model):
        '''
        :param path: path to saved model
        :param model: the model loaded
        :return:
        '''
        if not os.path.isfile(path):
            print("model path %s non-exists" % path)
            raise Exception
        checkpoint = torch.load(path, map_location=self.device)
        if "auc" in checkpoint:
            logger.info("checkpoint loaded: auc %f , mse: %f \n args %s" %(checkpoint['auc'], checkpoint['mse'], checkpoint['args']))
        model.load_state_dict(checkpoint['model'])
        model.eval()
        # self.model.to(self.device)
        self.embmodel =model.embmodel
        self.embmodel.to(self.device)


    def encode_ast(self, tree):
        '''
        :param tree: An Tree object instance
        :return: a numpy vector （64 or 150 d）
        '''
        with torch.no_grad():
            state, hidden = self.embmodel(tree, get_tree_flat_nodes(tree).to(self.device))
            return state.detach().squeeze(0).cpu().numpy()

    def similarity_treeencoding_with_correction(self, ltree, rtree, lcallee, rcallee):
        '''
        :param ltree: tree encoding vector
        :param rtree: tree encoding vector
        :param lcallee: (caller, callee) corresponds to ltree
        :param rcallee: (caller, callee) corresponds to rtree
        :return:
        '''

        sim_tree = self.similarity_vec(ltree, rtree)
        # return sim_tree
        # scale lcallee and rcallee in case that zero vector
        lcallee = list(map(lambda x: x + 1, lcallee))
        rcallee = list(map(lambda x: x + 1, rcallee))

        cs = cosine_similarity([lcallee], [rcallee])[0][0]  # cosine distance
        scale = exp(0 - abs(lcallee[-1] - rcallee[-1]))
        return sim_tree * scale * cs

    def similarity_tree_with_correction(self, ltree, rtree, lcallee, rcallee):
        '''
        :param ltree: AST1
        :param rtree: AST2
        :param lcallee: (caller, callee) corresponds to ltree
        :param rcallee:(caller, callee) corresponds to rtree
        :return:
        '''

        sim_tree = self.similarity_tree(ltree, rtree)
        #return sim_tree
        # scale lcallee and rcallee in case that zero vector
        lcallee=list(map(lambda x:x+1, lcallee))
        rcallee=list(map(lambda x:x+1, rcallee))

        cs = cosine_similarity([lcallee], [rcallee])[0][0] # （caller,callee）
        scale = exp(0 - abs(lcallee[-1]-rcallee[-1]))
        return sim_tree * scale * cs

    def similarity_tree(self, ltree, rtree):
        '''
        calculate the similarity of two asts
        :param ltree:  first tree
        :param rtree:  first tree
        :return:
        '''
        with torch.no_grad():
            if self.model_name=='treelstm' or self.model_name=="binarytreelstm":
                res = self.model(ltree, rtree)[0][1].item()
            else:
                res = self.model(ltree, rtree)[1].item()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return res

    def similarity_vec(self, lvec, rvec):
        '''
        calculate the similarity of two ast encodings
        :param lvec: numpy.ndarray or torch.tensor
        :param rvec:
        :return: similairty score ranges 0~1.
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

    def get_conn(self, db):
        # return the database connection
        global DB_CONNECTION
        if DB_CONNECTION is None:
            global datahelper
            DB_CONNECTION = datahelper.load_database(db)
        return DB_CONNECTION

    def encode_ast_in_db(self, db_path, table_name="function"):
        '''
        Encode the asts in db file "db_path" and save the encoding vectors into table 'table_name'
        :param db_path: path to sqlite database
        '''
        db_conn = self.get_conn(db_path)
        cur = db_conn.cursor()
        # create table if not exists
        sql_create_new_table = """CREATE TABLE if not exists %s (
                            function_name varchar (255),
                            elf_path varchar (255),
                            ast_encode TEXT,
                            primary key (function_name, elf_path)
                            );""" % table_name
        try:
            cur.execute(sql_create_new_table)
            db_conn.commit()
        except Exception as e:
            logger.error("sql [%s] failed" % sql_create_new_table)
            logger.error(e)
        finally:
            cur.close()
        to_encode_list = []
        global datahelper
        for func_info, func_ast in old_datahelper.get_functions(db_path):
            to_encode_list.append((func_info, func_ast))
        encode_group = to_encode_list
        logger.info("Encoding for %d ast" % len(encode_group))
        self.encode_and_update(db_path, encode_group, table_name)

    def __del__(self):
        if self.db:
            self.db.close()

    def func_wrapper(self, func, *args):
        # to execute the function in a limited period time
        self.Timeout = 350 #8 minutes
        gevent.Timeout(self.Timeout, Exception).start()
        s = datetime.datetime.now()
        try:
            g = gevent.spawn(func, args[0])
            g.join()
            return g.get()
        except Exception as e:
            #print("Timeout Error. ")
            e = datetime.datetime.now()
            #print("Real run time is %d" % (e - s).seconds)
            r = numpy.zeros((1,150))
            return r
    def encode_and_update(self, db_path, functions, table_name):
        '''
        encode asts of the functions into vectors
        :param functions: a list contains asts to be encoded
        :param table_name: the new table to save ast encodings
        '''
        db_conn = self.get_conn(db_path)
        p = Pool(processes=10)
        res = []
        count = 1
        for func_info, func_ast in functions:
            res.append((p.apply_async(self.func_wrapper, (self.encode_ast, func_ast)), func_info[0], func_info[1])) # func_info[0] is function name; func_info[1] is elf_path
            count+=1
        #try:
        p.close()
        p.join()
        result = []
        try:
            logger.info("Fetching encode results!")
            for idx, r in tqdm(enumerate(res)):
                result.append((json.dumps(r[0].get().tolist()), r[1], r[2]))
            logger.info("All encode fetched!")
        except Exception as e:
            print("Exception when fetching {}".format(str(e)))
        try:
            logger.info("Writing encoded vectors to database")
            cur = db_conn.cursor()
            sql_update = """ 
            update {} set ast_encode=? where function_name=? AND elf_path=?
            """.format(table_name)
            cur.executemany(sql_update, result)
            cur.close()
            db_conn.commit()
        except Exception as e:
            db_conn.rollback()
            print("Error when INSERT [{}]\n".format(sql_update))
            print(e)
        #cur.cose()



def generate_openssl_ast_encode():
    '''
    sample code for ast encoding
    :return:
    '''
    db_names = ["ARM.sqlite",  "PPC.sqlite",  "X64.sqlite",  "X86.sqlite"]
    db_path = "/sqlite"
    app = Application(load_path="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch.pt", )
    for name in db_names:
        p = os.path.join(db_path, name)
        app.encode_ast_in_db(p)

def parse_args_in_app():
    '''
    :return:The args in dict
    '''
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--dbpath", type=str,
                    help="the path to sqlite db file where asts are saved")
    ap.add_argument("--checkpoint", type=str,
                    default=os.path.join(root_dir, "saved_model.pt"))
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args_in_app()
    if args.dbpath:
        app = Application(load_path=args.checkpoint)
        app.encode_ast_in_db(args.dbpath)

