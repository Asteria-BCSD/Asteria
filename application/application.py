#encoding=utf-8
'''

'''
import os,sys
f = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(f,"../"))
import logging
import torch, numpy
import gevent, datetime
from config import parse_args
from model import get_tree_flat_nodes
from Tree import Tree
import time
from model_others import Sieamens
from model import SimilarityTreeLSTM
from datahelper import DataHelper
from tqdm import tqdm
logger = logging.getLogger("application.py")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
from multiprocessing import Pool, cpu_count
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

DB_CONNECTION = None
datahelper = DataHelper()

class Application():
    '''
    This class loads the trained model, calculates the similarity of the provided ast, and provides the following functions:
     1. Given the model path, loading the model
     2. Given the ast, calculating the code  
    3. Given two asts, calculating the similarity
    '''
    def __init__(self, load_path, threshold = 0.84, cuda = False, model_name = "treelstm" ):
        '''
        :param load_path: path to saved model
        :param cuda: bool : with or witout GPU when computing
        :param model_name: model selector
        :param threshold:
        '''
        self.args = parse_args(["--cuda"]) #
        self.model =None
        self.embmodel = None #the model to encode ast
        self.model_name = model_name
        self.db = None
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        if model_name=="treelstm":
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
        :param path: path to saved model
        :param model: model selector
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
        :param tree: Tree class instance
        :return: numpy vector
        '''

        with torch.no_grad():
            state, hidden = self.embmodel(tree, get_tree_flat_nodes(tree).to(self.device))
            return state.squeeze(0).cpu().numpy()

    def similarity_tree_hash_sim(self, ltree, rtree):
        pass
    #TODO

    def similarity_tree(self, ltree, rtree):
        '''
        calculate the similarity between two trees
        :param ltree:
        :param rtree:
        :return:
        '''
        with torch.no_grad():
            if self.model_name=='treelstm':
                res = self.model(ltree, rtree)[0][1].item()
            else:
                res = self.model(ltree, rtree)[1].item()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res

    def similarity_vec(self, lvec, rvec):
        '''
        calculate the similarity between two tree encodings
        :param lvec: numpy.ndarray or torch.tensor
        :param rvec:
        :return: float: similarity
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
            if self.model_name=='treelstm':
                res = self.model.similarity(lvec, rvec)[0][1].float().item()
            else:
                res = self.model.similarity(lvec, rvec)[1].float().item()
            return res

    def get_conn(self, db):
        global DB_CONNECTION
        if DB_CONNECTION is None:
            global datahelper
            DB_CONNECTION = datahelper.load_database(db)
            DB_CONNECTION.execute('pragma journal_mode = wal')
        
        return DB_CONNECTION

    def encode_ast_in_db(self, db_path):
        '''
        Encode ASTs into vectors in database file
        :param db_path: path to sqlite db file
        :return:
        '''
        db_conn = self.get_conn(db_path)
        cur = db_conn.cursor()
        #alter table
        sql_add_column = """alter table function add column ast_encode TEXT"""
        try:
            cur.execute(sql_add_column)
            db_conn.commit()
        except Exception:
            logger.error("sql %s failed" % sql_add_column)
        finally:
            cur.close()
        to_encode_list = []
        global datahelper
        for func_info, func_ast in tqdm(datahelper.get_functions(db_path), desc="Load Database: "):
            to_encode_list.append((func_info, func_ast))
        encode_group = []
        count = 0
        for to_encode_tuple in tqdm(to_encode_list, desc= "Encoding: "):
            count+=1
            encode_group.append(to_encode_tuple)
        logger.info("Encoding for %d ast" % count)
        self.encode_and_update(encode_group)


    def __del__(self):
        if self.db:
            self.db.close()

    def func_wrapper(self, func, *args):
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
    def encode_and_update(self, functions):
        '''
        encode functions
        :param functions:
        :return:
        '''
        db_conn = self.get_conn(db_path)
        p = Pool(processes=10)
        res = []
        count = 1
        for func_info, func_ast in tqdm(functions, desc="Adding to pool : "):
            res.append((p.apply_async(self.func_wrapper, (self.encode_ast, func_ast)), func_info[0], func_info[1])) # func_info[0] is function name; 1 is elf_path
            count += 1
        #try:
        p.close()
        p.join()
        result = []
        try:
            logger.info("Fetching encode results!")
            for idx, r in enumerate(res):
                result.append((json.dumps(r[0].get().tolist()), r[1], r[2]))
            logger.info("All encode fetched!")
        except Exception as e:
            print("Exception when fetching {}".format(str(e)))
        try:
            logger.info("Writing encoded vectors to database")
            cur = db_conn.cursor()
            sql_updatetable = "update function set ast_encode=? where function_name=? and elf_path=?"
            #for t in result:
            db_conn.execute("PRAGMA synchronous=OFF")
            db_conn.execute("BEGIN TRANSACTION")
            cur.executemany(sql_updatetable, result)
            db_conn.execute("COMMIT")
            cur.close()
            logger.info("Total number of rows updated %d" % db_conn.total_changes)
        except Exception as e:
            print("Error when UPDATE\n")
            print(e)
        #cur.close()

def time_consumption_statistics():
    app = Application(load_path="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch.pt")
    dataset = torch.load("/root/treelstm.pytorch/data/cross_arch.pth")
    AST_SIZE_GROUP = [[] for i in range(5)] # Group and store AST according to different sizes, corresponding to 0~20, 20~40 ..., 80~100. Each group stores 100
    ast_size = 0
    interval = 50
    GROUP_SIZE = 100
    for i in range(len(AST_SIZE_GROUP)):
        ast_size = i*interval
        logger.info("Making size %d" % ast_size)
        for tup in dataset:
            if tup[0][-1].size() > ast_size and tup[0][-1].size() < ast_size+interval:
                AST_SIZE_GROUP[i].append(tup[0][-1])
                if len(AST_SIZE_GROUP[i]) >= GROUP_SIZE:
                    break
            if tup[1][-1].size() > ast_size and tup[1][-1].size() < ast_size+interval:
                AST_SIZE_GROUP[i].append(tup[1][-1])
                if len(AST_SIZE_GROUP[i]) >= GROUP_SIZE:
                    break

    #==times for encoding===
    import datetime
    TIME_encode = [0 for i in range(5)]
    TIME_encode_distribution = []
    ENCODE_SIZE_GROUP = []
    for i in tqdm(range(len(AST_SIZE_GROUP)),desc="encode time"):
        start_time = datetime.datetime.now()
        for ast in AST_SIZE_GROUP[i]:
            s = datetime.datetime.now()
            ENCODE_SIZE_GROUP.append(app.encode_ast(ast))
            e = datetime.datetime.now()
            TIME_encode_distribution.append((ast.size(), (e-s).total_seconds()))
        end_time = datetime.datetime.now()
        TIME_encode[i] = (end_time-start_time).total_seconds()

    # ===
    TIME_SIMILARITY = [0 for i in range(5)]
    source = ENCODE_SIZE_GROUP[0]
    Sim_Time_Statistic = []
    start_time = datetime.datetime.now()
    for i in tqdm(range(len(ENCODE_SIZE_GROUP)),desc="cal time"):
        s = datetime.datetime.now()
        app.similarity_vec(source, ENCODE_SIZE_GROUP[i])
        e = datetime.datetime.now()
        Sim_Time_Statistic.append((TIME_encode_distribution[i][0], (e-s).total_seconds()))
    end_time = datetime.datetime.now()
    time_sim = (end_time-start_time).total_seconds()
    with open("time_consumption_statistics.log","a") as f:
        f.write("#=== Time Statistics for paper\n")
        f.write("\tTime: %s \n" %(datetime.datetime.now()))
        f.write("Encoding Time:\n")
        for i in range(len(TIME_encode)):
            f.write("Size %d~%d: %f\n" % (i, i+interval, TIME_encode[i]/GROUP_SIZE))
        f.write("Encode distribution : %s\n" % str(TIME_encode_distribution))
        f.write("\tCalculation Time:\n")
        f.write("%d calculation cost %f seconds\n" %(len(ENCODE_SIZE_GROUP), time_sim))
        f.write("Time details : %s \n" % str(Sim_Time_Statistic))

if __name__ == '__main__':

    db_path = "../data/vul.sqlite"
    app = Application(load_path="../data/saved_model.pt", )
    app.encode_ast_in_db(db_path)

