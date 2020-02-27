#encoding=utf-8
'''

'''
import os,sys

sys.path.append("/root/treelstm.pytorch/ysg_treelstm")
sys.path.append("/root/treelstm.pytorch")
import logging
import torch, numpy
import gevent, datetime
from config import parse_args
from ysg_treelstm.Tree import Tree
import time
from ysg_treelstm.model_others import Sieamens
from ysg_treelstm.model import SimilarityTreeLSTM
from ysg_treelstm.datahelper import DataHelper
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
    这个类加载训练好的模型，对提供的ast进行计算相似度，提供以下功能：
    1. 给定模型路径，加载模型
    2. 给定ast，计算编码
    3. 给定两个ast，计算相似度
    '''
    def __init__(self, load_path, threshold = 0.5, cuda = False, model_name = "treelstm" ):
        '''
        :param load_path: 模型加载路径
        :param cuda: bool : 是否使用GPU, 对数据库中ast进行多进程编码时必须为False
        :param model_name: 使用的model类，例如 SplitedTreeLSTM
        :param threshold: 阈值,取值范围 (0~1)，建议在0.5左右
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
        :param path: 模型路径
        :param model: 使用的model类，例如 SplitedTreeLSTM
        :return:
        '''
        if not os.path.isfile(path):
            print("model path %s non-exists")
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
        :param tree: root节点， Tree 对象
        :return: 一个numpy向量 （64 or 150维）
        '''
        with torch.no_grad():
            state, hidden = self.embmodel(tree)
            return state.squeeze(0).cpu().numpy()

    def similarity_tree_hash_sim(self, ltree, rtree):
        pass
    #TODO

    def similarity_tree(self, ltree, rtree):
        '''
        利用神经网络treelstm 计算两个 抽象语法树 之间的相似度，树之间没有先后关系
        :param ltree:  第一个树
        :param rtree:  第二个树
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
        传入树已经编码好的向量，计算相似度
        :param lvec: numpy.ndarray or torch.tensor
        :param rvec:
        :return: 一个实数 0~1 之间，
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
        对数据库中的树利用训练好的模型进行编码，得到向量，保存到新的列 ast_encode
        :param db_path: 数据库路径
        :return:
        '''
        db_conn = self.get_conn(db_path)
        cur = db_conn.cursor()
        #修改表结构，添加一列
        sql_add_column = """alter table function add column ast_encode TEXT"""
        try:
            cur.execute(sql_add_column)
            db_conn.commit()
        except Exception:
            logger.error("sql %s failed" % sql_add_column)
        finally:
            cur.close()
        to_encode_list = []
        start = 650000
        end = start+40000
        #end = 1868280
        global datahelper
        for func_info, func_ast in tqdm(datahelper.get_functions(db_path, start=start, end=end), desc="Load Database: "):
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
        对functions中的 func_ast 利用神经网络进行编码
        :param functions:
        :return:
        '''
        db_conn = self.get_conn(db_path)
        p = Pool(processes=10)
        res = []
        count = 1
        for func_info, func_ast in tqdm(functions, desc="Adding to pool : "):
            res.append((p.apply_async(self.func_wrapper, (self.encode_ast, func_ast)), func_info[0], func_info[1])) # func_info[0] is function name; 1 is elf_path
            count+=1
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
    AST_SIZE_GROUP = [[] for i in range(5)] # 按照不同大小对AST进行分组存储，分别对应 0~20, 20~40 ... , 80~100. 每个分组存储100个
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

    #===统计编码时间===
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

    # === 统计不同量级相似度计算时间 量级 1v1, 1v50, 1v100, 1v150, 1v200, 1v250, 1v300
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


def test():
    app = Application(load_path="./data/crossarch.pt")
    dataset = torch.load("./data/cross_arch.pth") #
    for idx in range(len(dataset)):
        source, target, label = dataset[idx]
        source_tree = source[-1]
        target_tree = target[-1]
        source_encode = app.encode_ast(source_tree)
        target_encode = app.encode_ast(target_tree)
        print(str(source_encode))
        # print("source encode:%s \n source_encode %s" %(source_encode, target_encode))
        print("label %s,\t similarity: %f" % (label, app.similarity_vec(source_encode, target_encode)))

        app.encode_ast_in_db(p)

if __name__ == '__main__':

    db_path = "/root/data/firmwares/Netgearfirmwares.sqlite"
    app = Application(load_path="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/train_after_hash_calculated.pt", )
    app.encode_ast_in_db(db_path)
    # app.encode_ast_in_db("/root/data/firmwares/vul.sqlite")
    # app.encode_ast_in_db("/root/data/firmwares/Dlinkfirmwares.sqlite")
