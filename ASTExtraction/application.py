#encoding=utf-8
'''
2019年12月16日
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
DB_CONNECTION = None #全局数据库连接
old_datahelper = old_DataHelper()
datahelper = old_datahelper
class Application():
    '''
    这个类加载训练好的模型，对提供的ast进行计算相似度，提供以下功能：
    1. 给定模型路径，加载模型
    2. 给定ast，计算编码
    3. 给定两个ast，计算相似度
    '''
    def __init__(self, load_path, threshold = 0.5, cuda = False, model_name = "treelstm" , model = None):
        '''
        :param load_path: 模型加载路径
        :param cuda: bool : 是否使用GPU, 对数据库中ast进行多进程编码时必须为False
        :param model_name: 使用的model类，可用 [treelstm, treelstm_boosted, binarytreelstm]
        :param threshold: 阈值,取值范围 (0~1)，建议在0.5左右
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
        :param path: 模型路径
        :param model: 使用的model类，例如 SplitedTreeLSTM
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
        :param tree: root节点， Tree 对象
        :return: 一个numpy向量 （64 or 150维）
        '''
        with torch.no_grad():
            state, hidden = self.embmodel(tree, get_tree_flat_nodes(tree).to(self.device))
            return state.detach().squeeze(0).cpu().numpy()

    def similarity_treeencoding_with_correction(self, ltree, rtree, lcallee, rcallee):
        '''
        :param ltree: encoding vector
        :param rtree: AST2
        :param lcallee:AST1对应函数的callee个数 (caller, callee)
        :param rcallee:AST2多应函数的callee个数 (caller, callee)
        :return:
        '''

        sim_tree = self.similarity_vec(ltree, rtree)
        # return sim_tree
        # scale lcallee and rcallee in case that zero vector
        lcallee = list(map(lambda x: x + 1, lcallee))
        rcallee = list(map(lambda x: x + 1, rcallee))

        cs = cosine_similarity([lcallee], [rcallee])[0][0]  # （caller,callee）对之间的余弦距离
        scale = exp(0 - abs(lcallee[-1] - rcallee[-1]))
        return sim_tree * scale * cs

    def similarity_tree_with_correction(self, ltree, rtree, lcallee, rcallee):
        '''
        :param ltree: AST1
        :param rtree: AST2
        :param lcallee:AST1对应函数的callee个数 (caller, callee)
        :param rcallee:AST2多应函数的callee个数 (caller, callee)
        :return:
        '''

        sim_tree = self.similarity_tree(ltree, rtree)
        #return sim_tree
        # scale lcallee and rcallee in case that zero vector
        lcallee=list(map(lambda x:x+1, lcallee))
        rcallee=list(map(lambda x:x+1, rcallee))

        cs = cosine_similarity([lcallee], [rcallee])[0][0] # （caller,callee）对之间的余弦距离
        scale = exp(0 - abs(lcallee[-1]-rcallee[-1]))
        return sim_tree * scale * cs

    def similarity_tree(self, ltree, rtree):
        '''
        利用神经网络treelstm 计算两个 抽象语法树 之间的相似度，树之间没有先后关系
        :param ltree:  第一个树
        :param rtree:  第二个树
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
            if self.model_name in ['treelstm', 'treelstm_boosted']:
                res = self.model.similarity(lvec, rvec)[0][1].float().item()
            else:
                res = self.model.similarity(lvec, rvec)[1].float().item()
            return res

    def get_conn(self, db):
        global DB_CONNECTION
        if DB_CONNECTION is None:
            global datahelper
            DB_CONNECTION = datahelper.load_database(db)
            #DB_CONNECTION.execute('pragma journal_mode = wal')
        
        return DB_CONNECTION

    def encode_ast_in_db(self, db_path, table_name="function"):
        '''
        对数据库中的树利用训练好的模型进行编码，得到向量，保存到新的表
        :param db_path: 数据库路径
        :return:
        '''
        db_conn = self.get_conn(db_path)
        cur = db_conn.cursor()
        #创建表
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
        对functions中的 func_ast 利用神经网络进行编码
        :param functions:
        :param table_name: the new table to save ast encodings
        :return:
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
            #sql_updatetable = "insert into %s (function_name, elf_path, ast_encode)"  % table_name + " values (?,?,?)"
            #for t in result:
            cur.executemany(sql_update, result)
            cur.close()
            db_conn.commit()
        except Exception as e:
            db_conn.rollback()
            print("Error when INSERT [{}]\n".format(sql_update))
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


def aaaaaa():
    def average_time(size_times, size_start, size_end):
        times = list(map(lambda x:x[1], filter(lambda x:x[0] >= size_start and x[0] <= size_end, size_times)))
        if len(times) <1:
            print("Range from {} to {} no Record in function average_time.".format(size_start, size_end))
            return 0
        return sum(times)/len(times)


    app = Application(load_path =
                      "/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch_buildroot_all.pt",
                      model_name="treelstm")
    dataset = torch.load("/root/treelstm.pytorch/data/cross_arch.pth") # 数据集
    time_recorder = [] # (size, encoding_time)
    encoding_tuple = []
    for idx in tqdm(range(len(dataset))):
        source, target, label = dataset[idx]
        source_tree = source[-1]
        target_tree = target[-1]
        if source_tree.size()>300 or target_tree.size() > 300:
            continue
        source_encode = app.encode_ast(source_tree)
        target_encode = app.encode_ast(target_tree)
        encoding_tuple.append((source_tree.size(),
                               torch.tensor(source_encode).unsqueeze(0),
                               torch.tensor(target_encode).unsqueeze(0)))
    with torch.no_grad():
        for s_size, s, t in encoding_tuple:
            start_time = datetime.datetime.now()
            app.model.similarity(s, t)[0][1].float().item()
            end_time = datetime.datetime.now()
            time_recorder.append((s_size, (end_time-start_time).microseconds))

    with open("time_consumption_statistics.log", "a") as f:
        f.write("Time : {} For calculations\n".format(datetime.datetime.now()))
        ave = average_time(time_recorder, 1, 50)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(1, 50, ave))
        ave = average_time(time_recorder, 51, 100)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(51, 100, ave))
        ave = average_time(time_recorder, 101, 150)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(101, 150, ave))
        ave = average_time(time_recorder, 151, 200)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(151, 200, ave))
        ave = average_time(time_recorder, 201, 250)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(201, 250, ave))
        ave = average_time(time_recorder, 251, 300)
        f.write("Size range in [{} , {}] Average Encoding Time: {}\n".format(251, 300, ave))
        f.write("{}\n".format(str(time_recorder)))


def generate_openssl_ast_encode():
    '''
    生成openssl数据集的ast的编码
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

