#encoding=utf-8
'''
date: 20191224
'''

# conduct similarity calculation with trained model.
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(ROOT)
from math import exp
import sqlite3
from datahelper import DataHelper
from application import Application
from Tree import Tree # needed!
from tqdm import tqdm
import datetime
from sklearn.metrics.pairwise import cosine_similarity
import json
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import logging
l = logging.getLogger("main_app")
log_dir = "./log"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
l.addHandler(logging.FileHandler("./log/main_app.log"))
l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)
class Asteria():
    '''
    functionality：
    1. similarity between one function and a group of functions
    2. similarity between two groups of functions
    3. Load functions from database
    '''
    def __init__(self, checkpoint_path, model_selector, cuda=False):
        #cuda = True
        self.dh = DataHelper() # database
        self.checkpoint_path = checkpoint_path
        self.model_selector = model_selector
        self.cuda = cuda
        l.info("[I] Model Loading....")
        self.compute_app = Application(checkpoint_path, cuda=cuda, model_name=model_selector)
        l.info("[I] Model loaded...")

    def ast_encode_similarity(self, sources = [], targets = [], threshold= 0):
        '''
        :param sources: source encoded ast vectors
        :param targets: target encoded ast vectors
        :return: dict: key is function name + elf path，
                        value is a list of similarities: [(similarity, function_name, elf_path)]
        '''
        result = {}
        for function_name,elf_path,ast_encode in tqdm(sources):
            res = []
            pool = Pool(processes=cpu_count()-2)
            ast_encode = json.loads(ast_encode)
            for tfunction_name,telf_path,tast_encode in tqdm(targets, desc="func %s" % function_name):
                if tast_encode is None:
                    print("%s encode not exits" % tfunction_name)
                # res.append((pool.apply_async(self.compute_app.similarity_vec, (json.loads(ast_encode), json.loads(tast_encode))),
                #             tfunction_name, telf_path))
                res.append((self.compute_app.similarity_vec(ast_encode, json.loads(tast_encode)),
                            tfunction_name, telf_path))
            # pool.close()
            # pool.join()
            similarity_list = []
            for r in res:
                sim = r[0].get()
                similarity_list.append((sim, r[1], r[2]))
            similarity_list.sort(key=lambda x: x[0], reverse=True) #
            result[function_name+"+"+elf_path] = similarity_list
        return result

    def ast_similarity(self, sources = [], targets = [], astfilter = None, threshold = 0):
        '''
        :param sources: source  ast
        :param targets: target  ast
        func_info:[function_name, elf_path, elf_file_name, caller, callee, ast_encode]
        :param astfilter:
        :return: dict:
        {'rank':[], 'info':(function_name, elf_path, elf_file_name, caller, callee, ast_encode)}
        '''
        result = {}
        N = len(sources)
        i = 0
        for s_func_info, s_ast in sources:
            i+=1
            result[s_func_info[0]] = {'rank':'','info':''}
            res = []
            for func_info, t_ast in tqdm(targets, desc="%dth for total %d." % (i,N)):
                if astfilter and astfilter(s_ast, t_ast):
                    res.append([func_info, 0])
                else:
                    res.append([func_info, self.compute_app.similarity_tree_with_correction(
                        s_ast, t_ast,[s_func_info[-3],s_func_info[-2]],[func_info[-3], func_info[-2]])])

            res = list(filter(lambda x:x[1]>threshold, res))
            res.sort(key=lambda x: x[1], reverse=True) #

            result[s_func_info[0]]['rank'] = res
            result[s_func_info[0]]['info'] = s_func_info
        return result

    def db_similarity(self, source_db, target_db, ast, threshold, start=-1, end=-1):
        '''
        :param source_db: aught to be vulnerability database path
        :param target_db: firmware function database
        :param ast: True：直接使用ast进行计算相似度；False，使用ast的编码之后的向量进行相似度计算
        :param threshold: float: 0~1
        :param start/end: the position for select in sql limit
        :return:
        '''
        source_asts = []
        target_asts = []
        elf_names = set()
        where_suffix = ""
        #where_suffix = " where function_name like 'EVP_EncodeUpdate' " # TODO delete this line;
        if ast:
            for func in list(self.dh.get_functions(source_db, where_suffix=where_suffix)):
                # limit vul function number
                source_asts.append(func)
                elf_names.add("'"+func[0][2].split('.')[0]+"%'")

            elf_files = "libcrypto%"
            where_suffix = " where function_name is '{0}'  union " \
                           "select function_name, elf_path, elf_file_name, caller, callee, ast_encode, ast_pick_dump from function " \
                           "where function_name is not '{0}' limit 200".format(
                'EVP_EncodeUpdate'
            )
            #l.info("[DB] the firmware select filter is %s" % where_suffix)
            print('Loading target functions...')
            for func in self.dh.get_functions(target_db, start=start, end=end, where_suffix=where_suffix):
                target_asts.append(func)
            return self.ast_similarity(source_asts[:1], target_asts, threshold=threshold)

        else:

            try:
                asts = list(self.dh.get_function_ast_encode(source_db))
            except sqlite3.OperationalError as e:
                asts = list(self.dh.get_functions(source_db, where_suffix=where_suffix))
                newasts = []
                for s_info, s_ast in asts:
                    newasts.append((s_info[0], s_info[1], json.dumps(self.compute_app.encode_ast(s_ast).tolist())))
                asts = newasts
            try:
                elf_files = "libcrypto%"
                where_suffix = " where elf_path like '%%%s%%'" % elf_files
                target_asts = list(self.dh.get_function_ast_encode(target_db, where_suffix=where_suffix))
            except sqlite3.OperationalError as e:
                print(e)
                print("Please use encode_ast_in_db() in application to generate the encodings first")

            return self.ast_encode_similarity(asts, target_asts, threshold=threshold)

import distutils
def parse_args():
    ap = ArgumentParser()
    ap.add_argument("source_db", type=str,
                    help="source sqlite db file path, usually vulnerability function database")
    ap.add_argument("target_db", type=str,
                    help="target sqlite db file path, usually firmware function database")
    ap.add_argument("--use_ast", dest='use_ast', action='store_true', default=False,
                    help="use ast not ast_encode to compute similarity (default True)")
    ap.add_argument("--threshold", type=str, default="0.9",
                    help="The similarity threshold to filter candidate functions(default 0.9)")
    ap.add_argument("--checkpoint", type=str,
                    default="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/train_after_hash_calculated.pt",
                    help="pytorch model checkpoint path")
    ap.add_argument("--model_selector", type=str, default="treelstm", choices=['treelstm','binarytreelstm','lstm'])
    ap.add_argument("--result", type=str, default="", help="file path to save search results")
    return ap.parse_args()
def log_result(args, result):
    result_file = ""
    if len(args.result)>0:
        result_file = args.result
    else:
        result_file = "%s_%s.result" % (os.path.basename(args.source_db), os.path.basename(args.target_db))
    with open(result_file, 'a') as f:
        f.write("******Time: %s ******\n"%str(datetime.datetime.now()))
        f.write("******Args: %s ******\n" % str(args))
        for res in result:
            f.write("[VULFUNC]:%20s\tVULELF:%20s\n" % (result[res]['info'][0], result[res]['info'][2]))
            for func_info, sim in result[res]['rank']:
                f.write("\t|Sim:%5.3f\t|Func:%20s\t|ELFPath:%100s\n\n" %(sim, func_info[0], func_info[1]))

def binary_treelstm_search(): #search bugs with binary treelstm
    args = parse_args()
    args.checkpoint = "/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/binary_tree_lstm_model.pt"
    args.model_selector = "binarytreelstm"
    aa = Asteria(args.checkpoint, model_selector=args.model_selector)
    threshold = float(args.threshold)
    if threshold<0 or threshold>1:
        print("Threshold Value Error! range is 0~1!")
        exit(1)
    result = aa.db_similarity(args.source_db, args.target_db, args.use_ast, threshold=threshold, start=1, end=100)
    log_result(args, result)
    exit(0)

def treelstm_search():
    #search bug with normal treelstm
    args = parse_args()
    aa = Asteria(args.checkpoint, model_selector=args.model_selector)
    threshold = float(args.threshold)
    if threshold < 0 or threshold > 1:
        print("Threshold Value Error! range is 0~1!")
        exit(1)
    result = aa.db_similarity(args.source_db, args.target_db, args.use_ast, threshold=threshold)
    log_result(args, result)

def word_cloud():#search with normal treelstm based on function MD5_Update
    args = parse_args()
    aa = Asteria(args.checkpoint, model_selector=args.model_selector)
    threshold = float(args.threshold)
    if threshold < 0 or threshold > 1:
        print("Threshold Value Error! range is 0~1!")
        exit(1)
    threshold = 0.8
    result = aa.db_similarity(args.source_db, args.target_db, args.use_ast,
                              threshold=threshold, start=200000, end=220000)
    args.source_db = "wordcloud_"
    log_result(args, result)


if __name__ == '__main__':
    treelstm_search()

