#encoding=utf-8
'''
'''

# Vulnerability Search

import os,sys
f = os.path.abspath(os.path.dirname(__file__))
root=os.path.join(f,"../")
sys.path.append(root)
from datahelper import DataHelper
from application import Application
from Tree import Tree
from tqdm import tqdm
import datetime
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
        Functions：
    1. Similarity Calculation between a function and multiple functions
    2. Similarity Calculation between functions and multiple functions
    3. load ASTs from sqlite files
    '''
    def __init__(self, checkpoint_path, model_selector, cuda=False):
        #cuda = True
        self.dh = DataHelper() # 数据库读取
        self.checkpoint_path = checkpoint_path
        self.model_selector = model_selector
        self.cuda = cuda
        l.info("[I] Model Loading....")
        self.compute_app = Application(checkpoint_path, cuda=cuda, model_name=model_selector)
        l.info("[I] Model loaded...")

    def ast_encode_similarity(self, sources = [], targets = [], threshold= 0):
        '''
         :param sources:list: ASTs
        :param targets:list: ASTs
        :return: dict: key is the functionname+elf path，
        value is a list of tuples containing functions with corresponding similarity score: [(similarity, function_name, elf_path)]
        '''
        result = {}
        for function_name, elf_path, ast_encode in tqdm(sources):
            res = []
            result[function_name+"+"+elf_path] = {'rank': '', 'info': ''}
            pool = Pool(processes=cpu_count()-2)
            for tfunction_name,telf_path,tast_encode in tqdm(targets, desc="func %s" % function_name):
                if tast_encode is None:
                    print("%s encode not exits" % tfunction_name)
                res.append((pool.apply_async(self.compute_app.similarity_vec, (json.loads(ast_encode), json.loads(tast_encode))),
                            tfunction_name, telf_path))
            pool.close()
            pool.join()
            similarity_list = []
            for r in res:
                sim = r[0].get()
                similarity_list.append(( (r[1], r[2]), sim))
            similarity_list.sort(key=lambda x: x[0], reverse=True) # 排序
            result[function_name+"+"+elf_path]['rank'] = similarity_list
            result[function_name+"+"+elf_path]['info'] = (function_name, '', elf_path)
        return result

    def ast_similarity(self, sources = [], targets = [], astfilter = None, threshold = 0):
        '''
        :param sources:list: ASTs
        :param targets:list: ASTs
        :return: dict: key is the functionname+elf path，
        value is a list of tuples containing functions with corresponding similarity score: [(similarity, function_name, elf_path)]
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
            res.sort(key=lambda x: x[1], reverse=True) # 排序

            result[s_func_info[0]]['rank'] = res
            result[s_func_info[0]]['info'] = s_func_info
        return result

    def db_similarity(self, source_db, target_db, ast, threshold, start=-1, end=-1):
        '''
         :param source_db: str: path to a sqlite file containing ASTs of vulnerable functions
        :param target_db: str: path to a sqlite file containing ASTs of firmware functions
        :param ast: boolean: True：calculate similarity with AST；False: calculate similarity with AST encodings
        :param start/end: the position for select in sql limit
        :return:
        '''
        if ast:
            readdb_func = self.dh.get_functions
        else:
            readdb_func = self.dh.get_function_ast_encode
        source_asts = []
        target_asts = []
        for func in list(readdb_func(source_db)):
            source_asts.append(func)
        for func in readdb_func(target_db):
            target_asts.append(func)

        if ast:
            return self.ast_similarity(source_asts, target_asts, threshold=threshold)
        else:
            return self.ast_encode_similarity(source_asts, target_asts, threshold=threshold)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("source_db", type=str,
                    help="source sqlite db file path, usually vulnerability function database")
    ap.add_argument("target_db", type=str,
                    help="target sqlite db file path, usually firmware function database")
    ap.add_argument("--use_ast", type=bool, default=False,
                    help="use ast not ast_encode to compute similarity(default True)")
    ap.add_argument("--threshold", type=float, default=0.84,
                    help="The similarity threshold to filter candidate functions(default 0.84)")
    ap.add_argument("--checkpoint", type=str,
                    default="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch_buildroot.pt",
                    help="pytorch model checkpoint path")
    ap.add_argument("--model_selector", type=str, default="treelstm", choices=['treelstm','binarytreelstm','lstm'])
    ap.add_argument("--output", type=str, default="VulSearch.result", help="file path to save search results")
    return ap.parse_args()
def log_result(args, result):
    result_file = ""
    if len(args.output)>0:
        result_file = args.output
    else:
        result_file = "%s_%s.result" % (os.path.basename(args.source_db), os.path.basename(args.target_db))
    with open(result_file, 'a') as f:
        f.write("******Time: %s ******\n"%str(datetime.datetime.now()))
        f.write("******Args: %s ******\n" % str(args))
        for res in result:
            f.write("[VULFUNC]:%20s\tVULELF:%20s\n" % (result[res]['info'][0], result[res]['info'][2]))
            for func_info, sim in result[res]['rank']:
                if sim>args.threshold:
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
    result = aa.db_similarity(args.source_db, args.target_db, args.use_ast, threshold=threshold)
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

if __name__ == '__main__':
    treelstm_search()