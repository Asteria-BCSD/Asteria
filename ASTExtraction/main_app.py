#encoding=utf-8
'''
author: yangshouguo
date: 20191204
email: 891584158@qq.com
'''

# perform similarity calculation based on 'application.py'
import os, sys
from math import exp
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
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
from collections import defaultdict
l = logging.getLogger("main_app")
log_dir = "./log"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
l.addHandler(logging.FileHandler("./log/main_app.log"))
# l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)
class Asteria():
    '''
    Functions：
    1. calculate the similarity between functions
    2.  calculate the similarity between asts
    3.  calculate the similarity between ast encodings
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
        :param sources:list: source asts
        :param targets:list: target asts
        :return: dict: key is function_name， value is a dict :{'rank':[], 'info':(function_name, elf_path, elf_file_name, caller, callee, ast_encode)}
        '''
        result = defaultdict(dict)
        for (function_name,elf_path,elf_name,scaller, scallee, ast_encode),_ in tqdm(sources):
            res = []
            pool = Pool(processes=cpu_count()-2)
            for (tfunction_name, telf_path,telf_name,tcaller, tcallee, tast_encode),_ in targets:
                if tast_encode is None:
                    print("%s encode not exits" % tfunction_name)
                res.append((pool.apply_async(self.compute_app.similarity_treeencoding_with_correction, (json.loads(ast_encode), json.loads(tast_encode),
                                                                                                        (scaller,scallee),(tcaller, tcallee))),
                            tfunction_name, telf_path,telf_name))
            pool.close()
            pool.join()
            similarity_list = []
            for r in res:
                sim = r[0].get()
                if sim >= threshold:
                    similarity_list.append(((r[1],r[2]), sim))
            similarity_list.sort(key=lambda x: x[1], reverse=True) # 排序
            result[function_name]['rank'] = similarity_list
            result[function_name]['info'] = (function_name, elf_path, telf_name)
        return result

    def prefilter(self, ast1, ast2):
        '''
        :param ast1:
        :param ast2:
        :return: if ast1 and ast2 are too different , return 1.
        '''
        c1 = ast1.num_children
        c2 = ast2.num_children
        if abs(c1-c2) > 30:
            return 1

        if c1/c2 > 3 or c2/c1 > 3:
            return 1
        return 0

    def ast_similarity(self, sources = [], targets = [], astfilter = None, threshold = 0):
        '''
        :param sources: list: source asts
        :param targets: list: target asts
        func_info:[function_name, elf_path, elf_file_name, caller, callee, ast_encode]
        :param astfilter: a filter function to filter out ast pairs which are too different.
        :return: dict: key
        {'rank':[], 'info':(function_name, elf_path, elf_file_name, caller, callee, ast_encode)}
        '''
        result = {}
        N = len(sources)
        i = 0
        TN = len(targets)
        astfilter = self.prefilter

        if astfilter:
            l.error("Filter Function is applied.")
        for s_func_info, s_ast in sources:
            i+=1
            result[s_func_info[0]] = {'rank':'','info':''}
            res = []
            with tqdm(targets, desc="[%d] of %d" % (i,N), dynamic_ncols=True) as t:
                for func_info, t_ast in t:
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
        where_suffix = " limit 0,20" # the number of vulnerability functions does not exceeds 100
        for func in list(self.dh.get_functions(source_db, where_suffix=where_suffix)):
            # limit vul function number
            source_asts.append(func)
            elf_names.add("'"+func[0][2].split('.')[0]+"%'")
        elf_files = " or ".join(elf_names)
        # where_suffix = " where elf_file_name like %s" % elf_files
        #l.info("[DB] the firmware select filter is %s" % where_suffix)
        where_suffix = ""
        for func in self.dh.get_functions(target_db, start=start, end=end, where_suffix=where_suffix):
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
    ap.add_argument("--use_ast", dest='use_ast', action="store_true",
                    help="use ast not ast_encode to compute similarity(default True)")
    ap.add_argument("--no_ast", dest="use_ast", action="store_false")
    ap.add_argument("--threshold", type=str, default="0.9",
                    help="The similarity threshold to filter candidate functions(default 0.9)")
    ap.add_argument("--checkpoint", type=str,
                    default=os.path.join(root_dir, "saved_model.pt"),
                    help="pytorch model checkpoint path")
    ap.add_argument("--model_selector", type=str, default="treelstm", choices=['treelstm','binarytreelstm','lstm'])
    ap.add_argument("--result", type=str, default="", help="file path to save search results")
    return ap.parse_args()
def log_result(args, result):
    result_file = ""

    # if result file is specified with suffix '.json', use json.dump to save result
    if args.result.endswith(".json"):
        if not os.path.exists(os.path.dirname(args.result)):
            l.error("The Specified Output file does not exists.")
            return
        json.dump(result, open(args.result, 'w'))
        return

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

