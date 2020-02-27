#encoding=utf-8
'''
author: yangshouguo
date: 2019年12月24日
email: 891584158@qq.com
'''

# 应用训练好的模型，进行函数相似度计算
from ysg_treelstm.datahelper import DataHelper
from ysg_treelstm.application.application import Application
from ysg_treelstm.Tree import Tree
from tqdm import tqdm
import datetime
import json
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
class ABSApp():
    '''
    功能点：
    1. 计算一个函数和多个函数之间的相似度
    2. 计算多个函数和多个函数之间的相似度
    3. 从数据库读取一个或者多个函数
    '''
    def __init__(self, checkpoint_path):
        pass
        self.dh = DataHelper() # 数据库读取
        self.compute_app = Application(checkpoint_path, model_name="treelstm")

    def ast_encode_similarity(self, sources = [], targets = []):
        '''
        :param sources:源ast_encode列表
        :param targets: 目标ast_encode列表
        :return: dict: key是源函数名+elf路径名字， value是list 包含其他函数和相似度 [(similarity, function_name, elf_path)]
        '''
        result = {}
        for function_name,elf_path,ast_encode in tqdm(sources):
            res = []
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
                similarity_list.append((sim, r[1], r[2]))
            similarity_list.sort(key=lambda x: x[0], reverse=True) # 排序
            result[function_name+"+"+elf_path] = similarity_list
        return result

    def ast_similarity(self, sources = [], targets = [], filter = None):
        '''
        :param sources:源ast列表
        :param targets: 目标ast列表
        :param filter: 过滤函数， 应该接受两个参数，第一个为源ast， 第二个为目的ast，如果返回True，则相似度为0
        :return: dict: key是源函数名，rank是一个列表，对应目标函数信息，和相似度; info 存储本函数信息
        {'rank':[], 'info':(function_name, elf_file_name, elf_path, arch_name)}
        '''
        result = {}
        for s_func_info, s_ast in sources:
            res = []
            for func_info, t_ast in enumerate(tqdm(targets)):
                if filter and filter(s_ast, t_ast):
                    res.append((func_info, 0))
                else:
                    res.append((func_info, self.compute_app.similarity_tree(s_ast, t_ast)))
            res.sort(key=lambda x: x[1], reverse=True) # 排序
            result[s_func_info[0]]['rank'] = res
            result[s_func_info[0]]['info'] = s_func_info
        return result

    def db_similarity(self, source_db, target_db, ast):
        '''
        :param source_db:
        :param target_db:
        :param ast: True：直接使用ast进行计算相似度；False，使用ast的编码之后的向量进行相似度计算
        :return:
        '''
        source_asts = []
        target_asts = []
        if ast:
            for func in self.dh.get_functions(source_db):
                source_asts.append(func)
            for func in self.dh.get_functions(target_db):
                target_asts.append(func)
            return self.ast_similarity(source_asts, target_asts)
        else:
            for func in self.dh.get_function_ast_encode(source_db):
                source_asts.append(func)
            for func in self.dh.get_function_ast_encode(target_db):
                target_asts.append(func)
            return self.ast_encode_similarity(source_asts, target_asts)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("source_db", type=str, help="source sqlite db file path, usually represent vulnerability database")
    ap.add_argument("target_db", type=str, help="target sqlite db file path")
    ap.add_argument("--use_ast", type=bool, default=False, help="use ast not ast_encode to compute similarity")
    ap.add_argument("--checkpoint", type=str, default="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch_acc1.0.pt", help="pytorch model checkpoint path")
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    aa = ABSApp(args.checkpoint)
    result = aa.db_similarity(args.source_db, args.target_db, args.use_ast)
    import os
    with open("%s_%s.result" % (os.path.basename(args.source_db), os.path.basename(args.target_db)), 'a') as f:
        for res in result:
            f.write(str(datetime.datetime.now())+"===="+res+"====\n")
            for sim, function_name, elf_path in result[res]:
                if sim>0.9:
                    f.write("%8.5f\t %20s %s\n" %(sim, function_name, elf_path))