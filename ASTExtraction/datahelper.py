'''
:date: 20191205
:intro: To load data from sqlite database files; organize data to help model train and test
:user:yangshouguo
'''
import sqlite3
import pickle,os,sys
from collections import defaultdict
from random import randint
import Tree
from numpy import random
from copy import deepcopy
import random as rd
from tqdm import tqdm
import numpy as np
import hashlib
from multiprocessing import Pool
import logging
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(root_dir)
l = logging.getLogger("datahelper.log")
logpath = os.path.join(os.path.dirname(__file__),"log")
if not os.path.exists(logpath):
    os.mkdir(logpath)
l.addHandler(logging.FileHandler(os.path.join(logpath,"datahelper.log")))
l.addHandler(logging.StreamHandler())
l.setLevel(logging.ERROR)
'''
功能：
1. 加载多个数据库，进行数据查询
2. 将查询得到的数据中AST进行恢复
3. 将多个数据库中数据根据文件名和函数名进行组对；文件名和函数名相同，架构或endian或位数或编译优化不同，则为positive，否则，只要文件名以及函数名不同则negative
4. 数据库schema
    sql = """create table if not exists function (
                    id integer primary key,
                    function_name varchar(255),
                    address varchar(30),
                    elf_file_name varchar(255),
                    elf_path varchar(255),
                    arch_name varchar(10),
                    endian varchar(8),
                    optimization varchar(8),
                    ast_pick_dump text,
                    pseudocode text,
                    caller integer,
                    callee integer
            )"""
'''
class DataHelper:
    def __init__(self):
        self.db = None
        from difflib import SequenceMatcher
        self.sequecematcher = SequenceMatcher()

    def load_database(self, filename):
        '''
        :param filename: sqlite database file path
        :return: db connection
        '''

        return sqlite3.connect(filename)

    def do_query(self, db_path, optimization="O2", where="", limit=""):
        '''
        :param db_path:
        :param optimization:
        :return: A generator
        '''
        db = self.load_database(db_path)
        sql = """select function_name, elf_file_name, elf_path, arch_name, caller, callee, ast_pick_dump
         from function """ + where + " " +limit
        l.info(sql)

        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            yield list(line)
        cur.close()
        db.close()

    def get_functions(self, db_path, start=-1, end=-1, where_suffix=None):
        '''
        :param db_path:
        :param start: 开始查询的偏移量
        :param end: 结束查询的偏移量
        :param where_suffix: 限制查询的where子句
        :return: A generator : 第一个元素是 函数的一些信息，
        (函数名，elf路径名，elf文件名，调用函数数量，被调用函数数量，ast编码向量
        ) 第二个是函数的ast
        '''
        db = self.load_database(db_path)
        suffix = ""
        sql = """select function_name, elf_path, elf_file_name, caller, callee, ast_encode, ast_pick_dump
                        from function """
        if start < 0 :
            if where_suffix:
                suffix = where_suffix
        else:
            suffix = " limit %d,%d" % (start, end-start)
        sql += suffix
        l.info("[Query]"+sql)
        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            yield line[:-1], pickle.loads(line[-1].encode("utf-8"), encoding="latin1")
        cur.close()
        db.close()

    def get_function_ast_encode(self, db_path):
        '''
        :param db_path: sqlite 数据库路径
        :return: 函数名，elf文件路径，ast编码向量（需要使用 json.loads 恢复成list）
        '''
        db = self.load_database(db_path)
        sql = """select function_name, elf_path, ast_encode
         from function """

        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            yield line
        cur.close()
        db.close()

    def make_target_set(self, db_list = [], rate=2):
        '''
        :param db_list: list: 数据库文件列表
        :param rate: 非同源函数对数量与同源函数对数量的比例
        :return: list: {A:[target functions list]}
        '''
        pool = Pool(len(db_list)-1)
        source_db = db_list[0]
        pairs = []
        result = []
        for target_db in db_list[1:]:
            res = pool.apply_async(self._make_cross_architecture_paires, (source_db, target_db, rate))
            result.append(res)
        pool.close()
        pool.join()
        for res in result:
            (hologomous, nonhologomoes) = res.get()
            pairs += hologomous
            pairs += nonhologomoes

        return pairs

    def loads_ast_tree(self, ast_pickle_dumps):
        return pickle.loads(ast_pickle_dumps.encode("utf-8"), encoding="latin1")

    def recvore_ast_tree(self, data, idx):
        '''
        :param data: data[idx]是使用 pick.dumps 保存的字符串
        :return: 将data[idx]使用pick.loads 加载的Tree对象
        '''
        for d in data:
            d[idx] = self.loads_ast_tree(d[idx])
        return list(filter(lambda d:d[idx].size()>3, data))# 过滤AST大小小于3的AST

    def _make_cross_architecture_paires(self, arch1_db, arch2_db, non_homologous_pair_rate = 2, limit=""):
        '''
        给定两个数据库文件，读取其中函数，提取共同函数，根据共同函数构建同源函数对和非同源函数对.
        :param arch1_db: 架构1 数据库文件路径
        :param arch2_db: 架构2 数据库文件路径
        :param non_homologous_pair_rate: int; 非同源函数对数与同源函数对的比例
        :return:同源函数对数组 和 非同源函数对数组
        '''
        def random_choose_N(list, N):
            ln = len(list)
            if N > ln:
                return list
            return rd.sample(list, N)


        def get_componet_and_binary_name(path):
            # 返回组件名和二进制文件名
            id1 = path.rfind("/")
            id2 = path.rfind("/", 0, id1 - 1)
            return path[id2 + 1:]

        homologous_pair = []
        nonhomologous_pair = []
        arch1_data = list(self.do_query(arch1_db, limit=limit)) #, where="where elf_file_name not like 'libcrypto%' or 'libssl%' ", limit=limit))
        arch2_data = list(self.do_query(arch2_db, limit=limit)) #, where="where elf_file_name not like 'libcrypto%' or 'libssl%' ", limit=limit))

        arch1_data = self.recvore_ast_tree(arch1_data, idx=-1)
        arch2_data = self.recvore_ast_tree(arch2_data, idx=-1)
        arch2_len = len(list(arch2_data))
        same_source_function = defaultdict(list) #第一个值表示该函数在arch1_data中的下标，第二个值表示该函数在arch2_data中下标;长度小于2，表示该函数不存在同源函数对 key:function_name+elf_file_name
        for idx, line in enumerate(arch1_data):
            key = "+".join([line[0], get_componet_and_binary_name(line[2])]) # function_name + binary_name
            same_source_function[key].append(idx)

        for idx, line in enumerate(arch2_data):
            key = "+".join([line[0], get_componet_and_binary_name(line[2])])
            same_source_function[key].append(idx)
        names = list(map(lambda x:x if len(same_source_function[x])>1 else None, same_source_function))
        names = list(filter(lambda x:x is not None, names))
        # names = random_choose_N(names, N=1000) #for buildroot train
        l.info("From Two database selected {} homologous functions".format(len(names)))
        for name in tqdm(names, desc="cross arch dataset producing......"):
            try:
                homologous_pair.append((arch1_data[same_source_function[name][0]], arch2_data[same_source_function[name][1]], 1)) #with label
            except IndexError as err:
                print(err)
            TRY = 20
            while TRY>0:
                TRY-=1
                idx2 = randint(0, arch2_len-1)
                if idx2 == same_source_function[name][1]:
                    continue
                else:
                    try:
                        nonhomologous_pair.append((arch1_data[same_source_function[name][0]], arch2_data[idx2], -1))
                    except IndexError as err:
                        print(err)
                    if non_homologous_pair_rate <= 0:
                        break
                    non_homologous_pair_rate -= 1
        hp = deepcopy(homologous_pair)
        np = deepcopy(nonhomologous_pair)
        del  homologous_pair, nonhomologous_pair
        return hp, np
# ============== ast 哈希 encode ======================
    def hashencode_ast(self, ast):
        mu = hashlib.md5()
        mu.update(str(ast.op).encode("utf-8"))
        for child in ast.children:
            self.hashencode_ast(child)
            mu.update(child.hash_op.encode("utf-8"))
        ast.hash_op = mu.hexdigest()
    def visit_tree(self, ast, node_list):  # 先序遍历
        node_list.append(ast.hash_op)
        for child in ast.children:
            self.visit_tree(child, node_list)
    def get_hash_code(self, ast):  # 对树进行哈希编码，返回编码之后的向量
        self.hashencode_ast(ast)
        res = []
        self.visit_tree(ast, res)
        return res
    def similarity_ast_has_encode(self, hash_encode1, hash_encode2):
        '''
        计算两个哈希之后树生成的向量的相似度
        :param hash_encode1:
        :param hash_encode2:
        :return:
        '''
        self.sequecematcher.set_seqs(hash_encode1, hash_encode2)
        return self.sequecematcher.quick_ratio()

    def generate_ast_hash_encode(self, db):
        '''
        生成数据库的哈希更新code
        :param db:
        :return:
        '''
        #首先对数据库插入一列属性  ast_hash_encode
        db_conn = self.load_database(db)
        cur = db_conn.cursor()
        # 修改表结构，添加一列
        sql_add_column = """alter table function add column ast_hash_encode TEXT"""
        try:
            cur.execute(sql_add_column)
            db_conn.commit()
        except Exception:
            print("sql %s failed" % sql_add_column)
        finally:
            cur.close()
            db_conn.close()

        encode_result = []
        result = []
        p = Pool(processes=10)
        for function_name, elf_file_name, elf_path, arch_name, ast_pick_dump in self.do_query(db):
            # self.get_hash_code(self.loads_ast_tree(ast_pick_dump))
            result.append(((p.apply_async(self.get_hash_code, (self.loads_ast_tree(ast_pick_dump),))),function_name, elf_path))
        p.close()
        p.join()
        for r, function_name, elf_path in result:
            encode_result.append((pickle.dumps(r.get()), function_name, elf_path))
        db_conn = self.load_database(db)
        cur = db_conn.cursor()
        sql_updatetable = "update function set ast_hash_encode=? where function_name=? and elf_path=?"
        cur.executemany(sql_updatetable, encode_result)
        db_conn.commit()
        cur.close()
        db_conn.close()
    def get_ast_hash_encode(self, db):
        db_conn = self.load_database(db)
        cur = db_conn.cursor()
        sql = """select function_name, elf_path, ast_pick_dump ,ast_hash_encode from function"""
        cur.execute(sql)
        for l in cur.fetchall():
            t = list(l)
            t[-1] = pickle.loads(t[-1])
            t[-2] = self.loads_ast_tree(t[-2])
            yield t
        cur.close()
        db_conn.close()
# ============== ast 哈希 encode ====================== end
    def get_cross_archtecture_pair(self, *archs, limit=""): #生成跨架构数据集
        pairs = []
        arch_record = set()
        for arch1 in archs:
            for arch2 in archs:
                if arch1 == arch2 or (arch1, arch2) in arch_record:
                    continue
                arch_record.add((arch1,arch2))
                arch_record.add((arch2,arch1))
                l.info("[I] Combining %s and %s" %(arch1, arch2))
                homologous_pair, non_homologous_pair = self._make_cross_architecture_paires(arch1, arch2, limit=limit)
                pairs += (homologous_pair+non_homologous_pair)
        sp = random.permutation(len(pairs))
        pairs = np.array(pairs)
        l.info("[I] Dataset preparation finshed")
        return pairs[sp]


    def make_cross_optimization_paires(self, arch1_db, arch2_db):
        pass


    def make_cross_optimization_architecure_paires(self, db):
        pass

def mk_ast_hash_encode_dataset():
    dh = DataHelper()
    dbs = ["/root/data/openssl/openssl-x86.sqlite", "/root/data/openssl/openssl-arm.sqlite",
           "/root/data/openssl/openssl-ppc.sqlite", "/root/data/openssl/openssl-x64.sqlite"]
    # for db in dbs:
    #     dh.generate_ast_hash_encode(db)
    all_function_pairs =[] #[(ast1, ast2, label)]
    for idx in range(len(dbs)):
        s_db = dbs[idx]
        s_functions = list(dh.get_ast_hash_encode(s_db))
        for idx2 in range(idx+1, len(dbs)):
            t_db = dbs[idx2]
            t_functions = list(dh.get_ast_hash_encode(t_db))
            print("source db: %s , target db %s" %(s_db, t_db))
            same_name_functions = defaultdict(list)
            for idt, t_func in enumerate(t_functions):
                same_name_functions[t_func[0]].append(idt)
            for idx, s_func in enumerate(s_functions):
                sims = []
                same_name_functions[s_func[0]].append(idx)
                for idy, t_func in enumerate(t_functions):
                    sims.append((dh.similarity_ast_has_encode(s_func[-1], t_func[-1]), idy))
                sims.sort(key=lambda x:x[0], reverse=True)
                i = 0
                count = 3
                while i < len(sims) and count>0:
                    if t_functions[sims[i][1]][0] != s_functions[0]: #非同名函数
                        all_function_pairs.append((s_func[2], t_functions[sims[i][1]][2], -1))
                        count-=1
                    i+=1
            for key in same_name_functions:
                if len(same_name_functions[key]) == 2:
                    s_idx = same_name_functions[key][1]
                    t_idx = same_name_functions[key][0]
                    try:
                        all_function_pairs.append((s_functions[s_idx][2],
                                               t_functions[t_idx][2], 1))
                    except Exception as e:
                        print(e)
    import torch
    torch.save(all_function_pairs, "/root/data/cross_arch_dataset_with_ast_hash_encode.pth")



def make_target_set():
    dh = DataHelper()
    dataset = dh.make_target_set(["/root/data/openssl/openssl-x86.sqlite", "/root/data/openssl/openssl-arm.sqlite"])
    print(len(dataset))


if __name__ == '__main__':
    mk_ast_hash_encode_dataset()