'''
:date: 20191205
:intro: To load data from sqlite database files; organize data to help model train and test
:user:yangshouguo
'''
import sqlite3
import pickle,os,sys
from collections import defaultdict
from random import randint
from Tree import Tree
from numpy import random
from copy import deepcopy
import random as rd
from tqdm import tqdm
import numpy as np
import hashlib
from multiprocessing import Pool
import logging
l = logging.getLogger("datahelper.log")
logpath = os.path.join(os.path.dirname(__file__), "log")
if not os.path.exists(logpath):
    os.mkdir(logpath)
l.addHandler(logging.FileHandler(os.path.join(logpath, "datahelper.log")))
l.addHandler(logging.StreamHandler())
l.setLevel(logging.INFO)
'''
Functions: 1. Load multiple databases and perform data query 2. Restore the data obtained from the query in AST 3. Group the data in multiple databases according to the file name and function name; the file name and function name are the same, the structure or endian or If the number of bits or compilation optimization is different, it is positive, otherwise, as long as the file name and function name are different, it is negative4. Database schema
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
            yield line[:-1], pickle.loads(line[-1], encoding="latin1")
        cur.close()
        db.close()

    def get_function_ast_encode(self, db_path, table_name = 'normaltreelstm', where_suffix = ""):
        '''

        '''
        db = self.load_database(db_path)
        sql = """select function_name, elf_path, ast_encode from %s  %s """ % (table_name, where_suffix)

        cur = db.cursor()
        lines = cur.execute(sql)
        for line in lines:
            yield line
        cur.close()
        db.close()

    def make_target_set(self, db_list = [], rate=2):
        '''

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

        '''
        for d in data:
            d[idx] = self.loads_ast_tree(d[idx])
        return list(filter(lambda d:d[idx].size()>3, data))#

    def _make_cross_architecture_paires(self, arch1_db, arch2_db, non_homologous_pair_rate = 2, limit=""):
        '''

        '''
        def random_choose_N(list, N):
            ln = len(list)
            if N > ln:
                return list
            return rd.sample(list, N)


        def get_componet_and_binary_name(path):
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
        same_source_function = defaultdict(list) # key:function_name+elf_file_name
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
# ============== ast  encode ======================
    def hashencode_ast(self, ast):
        mu = hashlib.md5()
        mu.update(str(ast.op).encode("utf-8"))
        for child in ast.children:
            self.hashencode_ast(child)
            mu.update(child.hash_op.encode("utf-8"))
        ast.hash_op = mu.hexdigest()
    def visit_tree(self, ast, node_list):  # preorder
        node_list.append(ast.hash_op)
        for child in ast.children:
            self.visit_tree(child, node_list)
    def get_hash_code(self, ast):  #
        self.hashencode_ast(ast)
        res = []
        self.visit_tree(ast, res)
        return res
    def similarity_ast_has_encode(self, hash_encode1, hash_encode2):
        '''
        :param hash_encode1:
        :param hash_encode2:
        :return:
        '''
        self.sequecematcher.set_seqs(hash_encode1, hash_encode2)
        return self.sequecematcher.quick_ratio()

    def generate_ast_hash_encode(self, db):
        '''

        :param db:
        :return:
        '''
        #  ast_hash_encode
        db_conn = self.load_database(db)
        cur = db_conn.cursor()
        # ，
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
# ============== ast  encode ====================== end
    def get_cross_archtecture_pair(self, *archs, limit=""): #
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
                    if t_functions[sims[i][1]][0] != s_functions[0]: #
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