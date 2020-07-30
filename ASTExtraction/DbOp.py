#encoding=utf-8
"""
Providing the operation on database
"""
import sqlite3
import os, sys, logging
DBNAME = ""
l = logging.getLogger("DbOp.py")
l.setLevel(logging.WARNING)
l.addHandler(logging.StreamHandler())
l.addHandler(logging.FileHandler("DbOp.log"))
l.handlers[0].setFormatter(logging.Formatter("%(filename)s : %(message)s"))
root = os.path.dirname(os.path.abspath(__file__))
class DBOP():
    def __init__(self, db_path=None):
        self._db_path = db_path
        if not self._db_path or self._db_path=="":
            self.init_db_name()
        self.open_db()

    def __del__(self):
        self.db.commit()
        self.db.close()

    def init_db_name(self):
        '''
        To initialize the path of database
        '''
        dir = os.path.dirname(os.path.abspath(__file__))
        self._db_path = os.path.join(dir,"all.sqlite")

    def open_db(self):
        db = sqlite3.connect(self._db_path)
        db.text_factory = str
        db.row_factory = sqlite3.Row
        self.db = db
        self.create_schema()
        db.execute("analyze") #To boost the query

    def create_schema(self):
        '''
        :return:
        '''
        cur = self.db.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        # table "function"
        sql = """create table if not exists function (
                        id integer,
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
                        callee integer,
                        ast_encode text,
                        constraint pk PRIMARY KEY (function_name, elf_path)
                )"""
        cur.execute(sql)
        #create index
        cur.close()
        l.info("created table function")

    def insert_function(self, func_name, address, elf_file_name, elf_path, arch_name, endian,
                        optimization, ast_pick_dump, psesudocode, caller, callee):
        '''
        :param func_name: str
        :param address: str
        :param elf_file_name: str
        :param elf_path: str
        :param arch_name: str
        :param endian : str
        :param optimization: str
        :param ast_pick_dump: str
        :param psesudocode: str
        :param caller: int
        :param callee: int
        :return:
        '''
        sql = """insert into function(function_name, address, elf_file_name, elf_path,
                    arch_name, endian, optimization, ast_pick_dump, pseudocode, caller, callee)
                    values (?,?,?,?,?,?,?,?,?,?,?)"""
        values =(func_name, address, elf_file_name, elf_path, arch_name, endian,
                        optimization, ast_pick_dump, psesudocode, caller, callee)
        try:
            cur = self.db.cursor()
            cur.execute(sql, values)
            cur.close()
            l.info("elfpath:functionname %s:%s" % (elf_path, func_name))
        except :
            # l.error("[E]insert error. args : %s\n %s" % (str(values), Exception.message))
            l.error("Function %s's AST insertion to database fails" % (func_name))

