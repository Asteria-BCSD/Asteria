import pickle
from pymysql import connect
'''
CREATE TABLE `functions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `functions_name` varchar(255) DEFAULT NULL,
  `address` varchar(30) DEFAULT NULL,
  `elf_file_name` varchar(255) DEFAULT NULL,
  `elf_path` varchar(255) DEFAULT NULL,
  `arch_name` varchar(10) DEFAULT NULL,
  `endian` varchar(8) DEFAULT NULL,
  `optimization` varchar(8) DEFAULT NULL,
  `ast_pick_dump` text,
  `pseudocode` text,
  `caller` int DEFAULT NULL,
  `callee` int DEFAULT NULL,
  `ast_encode` text,
  PRIMARY KEY (`id`)
)
'''

class DataHelper:
    def __init__(self, db_ip='172.17.0.3', usr='root', passwd ='ysg',db_name='netgear'):
        self.db = connect(db_ip, usr, passwd, db_name)

    def load_database(self,f):
        return self.db

    def insert(self, sql):
        cur = self.db.cursor()
        cur.execute(sql)
        #TODO 完成
        cur.close()

    def query(self, sql):
        cur = self.db.cursor()
        cur.execute(sql)
        for line in cur.fetchall():
            yield line
        cur.close()


    def get_functions(self, f, limit=None):
        '''
        :param f:
        :param limit:tuple: (start, end) , for sql limit
        :return:
        '''
        lim = ""
        if limit:
            lim = " limit %d,%d" %(limit[0], limit[1])
        sql = """select functions_name, elf_path, ast_encode, ast_pick_dump
                        from functions """ + lim
        for res in self.query(sql):
            try:
                yield res[:-1], pickle.loads(res[-1].encode('utf-8'), encoding='latin1')
            except Exception as e:
                print(e)
                print("function name is %s" % res[0])


def test_query(dh):
    sql = '''select functions_name, elf_path, ast_encode, ast_pick_dump
                        from functions limit 7'''
    for i in dh.query(sql):
        print(i[0])
    pass
if __name__ == '__main__':
    dh = DataHelper()
    test_query(dh)