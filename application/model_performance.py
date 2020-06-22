#encoding=utf-8
'''
yangshouguo
'''
import sys
sys.path.append("/root/treelstm.pytorch")
import logging, json
from application.application import Application
from datahelper import DataHelper
import numpy as np
logger = logging.getLogger("model_performance")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler("model_performance.log"))
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from sklearn.metrics import roc_curve, auc, roc_auc_score
import Tree
from collections import defaultdict
from datetime import datetime
def parse_arg():
    pass

class Model_Performance():
    '''

    '''
    def __init__(self, checkpoint_path, model_name):
        self.chkpoint = checkpoint_path
        self.app = Application(load_path=checkpoint_path, model_name = model_name)
        self.datahelper = DataHelper()

    def CrossArchTestFast(self, arch1, arch2):
        '''
        :param arch1: path to sqlite database
        :param arch2: path to sqlite database
        :return:
        '''
        pairs = self.datahelper.get_cross_archtecture_pair(arch1, arch2)

        result = []
        lables = []
        pool = mp.Pool(processes=10)
        start_time = datetime.now()
        pairs = pairs[0:10]
        for (source, target, label) in tqdm(pairs, desc="Sim Computing."):
            source_tree = source[-1]
            target_tree = target[-1]
            lables.append(label)
            result.append(pool.apply_async(self.app.similarity_tree, (source_tree, target_tree,)))
        pool.close()
        pool.join()
        predictions = []
        for res in result:
            predictions.append(res.get())
        time_cost = (datetime.now()-start_time).seconds
        fpr, tpr, thresholds = roc_curve(np.array(lables), np.array(predictions))
        logger.info("===> architecture info : %s vs %s" % (arch1, arch2))
        logger.info("===> time: %s" % (datetime.now()))
        logger.info("model: %s" % self.chkpoint)
        logger.info("compute %d function pairs, which cost %d seconds; %f pairs per sec" % (
        len(pairs), time_cost, len(pairs) * 1.0 / time_cost))
        logger.info("predictions= %s" % json.dumps(predictions))
        logger.info("labels= %s" % json.dumps(lables))
        logger.info("fpr= %s" % json.dumps(fpr.tolist()))
        logger.info("tpr= %s" % json.dumps(tpr.tolist()))
        logger.info("thresholds= %s" % json.dumps(thresholds.tolist()))



def test_cross_arch():
    mp = Model_Performance("/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch.pt", model_name="treelstm")
    # mp = Model_Performance("/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarchflatlstm.ptlstm", model_name="flatlstm")
    dbs = ["/root/data/openssl/openssl-x86.sqlite", "/root/data/openssl/openssl-arm.sqlite",
           "/root/data/openssl/openssl-ppc.sqlite", "/root/data/openssl/openssl-x64.sqlite"]
    for sdb in dbs:
        for tdb in dbs:
            if sdb == tdb:
                continue
            mp.CrossArchTestFast(sdb, tdb)

if __name__ == '__main__':
    test_cross_arch()
