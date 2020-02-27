import sys,os
sys.path.append("/root/treelstm.pytorch")
from Tree import Tree
import torch
import datetime,tqdm
from sklearn.metrics import roc_curve, auc
from diaphora_test.factor import difference_ratio
def primesbelow(N):
  # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
  #""" Input N>=6, Returns a list of primes, 2 <= p < N """
  correction = N % 6 > 1
  N = {0:N, 1:N-1, 2:N+4, 3:N+3, 4:N+2, 5:N+1}[N%6]
  sieve = [True] * (N // 3)
  sieve[0] = False
  for i in range(int(N ** .5) // 3 + 1):
    if sieve[i]:
      k = (3 * i + 1) | 1
      sieve[k*k // 3::2*k] = [False] * ((N//6 - (k*k)//6 - 1)//k + 1)
      sieve[(k*k + 4*k - 2*k*(i%2)) // 3::2*k] = [False] * ((N // 6 - (k*k + 4*k - 2*k*(i%2))//6 - 1) // k + 1)
  return [2, 3] + [(3 * i + 1) | 1 for i in range(1, N//3 - correction) if sieve[i]]

def get_ast_prime_and_size(ast):
    '''prime'''
    primes = primesbelow(4096)
    size = 0
    ast_prime = 1

    def visit_tree(root):
        nonlocal ast_prime, size
        size+=1
        ast_prime *= primes[root.op]
        for child in root.children:
            visit_tree(child)
    visit_tree(ast)
    return ast_prime, size

prim_dataset = "./primes_dataset.pkt"
if os.path.exists(prim_dataset):
    dic = torch.load(prim_dataset)
    prime_tuples = dic["prime_tuples"]
    size_tuples = dic["size_tuples"]
    labels = dic["labels"]
    N = len(prime_tuples)
    # import random
    # shuffle_index = list(range(N))
    # random.shuffle(shuffle_index)
    # prime_tuples, size_tuples, labels = prime_tuples[shuffle_index], size_tuples[shuffle_index], labels[shuffle_index]
    e = int(N/100)
    prime_tuples = prime_tuples[:e]
    size_tuples = size_tuples[:e]
    labels = labels[:e]
    print("#====== Dataset Exits and loaded!")
else:
    dataset = torch.load("/root/data/cross_arch_dataset_with_ast_hash_encode_size_gt5.pth")
    print("dataset load done!")
    X = dataset[0][0]
    print(get_ast_prime_and_size(X))
    prime_tuples = []
    size_tuples = []
    labels = []
    for ltree, rtree, label in tqdm.tqdm(dataset, desc="prime_generate"):
        l_p , l_size = get_ast_prime_and_size(ltree)
        r_p, r_size = get_ast_prime_and_size(rtree)
        prime_tuples.append((l_p, r_p))
        size_tuples.append((l_size, r_size))
        labels.append(label)
    torch.save({"prime_tuples": prime_tuples,
                              "size_tuples": size_tuples,
                              "labels":labels}, prim_dataset)


times_compute =[]
sims = []
print("Length of Dataset : %d" % len(prime_tuples))


for t in tqdm.tqdm(prime_tuples, desc='prime_cal'):
    s = datetime.datetime.now()
    sims.append(difference_ratio(t[0],t[1]))
    e = datetime.datetime.now()
    times_compute.append((e-s).total_seconds())

fpr, tpr, thresholds = roc_curve(labels, sims)
aucnumber = auc(fpr, tpr)

line = """==============
        time: %s
        auc: %f
        fpr = %s
        tpr = %s
        times_compute = %s
        ast_sizes = %s
        =================
        """ % (datetime.datetime.now(), aucnumber, str(fpr), str(tpr), str(times_compute), str(size_tuples))

with open("Diapgora_ROC.log", "a") as f:
    f.write(line+"\n")


