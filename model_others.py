#encoding=utf-8
'''

'''
from copy import deepcopy
from torch import nn
import torch
import Tree

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        ).cuda()

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SplitedTreeLSTM(nn.Module):
    def __init__(self, vocb_size, in_size, hidden_size, out_size):
        '''
        :param vocb_size: 字典大小
        :param in_size: lstm输入维度
        :param hidden_size: lstm隐藏维度
        :param out_size: 输出层输出维度
        '''
        super(SplitedTreeLSTM, self).__init__()
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocb_size, in_size).cuda()
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=2).cuda()
        self.out_layer = nn.Linear(hidden_size, out_size).cuda()
        self.attetion = SelfAttention(hidden_size)

    def forward(self, tree):
        all_splited_trace = []
        def get_tree_splited(root, recorder):#post order
            '''
            :param recorder: 记录到root节点为止的路径trace节点
            :param root: 子树根节点
            :return: 对树进行自上向下的遍历，记录所有从根到叶子节点的路径，并且每个路径上的节点对应一个list，返回之
            '''
            recorder.append(root.op)
            if root.num_children == 0: # 如果是叶子节点
                all_splited_trace.append(recorder)
            for child in root.children:
                new_recorder = deepcopy(recorder)
                get_tree_splited(child, new_recorder)
        first_order = []
        get_tree_splited(tree, first_order)
        assert len(all_splited_trace)!=0
        all_trace_emb = torch.Tensor(len(all_splited_trace), self.hidden_size).cuda() #所有分割的路径编码
        for idx, splited_trace in enumerate(all_splited_trace):
            trace = torch.LongTensor(splited_trace).unsqueeze(0).cuda()
            trace_emb = self.trace_embeding(trace)
            all_trace_emb[idx] = trace_emb
        final_out = torch.sum(all_trace_emb, 0)
        return self.out_layer(final_out)

    def trace_embeding(self, trace): #对一个trace序列进行编码 trace:Tensor
        in_emb = self.emb(trace)
        hidden_out, (hn, hc) = self.lstm(in_emb)
        attention_out, attention_weight = self.attetion(hidden_out)
        # #TODO 怎么把lstm输出的步长统一化
        lstm_out = attention_out.squeeze(0)
        # out = self.out_layer(lstm_out)
        return lstm_out

class FLATLSTM(nn.Module):#对树进行先序遍历，生成一个序列，然后使用lstm进行编码
    def __init__(self, vocb_size, in_size, hidden_size, out_size):
        '''
        :param vocb_size: 字典大小
        :param in_size: lstm输入维度
        :param hidden_size: lstm隐藏维度
        :param out_size: 输出层输出维度
        '''
        super(FLATLSTM, self).__init__()
        self.out_size = out_size
        self.emb = nn.Embedding(vocb_size, in_size).cuda()
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=2).cuda()
        self.out_layer = nn.Linear(hidden_size, out_size).cuda()
        self.attetion = SelfAttention(hidden_size)

    def forward(self, tree):

        def visit_tree(root, node_list):#post order
            for child in root.children:
                visit_tree(child, node_list)
            node_list.append(root.op)
        first_order = []
        visit_tree(tree, first_order)
        assert len(first_order)!=0
        input = torch.LongTensor([first_order]).cuda()
        in_emb = self.emb(input)
        hidden_out, (hn, hc) = self.lstm(in_emb)
        # l = len(first_order)
        # ll = 0 if l == 1 else l-2
        # index = torch.LongTensor([ll,l-1]).cuda()
        # hidden_out = torch.index_select(hidden_out, 1, index)
        # hidden_out = hidden_out.view(1,-1)
        # hidden_out = hidden_out.squeeze(0) #now shape is (2, hidden)
        attention_out, attention_weight = self.attetion(hidden_out)
        # hidden_out =torch.sum(attention_out, 1) #TODO 怎么把lstm输出的步长统一化
        lstm_out = attention_out.squeeze(0)
        out = self.out_layer(lstm_out)
        return out

class Sieamens(nn.Module):
    # compute similarity between two ast encodes
    def __init__(self, vocab_size, input_dim, mem_dim, hidden_dim, num_classes, modelstr = "flatlstm"):
        '''
        vocab_size,
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes
        '''
        super(Sieamens, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.out_dim = 64 # self.model 输出向量的大小
        if modelstr=="flatlstm":
            self.embmodel = FLATLSTM(vocab_size, input_dim, hidden_dim, self.out_dim)
        elif modelstr=="splitedtracelstm":
            self.embmodel = SplitedTreeLSTM(vocab_size, input_dim, hidden_dim, self.out_dim)
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.out_dim, self.out_dim).cuda()
        self.wp = nn.Linear(self.out_dim, self.num_classes).cuda()
        self._out = nn.Linear(self.mem_dim, self.hidden_dim).cuda()

    def forward(self, ltree, rtree):
        lout = self.embmodel(ltree)
        rout = self.embmodel(rtree)
        return self.similarity(lout, rout)

    def similarity(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 0)
        out = torch.sigmoid(self.wh(vec_dist))
        out = torch.softmax(self.wp(out), dim=0)
        return out