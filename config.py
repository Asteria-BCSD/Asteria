import argparse


def parse_args(iargs = []):
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # data arguments
    # parser.add_argument('--data', default='data/sick/',
    #                     help='path to dataset')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--model_selector', default='flatlstm', choices=["flatlstm", "splitedtracelstm", "treelstm"],
                        help='choose model')
    parser.add_argument('--expname', type=str, default='crossarch',
                        help='Name to identify experiment')
    # model arguments
    parser.add_argument('--input_dim', default=16, type=int,
                        help='Size of input word vector')
    parser.add_argument('--mem_dim', default=150, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of classes in dataset')
    # parser.add_argument('--freeze_embed', action='store_true',
    #                     help='Freeze word embeddings')
    # training arguments
    parser.add_argument('--epochs', default=60, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--vocab_size', default=88, type=int,
                        help='type number of ast node')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    if len(iargs) == 0:
        args = parser.parse_args()
    else:
        args = parser.parse_args(iargs)
    return args
