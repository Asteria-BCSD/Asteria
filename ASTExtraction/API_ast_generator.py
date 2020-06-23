#encoding=utf-8
'''
python3
A script for facilitating the usage of ast_generator.py
'''
import sys, os, logging, argparse
root = os.path.dirname(os.path.abspath(__file__))
import subprocess
from tqdm import tqdm
from datetime import datetime
def parse_arg():
    argparser = argparse.ArgumentParser(description="AST generator script based on IDA and \033[1;31m Python3 \033[0m!")
    argparser.add_argument("--ida_path", default="/opt/idapro7.3/idat",
                           help="path to idal(before 7.2) or idat(after 7.2) with decompiler plugins, idal64 for 64bit binary(also for 32bit)")
    argparser.add_argument("--binary", help="path to binary to be analysed")
    argparser.add_argument("--directory", help="A path where all binaries in the dir will be analysed. --binary will not work if this option specified")
    argparser.add_argument("--database", help="path to sqlite database to save the extracted asts", default="default.sqlite")
    argparser.add_argument('--logfile', help="log file when ida runs", default="api_ast_generator.log")
    argparser.add_argument('--function', help="specific function name, default is to get all function ast", default="")
    argparser.add_argument("--compilation", choices=["default","O0","O1","O2","O3","Os"], default="default",
                           help="specify the compilation level of binary")
    argparser.add_argument("--timeout", default=360, type=int, help="max seconds the compilation of a binary cost")

    return argparser.parse_args()
class AstGenerator():
    '''
    This class implements these functions:  
    1. Extract all functions ast from the specified binary file and save to the database  
    2. Extract A specific function ast from the specified binary file and save to the database  
    3. Batch version of function1: accessing all ELF files from the specified folder, extract all function asts and save to Database
     4. Read A specific ELF file from the specified folder, extract all the functions ast, save to the database
    '''
    def __init__(self, args):
        self.Script = os.path.join(root,'ast_generator.py')  #the path to the IDAPython script for extracting AST
        self.args = args
        self.logger = logging.getLogger("AstGenerator")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.arch_support = ['powerpc','ppc','x86','x86-64','arm', "80386"]

    def extract_all_function_ast(self, binary_path): # the described function 1
        self.extract_function_ast(binary_path, func_name="")

    def extract_function_ast(self, binary_path, func_name): # the described function 2
        IDA_ARGS = ["-o %s" % self.args.compilation,  "-g gcc", "-d %s" % self.args.database]
        if self.args.function:
            IDA_ARGS.append("-f %s" % self.args.function)
        IDA_ARGS = " ".join(IDA_ARGS)
        cmd_list = ["TVHEADLESS=1", self.args.ida_path, "-c" ,"-A", '-S"%s %s"' % (self.Script, IDA_ARGS), binary_path]
        cmd = " ".join(cmd_list)
        p = subprocess.Popen(cmd, shell =True)
        try:
            p.wait(timeout=self.args.timeout) # after waiting , kill the subprocess
        except subprocess.TimeoutExpired as e:
            self.logger.error("[Error] time out for %s" % binary_path)
        if p.returncode != 0:
            self.logger.error("[ERROR] cmd: %s" %(cmd))
            if p:
                p.kill()
        else:
            self.logger.info("[OK] cmd %s " % cmd)

    def get_arch(self, binary):
        '''
        :param binary:
        :return: the architecture information output by 'file' command
        '''
        cmd = "file %s| grep ELF |cut -d, -f2" % (binary)
        retinfo = subprocess.getoutput(cmd).strip().lower()
        return retinfo

    def supported_arch_binary(self, path):
        ''' To see whether the binary is supported
        :param path:
        :return:
        '''
        archinfo = self.get_arch(path)
        for arch in self.arch_support:
            if arch in archinfo:
                return True
        return False

    def extract_ast_from_dir(self, dir):# the described function 3
        for root, dirs, files in os.walk(dir):
            for f in tqdm(files, desc=os.path.basename(root)):
                binary_path = os.path.join(root, f)
                if self.supported_arch_binary(binary_path):
                    self.extract_function_ast(binary_path, func_name=self.args.function)


if __name__ == '__main__':
    args =parse_arg()
    ag = AstGenerator(args)
    if args.directory:
        ag.extract_ast_from_dir(args.directory)
    elif args.binary and args.function:
        ag.extract_function_ast(args.binary, func_name=args.function)
