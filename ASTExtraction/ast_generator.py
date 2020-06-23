#encoding=utf-8
"""
Python2.7
IDAPython script running with Hexray plugin !!!
usage: idat -Sast_generator.py binary|binary.idb
Extracting the asts of all functions in the binary file and save the function information along with ast to the database file
"""

import idautils
import idaapi
from idc import *
from idaapi import *
from idautils import *
import logging,os,sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
import pickle
from DbOp import DBOP
l = logging.getLogger("ast_generator.py")
l.addHandler(logging.StreamHandler())
l.addHandler(logging.FileHandler("/tmp/ast_generator.log"))
l.setLevel(logging.WARNING)
#---- prepare environment
def wait_for_analysis_to_finish():
    '''
    :return:
    '''
    l.info('[+] waiting for analysis to finish...')
    idaapi.autoWait()
    idc.Wait()
    l.info('[+] analysis finished')

def load_plugin_decompiler():
    '''
    load the hexray plugins
    :return: success or not
    '''
    is_ida64 = GetIdbPath().endswith(".i64")
    if not is_ida64:
        idaapi.load_plugin("hexrays")
        idaapi.load_plugin("hexarm")
    else:
        idaapi.load_plugin("hexx64")
    if not idaapi.init_hexrays_plugin():
        l.error('[+] decompiler plugins load failed. IDAdb: %s' % GetInputFilePath())
        idc.Exit(0)

wait_for_analysis_to_finish()
load_plugin_decompiler()

#-----------------------------------

class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.op = None # value of node
        self.value = None # the value of constant or string if the parent node is a constant or string node.
        self.opname = "" # name of node, e.g. 'asg' (assignment)
    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
    def __str__(self):
        return self.opname
#--------------------------
spliter = "************"
class Visitor(idaapi.ctree_visitor_t):
    #preorder traversal tree
    def __init__(self, cfunc):
        idaapi.ctree_visitor_t.__init__(self, idaapi.CV_FAST|idaapi.CV_INSNS)
        self.cfunc = cfunc
        self._op_type_list =[]
        self._op_name_list =[]
        self._tree_struction_list = []
        self._id_list =[]
        self._statement_num = 0
        self._callee_set = set()
        self.root = None # root node of tree

    # Generate the sub tree
    def GenerateAST(self, ins):
        self._statement_num += 1
        AST = Tree()
        try:
            l.info("[insn] op  %s" % (ins.opname))

            AST.op=ins.op
            AST.opname = ins.opname

            if ins.op == idaapi.cit_block:
                self.dump_block(ins.ea, ins.cblock, AST)
            elif ins.op == idaapi.cit_expr:
                AST.add_child(self.dump_expr(ins.cexpr))

            elif ins.op == idaapi.cit_if:
                l.info("[if]"+spliter)
                cif = ins.details
                cexpr = cif.expr
                ithen = cif.ithen
                ielse = cif.ielse

                AST.add_child(self.dump_expr(cexpr))
                if ithen:
                    AST.add_child(self.GenerateAST(ithen))
                if ielse:
                    AST.add_child(self.GenerateAST(ielse))

            elif ins.op == idaapi.cit_while:
                cwhile = ins.details
                self.dump_while(cwhile,AST)

            elif ins.op == idaapi.cit_return:
                creturn = ins.details
                AST.add_child( self.dump_return(creturn) )

            elif ins.op == idaapi.cit_for:
                l.info( '[for]'+spliter)
                cfor = ins.details
                AST.add_child(self.dump_expr(cfor.init))
                AST.add_child(self.dump_expr(cfor.step))
                AST.add_child(self.dump_expr(cfor.expr))
                AST.add_child(self.GenerateAST(cfor.body))
            elif ins.op == idaapi.cit_switch:
                l.info('[switch]'+spliter)
                cswitch = ins.details
                cexpr = cswitch.expr
                ccases = cswitch.cases #Switch cases: values and instructions.
                cnumber = cswitch.mvnf #Maximal switch value and number format.
                AST.add_child(self.dump_expr(cexpr))
                self.dump_ccases(ccases, AST)
            elif ins.op == idaapi.cit_do:
                l.info('[do]'+spliter)
                cdo = ins.details
                cbody = cdo.body
                cwhile = cdo.expr
                AST.add_child(self.GenerateAST(cbody))
                AST.add_child(self.dump_expr(cwhile))
            elif ins.op == idaapi.cit_break or ins.op == idaapi.cit_continue:
                pass
            elif ins.op == idaapi.cit_goto:
                pass
            else:
                l.warning('[error] not handled op type %s' % ins.opname)

        except Exception, e:
            l.warning("[E] exception here ! " + e.message)

        return AST


    def visit_insn(self, ins):
        # pre-order visit ctree Generate new AST
        # ins maybe None , why ?

        if not ins:
            return 1
        # l.info("[AST] address and op %s %s" % (hex(ins.ea), ins.opname))
        self.root = self.GenerateAST(ins)
        l.info(self.root)
        return 1

    def dump_return(self, creturn):
        '''
        return an expression?
        '''
        return self.dump_expr(creturn.expr)

    def dump_while(self, cwhile, parent):
        '''
        visit while statement
        return:
            condition: expression tuple
            body : block
        '''
        expr = cwhile.expr
        parent.add_child(self.dump_expr(expr))
        whilebody = None
        body = cwhile.body
        if body:
            parent.add_child(self.GenerateAST(body))

    def dump_ccases(self, ccases, parent_node):
        '''
        :param ccases:
        :return: return a list of cases
        '''
        for ccase in ccases:
            AST = Tree()
            AST.opname = 'case'
            AST.op = ccase.op
            l.info('case opname %s, op %d' % (ccase.opname, ccase.op))
            value = 0 #default
            size = ccase.size() #List of case values. if empty, then 'default' case , 此处只取第一个值, 该对象属性值 ： 'acquire', 'append', 'disown', 'next', 'own
            if size > 0:
                value = ccase.value(0)
            AST.value = value
            block = self.dump_block(ccase.ea, ccase.cblock, AST)
            parent_node.add_child(AST)



    def dump_expr(self, cexpr):
        '''
        l.info the expression
        :return: AST with two nodes op and oprand : op Types.NODETYPE.OPTYPE, oprand : list[]
        '''
        # l.info "dumping expression %x" % (cexpr.ea)

        #操作数
        oprand =[] # a list of Tree()
        l.info("[expr] op %s" % cexpr.opname)

        if cexpr.op == idaapi.cot_call:
            # oprand = args
            # get the function call arguments
            self._get_callee(cexpr.ea)
            l.info('[call]'+spliter)
            args = cexpr.a
            for arg in args:
                oprand.append(self.dump_expr(arg))
        elif cexpr.op == idaapi.cot_idx:
            l.info('[idx]'+spliter)
            oprand.append(self.dump_expr(cexpr.x))
            oprand.append(self.dump_expr(cexpr.y))

        elif cexpr.op == idaapi.cot_memptr:
            l.info('[memptr]'+spliter)
            #TODO
            AST=Tree()
            AST.op = idaapi.cot_num #consider the mem size pointed by memptr
            AST.value = cexpr.ptrsize
            AST.opname = "value"
            oprand.append(AST)
            # oprand.append(cexpr.m) # cexpr.m : member offset
            # oprand.append(cexpr.ptrsize)
        elif cexpr.op == idaapi.cot_memref:

            l.info('[memref]'+spliter)
            # oprand.append(cexpr.m)

        elif cexpr.op == idaapi.cot_num:
            l.info ('[num]' + str(cexpr.n._value))
            AST = Tree()
            AST.op = idaapi.cot_num  # consider the mem size pointed by memptr
            AST.value = cexpr.n._value
            AST.opname = "value"
            oprand.append(AST)

        elif cexpr.op == idaapi.cot_var:
            # var : cexpr.v
            l.info ('[var]' + str(cexpr.v.idx))
            # oprand.append(cexpr.v.idx) # which var (index for var)
            # TODO handle array type
        elif cexpr.op == idaapi.cot_str:
            # string constant
            l.info( '[str]' + cexpr.string)
            AST =Tree()
            AST.opname = "string"
            AST.op = cexpr.op
            AST.value = cexpr.string
            oprand.append(AST)

        elif cexpr.op == idaapi.cot_obj:
            l.info ('[cot_obj]' + hex(cexpr.obj_ea))
            # oprand.append(cexpr.obj_ea)
            # Many strings are defined as 'obj'
            # I wonder if 'obj' still points to other types of data?
            # notice that the address of 'obj' is not in .text segment
            if get_segm_name(getseg(cexpr.obj_ea)) not in ['.text']:
                AST = Tree()
                AST.opname = "string"
                AST.op = cexpr.op
                AST.value = GetString(cexpr.obj_ea)
                oprand.append(AST)

        elif cexpr.op <= idaapi.cot_fdiv and cexpr.op >= idaapi.cot_comma:
            #All binocular operators
            oprand.append(self.dump_expr(cexpr.x))
            oprand.append(self.dump_expr(cexpr.y))

        elif cexpr.op >= idaapi.cot_fneg and cexpr.op <= idaapi.cot_call:
            # All unary operators
            l.info( '[single]' + spliter)
            oprand.append(self.dump_expr(cexpr.x))
        else:
            l.warning ('[error] %s not handled ' % cexpr.opname)
        AST = Tree()
        AST.opname=cexpr.opname
        AST.op=cexpr.op
        for tree in oprand:
            AST.add_child(tree)
        return AST

    def dump_block(self, ea, b, parent):
        '''
        :param ea: block address
        :param b:  block_structure
        :param parent: parent node
        :return:
        '''
        # iterate over all block instructions
        for ins in b:
            if ins:
                parent.add_child(self.GenerateAST(ins))

    def get_pseudocode(self):
        sv = self.cfunc.get_pseudocode()
        code_lines = []
        for sline in sv:
            code_lines.append(tag_remove(sline.line))
        return "\n".join(code_lines)

    def get_caller(self):
        call_addrs = list(idautils.CodeRefsTo(self.cfunc.entry_ea, 0))
        return len(set(call_addrs))

    def get_callee(self):
        return len(self._callee_set)

    def _get_callee(self, ea):
        '''
        :param ea:  where the call instruction points to
        :return: None
        '''
        l.info('analyse addr %s callee' % hex(ea))
        addrs = list(idautils.CodeRefsFrom(ea,0))
        for addr in addrs:
            if addr == GetFunctionAttr(addr, 0):
                self._callee_set.add(addr)

class AstGenerator():

    def __init__(self, optimization_level, compiler = 'gcc'):
        '''
        :param optimization_level: the level of optimization when compile
        :param compiler: the compiler name like gcc
        '''
        if optimization_level not in ["O0","O1","O2","O3","Os","default"]:
            l.warning("No specific optimization level !!!")
        self.optimization_level =optimization_level
        self.bin_file_path = GetInputFilePath() # path to binary
        self.file_name = GetInputFile() # name of binary
        # get process info
        self.bits, self.arch, self.endian = self._get_process_info()
        self.function_info_list = list()
        #Save the information of all functions, of which ast class is saved using pick.dump.
        # Each function is saved with a tuple (func_name. func_addr, ast_pick_dump, pseudocode, callee, caller)

    def _get_process_info(self):
        '''
        :return: 32 or 64 bit, arch, endian
        '''
        info = idaapi.get_inf_structure()
        bits = 32
        if info.is_64bit():
            bits=64
        try:
            is_be = info.is_be()
        except:
            is_be = info.mf
        endian = "big" if is_be else "little"
        return bits, info.procName, endian

    def run(self, fn, specical_name = ""):
        '''
        :param fn: a function to handle the functions in binary
        :param specical_name: specific function name while other functions are ignored
        :return:
        '''
        if specical_name!="":
            l.info("specific functino name %s" % specical_name)
        for i in range(0, get_func_qty()):
            func = getn_func(i)
            segname = get_segm_name(getseg(func.startEA))
            if segname[1:3] not in ["OA", "OM", "te", "_t"]:
                continue
            func_name = GetFunctionName(func.startEA)
            if len(specical_name) > 0 and specical_name != func_name:
                continue
            try:
                ast_tree, pseudocode, callee_num, caller_num = fn(func)
                self.function_info_list.append((func_name, hex(func.startEA) ,pickle.dumps(ast_tree), pseudocode, callee_num, caller_num))
            except Exception, e:
                l.warning(e.message)

    def save_to(self, db):
        '''
        :param db: DBOP instance
        :return:
        '''
        for info in self.function_info_list:
            try:
                db.insert_function(info[0], info[1], self.file_name, self.bin_file_path,
                                   self.arch+str(self.bits), self.endian, self.optimization_level, info[2], info[3], info[4],info[5])
            except Exception,e:
                l.error("insert operation exception when insert %s" % self.bin_file_path+" "+info[0])
                l.error(e.message)

    @staticmethod
    def get_info_of_func(func):
        '''
        :param func:
        :return:
        '''
        try:
            cfunc = idaapi.decompile(func.startEA)
            vis = Visitor(cfunc)
            vis.apply_to(cfunc.body, None)
            return vis.root, vis.get_pseudocode(), vis.get_callee(), vis.get_caller()
        except:
            l.warning("function %s decompile failed" % (GetFunctionName(func.startEA)))
            raise


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o","--optimization", default="default", help="optimization level when compilation")
    ap.add_argument("-f","--function", default="", help="extract the specific function info")
    ap.add_argument("-g","--compiler", default="gcc", help="compiler name adopted during compilation")
    ap.add_argument("-d","--database", default="default.sqlite", type=str, help="path to database")
    args = ap.parse_args(idc.ARGV[1:])
    astg = AstGenerator(args.optimization, compiler=args.compiler)
    astg.run(astg.get_info_of_func, specical_name=args.function)
    # astg.run(astg.get_info_of_func, specical_name="SSL_get_ciphers") # this line code for test
    dbop = DBOP(args.database)
    astg.save_to(dbop)
    del dbop # free to call dbop.__del__() , flush database
    idc.Exit(0)
