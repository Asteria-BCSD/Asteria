from ysg_treelstm import Tree

def print_tree(root):
    '''
    first travel
    :param root:
    :return:
    '''
    print(root.op, end=" ")
    for c in root.children:
        print_tree(c)

def running_example():
    '''
    :return:
    '''
    from ysg_treelstm.plot.Types import NODETYPE
    tree1 = Tree()
    tree1.op = NODETYPE.cit_block
    # left tree
    asg_root = Tree()
    asg_root.op = NODETYPE.OPTYPE.cot_asg
    # var
    asg_l = Tree()
    asg_l.op = NODETYPE.OPTYPE.cot_var
    asg_root.add_child(asg_l)
    # num
    asg_r = Tree()
    asg_r.op = NODETYPE.OPTYPE.cot_num
    asg_root.add_child(asg_r)

    # middle
    if_root = Tree()
    if_root.op = NODETYPE.cit_if
    # sgt
    if_sgt = Tree()
    if_sgt.op = NODETYPE.OPTYPE.cot_sgt
    # vars
    sgt_var1 = Tree()
    sgt_var1.op = NODETYPE.OPTYPE.cot_var
    sgt_var2 = Tree()
    sgt_var2.op = NODETYPE.OPTYPE.cot_var
    if_sgt.add_child(sgt_var1)
    if_sgt.add_child(sgt_var2)

    if_root.add_child(if_sgt)
    # if-block
    if_block = Tree()
    if_block.op = NODETYPE.cit_block
    # if-block-asg
    block_asg = Tree()
    block_asg.op = NODETYPE.OPTYPE.cot_asg
    # if-block-asg-var
    asg_var = Tree()
    asg_var.op = NODETYPE.OPTYPE.cot_var
    block_asg.add_child(asg_var)
    # if-block-asg-sub
    asg_sub = Tree()
    asg_sub.op = NODETYPE.OPTYPE.cot_sub
    # if-block-asg-sub-vars
    sub_var = Tree()
    sub_var.op = NODETYPE.OPTYPE.cot_var
    asg_sub.add_child(sub_var)
    sub_var = Tree()
    sub_var.op = NODETYPE.OPTYPE.cot_var
    asg_sub.add_child(sub_var)
    block_asg.add_child(asg_sub)
    if_block.add_child(block_asg)
    if_root.add_child(if_block)
    # Return
    return_root = Tree()
    return_root.op = NODETYPE.cit_return
    # return-add
    return_add = Tree()
    return_add.op = NODETYPE.OPTYPE.cot_add
    # return-add-vars
    add_vars = Tree()
    add_vars.op = NODETYPE.OPTYPE.cot_var
    return_add.add_child(add_vars)
    add_vars = Tree()
    add_vars.op = NODETYPE.OPTYPE.cot_var
    return_add.add_child(add_vars)
    return_root.add_child(return_add)

    tree1.add_child(asg_root)
    tree1.add_child(if_root)
    tree1.add_child(return_root)

    #################Tree2
    tree2 = Tree()
    tree2.op = NODETYPE.cit_block
    # left tree
    asg_root = Tree()
    asg_root.op = NODETYPE.OPTYPE.cot_asg
    # var
    asg_l = Tree()
    asg_l.op = NODETYPE.OPTYPE.cot_var
    asg_root.add_child(asg_l)
    # num
    asg_r = Tree()
    asg_r.op = NODETYPE.OPTYPE.cot_num
    asg_root.add_child(asg_r)

    # middle
    if_root = Tree()
    if_root.op = NODETYPE.cit_if
    # sgt
    if_sgt = Tree()
    if_sgt.op = NODETYPE.OPTYPE.cot_sgt
    # vars
    sgt_var1 = Tree()
    sgt_var1.op = NODETYPE.OPTYPE.cot_var
    sgt_var2 = Tree()
    sgt_var2.op = NODETYPE.OPTYPE.cot_var
    if_sgt.add_child(sgt_var1)
    if_sgt.add_child(sgt_var2)

    if_root.add_child(if_sgt)
    # if-block
    if_block = Tree()
    if_block.op = NODETYPE.cit_block
    # if-block-asg
    block_asg = Tree()
    block_asg.op = NODETYPE.OPTYPE.cot_asg
    # if-block-asg-var
    asg_var = Tree()
    asg_var.op = NODETYPE.OPTYPE.cot_var
    block_asg.add_child(asg_var)
    # if-block-asg-sub
    asg_sub = Tree()
    asg_sub.op = NODETYPE.OPTYPE.cot_sub
    # if-block-asg-sub-vars
    sub_var = Tree()
    sub_var.op = NODETYPE.OPTYPE.cot_var
    asg_sub.add_child(sub_var)
    sub_var = Tree()
    sub_var.op = NODETYPE.OPTYPE.cot_var
    asg_sub.add_child(sub_var)
    block_asg.add_child(asg_sub)
    if_block.add_child(block_asg)
    if_root.add_child(if_block)
    # Return
    return_root = Tree()
    return_root.op = NODETYPE.cit_return
    # return-add
    return_add = Tree()
    return_add.op = NODETYPE.OPTYPE.cot_add
    # return-add-vars
    add_vars = Tree()
    add_vars.op = NODETYPE.OPTYPE.cot_var
    return_add.add_child(add_vars)
    add_vars = Tree()
    add_vars.op = NODETYPE.OPTYPE.cot_var
    return_add.add_child(add_vars)
    return_root.add_child(return_add)
    #tree2.add_child(asg_root)
    tree2.add_child(if_root)
    tree2.add_child(return_root)

    #check
    from ysg_treelstm.application.application import Application
    app = Application(load_path="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/crossarch.pt", )
    encode1 = app.encode_ast(tree1)
    encode2 = app.encode_ast(tree2)
    print(encode1)
    print(encode2)
    print(app.similarity_vec(encode1, encode2))


def encode_example():
    sub_tree = Tree()
    sub_tree.op=7
    left_tree=Tree()
    left_tree.op=35
    sub_tree.add_child(left_tree)

    right_tree=Tree()
    right_tree.op=21
    right_tree_var1=Tree()
    right_tree_var1.op=35
    right_tree.add_child(right_tree_var1)
    right_tree_var1=Tree()
    right_tree_var1.op=35
    right_tree.add_child(right_tree_var1)

    sub_tree.add_child(right_tree)
    from ysg_treelstm.application.application import Application
    app = Application(
        load_path="/root/treelstm.pytorch/ysg_treelstm/checkpoints/backup/train_after_hash_calculated.pt", )
    coder = app.encode_ast(sub_tree)
    print(coder)

if __name__ == '__main__':
    running_example()
    #encode_example()
