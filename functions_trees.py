# Tree functions.
# Each time a tree is created or modified,
# its node index list must be updated.
# Ex:
#   a = tree(8, 'grow');
#   a_idx = tree_idx(a);

import numpy as np
from copy import *
from itertools import *
from config_GP import *

########################################################
# Tree information
#
########################################################
# Return the type (T/NT) of a node
def node_type(node):
    if type(node) == list:
        node = node[0]
        
    if node in NT:
        return 'NT'
    elif node in T or type(node) == float:
        return 'T'

# Return number of levels of a tree
def depth(tree):
    if node_type(tree) == 'T':
        res = 0
    else:
        if tree[0] in NT_1:
            res = 1 + depth( tree[1] )
            
        if tree[0] in NT_2:
            res = 1 + max( depth( tree[1][0]), depth(tree[1][1]) )
 
    return res

# Create string from a tree
def expr(tree, dec = 2):
    if node_type(tree) == 'T':
        if type(tree) == np.str_ or  type(tree) == str:
            res = str(tree)
        else:
            res = str( round(tree, dec) )
    else:
        if tree[0] in NT_1:
            res = '<' + str( tree[0]) +'|' + expr( tree[1], dec) +'>'
        if tree[0] in NT_2:
            res = '<' + expr(tree[1][0], dec) +'|'+  str(tree[0]) +'|' + expr(tree[1][1], dec)+'>'
    return res

# Number of nodes in a tree:
def n_nodes(tree):
    if node_type(tree) == 'T':
        return 1
    
    if tree[0] in NT_1:
        return n_nodes(tree[1]) + 1
    
    if tree[0] in NT_2:
        return ( n_nodes(tree[1][0]) + n_nodes(tree[1][1]) + 1 )

# Number of NT nodes in a tree:
def NT_nodes(tree):
    if node_type(tree) == 'T' or len(tree)== 1:
        return 0
    
    if tree[0] in NT_1:
        return NT_nodes(tree[1]) + 1
    
    if tree[0] in NT_2:
        return ( NT_nodes(tree[1][0]) + NT_nodes(tree[1][1]) + 1 )
    
########################################################
# Tree indices list.
#
########################################################
# Obtain a sub-tree indexed by a list of indices
def tree_by_idx(tree, l):
    res = tree
    for each in l:
        res = res[each]
    return res

# List of a tree nodes
def tree_idx(tree):
    global my_idx
    my_idx = []
    get_indices(tree)
    return my_idx

# Not to be called. Internal function
def get_indices(tree, res = []):
    global my_idx
    my_idx.append(res)
    #print(tree, res)
    if node_type(tree) == 'T': #type(tree) == float or len(tree)== 1:
        return  res + [0]
    else:
        if tree[0] in NT_1:
            return get_indices(tree[1], res + [1])
        
        elif tree[0] in NT_2:
            return [ get_indices(tree[1][0], res + [1,0]) +\
                     get_indices(tree[1][1], res + [1,1]) ]

########################################################
# Modification of numerical values
#
########################################################
# List of numeric variables
def extract_tree_num(tree):
    var_list = []
    idx = tree_idx(tree)
    for each in idx:
        data = tree_by_idx(tree, each)
        if type(data)== float:
            var_list.append( data)
    return var_list

# Substitute numeric values in a tree by var_list
def insert_tree_num(tree, var_list):
    idx = tree_idx(tree)
    cont = 0
    for each_idx in idx:
        data = tree_by_idx(tree, each_idx)
        if type(data)== float:
           substitute(tree, each_idx, var_list[cont])
           cont += 1
    return tree

########################################################
# Tree manipulation
#
########################################################
# Random tree creation
def tree(depth, method = 'full', first = True):
    if depth == 0:
        res = np.random.choice(T)
        if res == 'coeff': res = coeff()
        return  res   
    else:
        if method == 'full':
            op = np.random.choice(NT)
            if op in NT_2:
                res = [op,[ tree(depth-1, method, False), tree(depth-1, method, False)]]
            elif op in NT_1:
                res = [op, tree(depth-1, method, False) ]

        if method == 'grow':
            if first:
                op = np.random.choice(NT) 
            else:
                op = np.random.choice(T + NT)
            
            if node_type(op) == 'T':
                if op == 'coeff':
                    op = coeff()
                res = op
                
            else:
                op = np.random.choice(NT)
                if op in NT_2:
                    res = [op,[ tree(depth-1, method, False),  tree(depth-1, method, False)]]
                elif op in NT_1:
                    res = [op, tree(depth-1, method, False) ]
        return res

########################################################
# Tree evaluation
#
########################################################
# Evaluate tree with a value
def evaluate2(tree, x):
    #print(tree)
    if type(tree) == float or len(tree)== 1:
        if type(tree) == float:
            return float(tree)
        if tree == 'x':
            return float(x)
    
    if tree[0] in NT_2:
        if tree[0] == '+':
            res = evaluate2(tree[1][0],x) + evaluate2(tree[1][1],x)
        if tree[0] == '-':
            res = evaluate2(tree[1][0],x) - evaluate2(tree[1][1],x)
        if tree[0] == '*':
            res = evaluate2(tree[1][0],x) * evaluate2(tree[1][1],x)
        if tree[0] == '%': # Protected division
            num = evaluate2(tree[1][0],x)
            den = evaluate2(tree[1][1],x)
            if den == 0: den = 1e5
            res = num / den
        return res

    if tree[0] in NT_1:
        if tree[0] == 'log': # Protected log
            arg = evaluate2(tree[1],x)
            arg = np.clip(arg, 1e-100, 1e100)
            res = np.log(arg)
            
        if tree[0] == 'exp': #Protected exp
            arg = evaluate2(tree[1],x)
            arg = np.clip(arg, -100, 100)
            res = np.exp(arg)
            
        if tree[0] == 'sin':
            arg = evaluate2(tree[1],x)
            res = np.sin(arg)
        return res

# Evaluate tree with array
def evaluate_arr(tree, x_arr):
    #if type(x_arr) == list:
    #    x_arr = np.array(x_arr)
        
    #print(tree)
    if type(tree) == float or type(tree) == np.float_ or len(tree)== 1:
        if type(tree) == float or type(tree) == np.float_:
            return np.float_(tree)
        if tree == 'x':
            return np.float_(x_arr)
    
    if tree[0] in NT_2:
        if tree[0] == '+':
            res = evaluate(tree[1][0],x_arr) + evaluate(tree[1][1],x_arr)
            
        if tree[0] == '-':
            res = evaluate(tree[1][0],x_arr) - evaluate(tree[1][1],x_arr)
            
        if tree[0] == '*':
            #res = evaluate(tree[1][0],x_arr) * evaluate(tree[1][1],x_arr)
            arg1 = evaluate(tree[1][0],x_arr)
            arg2 = evaluate(tree[1][1],x_arr)
            res = np.multiply(arg1, arg2)
            
        if tree[0] == '%': # Protected division
            #print('division')
            num = evaluate(tree[1][0],x_arr)
            den = evaluate(tree[1][1],x_arr)
            #print(den, type(den))
            if type(den) != np.ndarray and den == 0:
                den = 1e-5
            if type(den) == np.ndarray:
                den[ den == 0] += 1e-5
            #print(den)
            res = num / den

    if tree[0] in NT_1:
        if tree[0] == 'log': # Protected log
            arg = evaluate(tree[1],x_arr)
            arg = np.clip(arg, 1e-100, 1e100)
            res = np.log(arg)
            
        if tree[0] == 'exp': #Protected exp
            arg = evaluate(tree[1],x_arr)
            arg = np.clip(arg, -100, 100)
            res = np.exp(arg)
            
        if tree[0] == 'sin':
            arg = evaluate(tree[1],x_arr)
            res = np.sin(arg)

    #print(tree, res, tree[0] == '%', tree[0] in NT_2)
    return res

# Non-recursive tree evaluation of array of values
def evaluate_(t, t_idx, x):
    # Traverse subtrees in reverse order
    a = np.zeros(100)
    cont = 0
    for each_i in t_idx[::-1]:
        each = tree_by_idx(t, each_i)
        #print(' ==> ',each)
        # Keep a inverse list (stack) of results
        if type(each) != list:
            if each == 'x':
                a[cont]= x
            else:
                a[cont]= each
            cont += 1
        else:
            # Always clip!
            #a[cont-1] = np.clip(a[cont-1],  -1e+35, 1e+35)
                                
            if each[0] == '+':
                res = a[cont-1] + a[cont-2]
                a[cont-2] = res
                cont = (cont -2)
                
            if each[0] == '-':
                res = a[cont-1] - a[cont-2]
                a[cont-2] = res
                cont = (cont -2)
                
            if each[0] == '*':
                res = a[cont-1] * a[cont-2]
                a[cont-2] = res
                cont = (cont -2)
                
            if each[0] == '%':
                if a[cont-2] == 0:
                    a[cont-2] += 1 #1e-3 #5

                res = a[cont-1] / a[cont-2]
                a[cont-2] = res
                cont = (cont -2)

            if each[0] == '^':
                a[cont-1] = np.clip(a[cont-1],  -1e2+10, 1e2-10)
                a[cont-2] = np.clip(a[cont-1],  -1e2+10, 1e2-10)
                
                res = np.sign(a[cont-1]) * np.power(abs(a[cont-1]), a[cont-2])
                a[cont-2] = res
                cont = (cont -2)

            if each[0] == 'log':
                a[cont-1] = np.clip(a[cont-1], 1e-10, 1e10)
                res = np.log(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            if each[0] == 'exp':
                a[cont-1] = np.clip(a[cont-1], -1e+10, 1e+2)
                res = np.exp(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            if each[0] == 'sin':
                res = np.sin(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            if each[0] == 'cos':
                res = np.cos(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            if each[0] == 'tanh':
                a[cont-1] = np.clip(a[cont-1], -100, 100)
                res = np.tanh(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            if each[0] == 'sqr':
                if a[cont-1] <= 0:
                    a[cont-1] = 1e-5
                res = np.sqrt(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)
                
            if each[0] == '--':
                res = -(a[cont-1])
                a[cont-1] = res
                cont = (cont -1)

            cont += 1

    # Check errors
    if a[cont-1] != a[cont-1]:
        print("Math error")
        import sys
        sys.exit()
    else:
        res = a[cont-1]
    return res

# Substitute a node in position idx by a tree
# Perform substitution in argument and returned value
def substitute(tree, node_idx, new_node):
    # scape single 'x'
    if type(new_node) == str or type(new_node) == np.str_:
        new_node = "'" + str(new_node) + "'"
    res = ''
    for each in node_idx:
        res += '[' +str(each)  +']'
    res = 'tree[:]' + res + '= ' + str(new_node)
    #print ('exec:', res)
    exec( res )
    return tree

########################################################
# Tests
########################################################
# TESTS
"""
np.random.seed(17)
c = tree(3); c_idx = tree_idx(c)
d = tree(3); d_idx = tree_idx(c)
print(c)
print(d)
c = substitute(c, c_idx[3], tree_by_idx(d, d_idx[4]))
print(c)

np.random.seed(21)
t = tree(12, "full"); t_idx = tree_idx(t)
print( evaluate_(t, t_idx, 1.0) )
"""


