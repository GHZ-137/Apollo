# Optimization
save = True
###output_folder = './output_GP_grow/simple_pc_0.4_gamma_10_alpha_0.001/run_2'
## ABAJO
runs = [0] # 3,4] #0 1 2 

optimization = 'Minimize'
use_multi = True
n_procs = 6 #6 #10

# Tree parameters
init_d = 2 # Initial depth << 
min_d = 2  # Mutate min depth

method = "grow" #"grow"

max_d = 50 #20 # Max depth during evolution
max_n_nodes = 400 #200 # Max n nodes during evolution
n_func = 3 # n of functions to optimize


# CMA Local search
# Generations between local search optimization:
# 1 (always), 0 (No), -1 (only when improvement)
freq_cma = 0 #-1 #5 +1
frac_ind_cma = 0.1
init_sigma = 1.0
max_ite_cma = 10

# NSGA
n_objs = 3
obj_1 = 0; obj_2 = 1
do_NSGA = False
do_NSGA_CMA = False

# GP evolution
Mu = 100 #60
max_gen = 100

# << Mutation subtree interchange
p_c = 0.4 # << 0.3   0.5 0.25 .75
p_m = 0.5
p_r = 0.1 #1 # << 0.1

sigma = 1.0
p_subtree = 1.

# Annealing
c = 100
k = 10

elitism = 'Yes'
selection = 'tournament'
p_Gamma = 0.10
Gamma = 10 #int(round(p_Gamma * Mu)) #Tournament size << 2 0.1 0.05 0.025
pars_alpha = 0.01  # Parsimony   < < <  0

#######################################################################
# Math operators
#######################################################################
# Terminal nodes
import numpy as np
T = ['x', 'coeff']

# Non terminal nodes by arguments number
NT_2 = ['+', '-', '*', '%' ] #, '^'] # Power is excluded
#NT_1 = ['log', 'exp', 'sin', 'cos', '--',] # 'sqr','tanh']
NT_1 = []  # < < < []
NT = NT_2 + NT_1

# Numerical range of function arguments (coeff)
n_1 = -1.; n_2 = 1.
def coeff(): return np.random.uniform(n_1, n_2)
bounds_orig = [n_1, n_2]


from config_model import * 
output_folder = './output_GP_grow/simple_'
output_folder += str(n_mc_iter) + '_' + str(Mu) + '_' + str(p_c) + '_' + str(Gamma) + '_' + str(pars_alpha)
output_folder += '/run_1'


