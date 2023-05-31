"""
Test controller in a file
"""

########################################################
# GA Inclusions.
########################################################
import warnings; warnings.filterwarnings('ignore')
import pickle, math, numpy as np
from functions_simulation import *
from functions_trees import *
from functions_plot import *

# Variables
seed = 101
n_iter_test = 100

# Create controller
apollo = apollo_controller.from_file(input_f + '/apollo_data_vref.npz')

# Load gains
#path = './output_GP_grow/3_test_N_10/GP_0'
path = './output_GP_grow/simple_10_100_0.4_10_0.01/run_1/GP_10_0.4_1_best'
condition = 'simple_10_100_pc_0.4_gamma_10_alpha_0.001_run_1'

f = open(path,"rb")
trees = pickle.load(f)
trees_idx = [ tree_idx(each) for each in trees]



# Sample gain functions
vs = apollo.ref_data.get_v() *1e-3
F1 = []; F2 = []; F3 = []

for v in vs:
    F1.append( evaluate_(trees[0], trees_idx[0], v)  )
    F2.append( evaluate_(trees[1], trees_idx[1], v)  )
    F3.append( evaluate_(trees[2], trees_idx[2], v)  )

F1 = np.array(F1)
F2 = np.array(F2)
F3 = np.array(F3)

results = mc_simulation(apollo.closed_loop_guidance, n_iter_test, seed)

simulation_image(results, apollo.ref_data, images_f + '/original.png')
print('Result footprint (original):', footprint(results), 'km')

apollo.ref_data.set_gains([F1, F2, F3])
results = mc_simulation(apollo.closed_loop_guidance, n_iter_test, seed)
simulation_image(results, apollo.ref_data, images_f + '/' + condition + '.png')
print('Result footprint (optimized):', footprint(results), 'km')
