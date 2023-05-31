"""
Simulation variables
"""
############################################
# Imports
############################################
import numpy as np

############################################
# Paths and name
############################################
images_f = './images'
input_f = './reference'
output_f = './model'
name = 'simulation'
mc_seed = 0

############################################
# Linearization
############################################
n_reg = 8 #8 # Number of piece-wise regions
use_interp = False
fixed_edges = True # Keep edges fixed

############################################
# MC simulation config
############################################
verbose = False
# A 100
n_mc_iter = 10 #100 #100 #30 #30 < <
rnd_mc_iter = False
num_block = 1 #6 
b_size = n_mc_iter // num_block


