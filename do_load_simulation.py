"""
NPCGC:
APGD simulation using a reference trajectory.
"""

############################################
# Imports
############################################
import numpy as np
import pickle
from functions_simulation import *
from config_GA import *

def save(x, y, name):
    global images_f
    plt.clf()
    plt.scatter(x, y)
    plt.xlabel('v')
    plt.ylabel(name)
    plt.grid(True)
    plt.title(name)
    plt.savefig(images_f + '/'+ name + '.png')
    
############################################
# Random number generator
############################################
seed = 0
#n_mc_iter = 100 #In config_model
    
############################################
# Create reference controller
############################################
apollo = apollo_controller.from_file(input_f + '/apollo_data_vref.npz')
F1, F2, F3 = apollo.ref_data.get_gains()
v = apollo.ref_data.X_and_lam[:, 2]
save(v, F1, '_F1') # (interp)')
save(v, F2, '_F2')
save(v, F3, '_F3')



############################################
# Simulations
############################################
results = mc_simulation(apollo.closed_loop_guidance, n_mc_iter, seed)
simulation_image(results, apollo.ref_data, images_f + '/original.png')
print('Result footprint (original):', footprint(results), 'km')


############################################
# Load
############################################
name = GA_output_f + '/30_100_best_93'
f2 = open(name, 'rb')
data = pickle.load(f2)

F1 = linearize(F1, n_reg, ranges[0][0], ranges[0][1], new_y = data[0] )
F2 = linearize(F2, n_reg, ranges[1][0], ranges[1][1], new_y = data[1] )
F3 = linearize(F3, n_reg, ranges[2][0], ranges[2][1], new_y = data[2] )
apollo.ref_data.set_gains([F1, F2, F3])

results = mc_simulation(apollo.closed_loop_guidance, n_mc_iter, seed)
simulation_image(results, apollo.ref_data, images_f + '/piecewise.png')
print('Result footprint (loaded piecewsite):', footprint(results), 'km')






