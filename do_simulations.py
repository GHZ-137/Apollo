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
#from class_apollo_guidance import *
#from config_conditions import *

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
# Interpolate piecewese affine regions
############################################

new_v = np.linspace(600, 5500, v.shape[0])
F1 = linearize(F1, n_reg, np.min(F1), np.max(F1)) #, np.max(F1)-np.min(F1) )
F2 = linearize(F2, n_reg, np.min(F2), np.max(F2)) #, np.max(F2)-np.min(F2) )
F3 = linearize(F3, n_reg, np.min(F3), np.max(F3)) #, np.max(F3)-np.min(F3) )
apollo.ref_data.set_gains([F1, F2, F3])


results = mc_simulation(apollo.closed_loop_guidance, n_mc_iter, seed)
simulation_image(results, apollo.ref_data, images_f + '/piecewise.png')
print('Result footprint (piecewsite):', footprint(results), 'km')






