"""
Plot of data 
Convergence graph of a single condition.

Jose Luis Rubén García-Zurdo
"""
########################################################
# Inclusions
########################################################
import numpy as np
from functions_plot import *

########################################################
# Variables
########################################################
data_labels = ['Method: grow']
in_folder = './output_GP_grow/CMA'
out_folder = in_folder #'./results/'
iterations = 10 #101
runs = 5

# Make big data-array
def make_big(data_folder, condition):
    global base
    res = []
    for cont1 in range(runs): #
        f_name = condition + '_' + str(cont1+1)
        
        #f_name = condition + '/run_' + str(cont1+1) + '/GP_10_0.4_' + str(cont1+1)
        print(data_folder + '/' + f_name + '.txt')
        f = open(data_folder + '/' + f_name + '.txt')
        data = f.readlines()[1:]
        f.close()
    
        # Convert to array
        for cont in range(len(data)):
            data[cont] = data[cont].split()[:]

        data = np.array(data).astype(np.float)###[:iterations,:]
        res.append(data)
    res = np.array(res)
    #base = res[0,0,4]
    return res

########################################################
# DATA. Read file.
#
########################################################
base = 0.0
data_folder = out_folder
condition = 'results'

if iterations >= 200:
    d_x = 25
else:
    d_x = 10
d_x = 5

########################################################
# Plot evolution
########################################################
out_f_name = 'simple_10_100_0.4_10_0.01_run_5'
title = 'CMA-ES' #'Genetic programming.'
var = 'Time [s]' #'Range dispersion [Km]' #'
condition = 'simple_10_100_0.4_10_0.01_run_5'

columm_id = 0
data = make_big(in_folder, condition)[:, :, columm_id][:,:iterations]
data_m = make_big(in_folder, condition)[:, :, columm_id].mean(axis=0)[:iterations]
data_s = make_big(in_folder, condition)[:, :, columm_id].std(axis=0)[:iterations]




plot(data, #[data_m],
     range(0, iterations),
     labels = [], #data_labels,
     err = [], #data_s],
     y_sup = 600, #100 60
     y_inf = 0,
     d_x = d_x,
     x_label = 'Generation',
     y_label = var,
     title = title,
     f_name = out_folder + '/' + out_f_name + '_ind.png')
