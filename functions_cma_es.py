# Function to optimize a circuit parameters
# using CMA-ES
#
# Requires: pip3 install cmaes

########################################################
# Inclusions.
########################################################
import time
from functions_trees import *
from config_GP import *
from cmaes import CMA

########################################################
# Apollo controller inclusions.
########################################################
from functions_simulation import *
seed = 0

# Create controller
apollo = apollo_controller.from_file(input_f + '/apollo_data_vref.npz', K=1)
apollo.K = 5

########################################################
# Fitness function
########################################################
def f_function(trees, trees_idx, save_img = False, cont = []):
    global apollo, n_mc_iter, rnd_mc_iter, seed
    global pars_alpha

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

    # Temp save
    if save_img: 
        def savefig(F, V, name, path):
            #print(path)
            plt.clf(); plt.scatter(vs, F); plt.ylabel(name); plt.title(name)
            plt.xlabel('v'); plt.grid(True)
            plt.savefig(path)
                
        savefig(F1, vs, 'F1', output_folder + '/' +str(cont)+'_F1.png')
        savefig(F2, vs, 'F2', output_folder + '/' +str(cont)+'_F2.png')
        savefig(F3, vs, 'F3', output_folder + '/' +str(cont)+'_F3.png')

    seed = 0
    if rnd_mc_iter:
        seed = int(np.random.rand()*1e3)
    apollo.ref_data.set_gains([F1, F2, F3])
    trajs = mc_simulation(apollo.closed_loop_guidance, n_iter = n_mc_iter, seed = seed)

    # Traj. dispersions
    disps = []
    for cont in range(num_block):    
        disps.append(footprint( trajs[cont * b_size: (cont+1) * b_size] ))
    avg_disp = np.mean([ each for each in disps  ])
  
    # Objectives
    nn = np.sum( [n_nodes(each) for each in trees ] )
    avg_disp = avg_disp + pars_alpha * nn
    objs = [avg_disp, None, None]  

    return [avg_disp, [objs]]

########################################################
# CMA-ES function
########################################################
def cma_es(trees, trees_idx):
    global init_sigma, bounds_orig, max_ite_cma

    prev_fit = f_function(trees, trees_idx)[0]
    
    # Secure original trees
    t_orig = trees[:]
    t = t_orig[:]

    # Extract lenghts and trees
    lens = []
    vars_ = []
    for each in trees:
        nums = extract_tree_num(each)
        vars_.extend(nums)
        lens.append(len(nums))
    pos = np.cumsum([0] + lens)
    
    # Initial values
    bounds = np.array( [bounds_orig]* len(vars_) )
    init_vals = np.array( vars_ ) 
    best_fit = float('Inf')
    best_t = None
    
    # CMAES
    print(len(init_vals), len(bounds))
    
    my_cma = CMA(init_vals, init_sigma)
    my_cma.set_bounds(bounds)
    
    for gen in range(max_ite_cma):
        start = time.perf_counter()
        #print(gen)
        evs = []
        best_val = float('Inf')
        best_vars = []
        for each in range(my_cma.population_size):        
            # EXTRACT EXPLICITELY

            s = my_cma.ask()

            for c_t in range(len(trees)):
                insert_tree_num(t[c_t], s[ pos[c_t]: pos[c_t+1] ] )

            # MO objective with Lambda
            ev = f_function( t, trees_idx )[0] #[1][0]
       
            if ev < best_val:
                best_val = ev
                best_vars = s
            evs.append( (s, ev) )
            #print(best_val)
            
        my_cma.tell(evs)
        for c_t in range(len(trees)):
            insert_tree_num(t[c_t], s[ pos[c_t]: pos[c_t+1] ] )

        fit, objs = f_function( t, trees_idx )
        if fit < best_fit:
            best_fit = fit
            best_t = deepcopy(t)
        
        objs = objs #np.sum(objs[0])
        end = time.perf_counter()
        my_t = end - start
        print(fit, '\t', my_t)
 
    return [best_t, best_fit, objs]


# Begin
seed = 200
np.random.seed(200)

# Variables
name = 'GP_10_0.4_5_best'
init_trees_f = './output_GP_grow/simple_10_100_0.4_10_0.01/run_5/' + name

# Read
with open(init_trees_f, 'rb') as f:
    init_trees = pickle.load(f)
init_trees_idx = [tree_idx(each) for each in init_trees]

t, fit, objs = cma_es(init_trees, init_trees_idx)


