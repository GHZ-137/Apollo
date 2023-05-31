"""
Genetic programming for symbolic regression
Test:
for each in pop.generations[0]: print (each)
pop.best(2)[1:]
"""

########################################################
# Inclusions.
########################################################
import time, warnings; warnings.filterwarnings('ignore')
import pickle, math, numpy as np

from functions_trees import *
from functions_plot import *
from class_GP import *

from config_GP import *
import concurrent.futures

if __name__ == '__main__':
########################################################
# Variables.
########################################################
    labels = ['Iter.', 'Best', 'Avg', 'Worst', 'Min_d', 'Max_d', 'Avg_d', 'Min_n', 'Max_n', 'Avg_n', 'Calls', 't']
    labels_f =  "  {: <10} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10}  {: <10} {: <10} {: <10} {: <10} {: <10}"

    def save_best():
        # Save best tree
        condition = 'best_' + str(Mu) + '_' + str(d_max)
        name = output_folder + '/' + condition + '_' + str(run+1) + '_best'
        print(name)
        with open(name, 'wb') as f: pickle.dump(pop.best()[0], f)
        #with open(name, 'rb') as f: r = pickle.load(f)
        return
    
########################################################
# Apollo controller inclusions.
########################################################
    from functions_simulation import *
    seed = 0

# Create controller
    apollo = apollo_controller.from_file(input_f + '/apollo_data_vref.npz', K=1)
    apollo.K = 5

########################################################
# Multiprocess
########################################################
    def shared_evaluate(inds):
        global f_function
        fits = []
        objs = []
        for ind in inds:
            val = f_function(ind.vars, ind.trees_idx)
            fits.append( val[0] )
            objs.append( val[1] )
        return objs #[fits , objs]

    def shared_mutate(inds):
        res = []
        for ind in inds:
            ind.mutate()
            res.append( ind )
        return res

    def shared_recombine(parent_pairs):
        res = []
        for pp in parent_pairs:
            s = [np.random.randint( n_func )]
            for select in range(n_func): # Mix every function
                    tot_d = float('Inf')     # Bloat control
                    n = float('Inf')
                    
                    while (tot_d > max_d or n > max_n_nodes):
                        p1_1, p1_2, p2 = copy(pp)
                        # Choose random idx from parent 1 and 2
                        l1 = len(p1_1.trees_idx[select])
                        l2 = len(p2.trees_idx[select])
                        idx1 = np.random.choice( range(1, l1) )
                        idx2 = np.random.choice( range(1, l2) )

                        # Crossing
                        substitute(p1_1.trees[select], p1_1.trees_idx[select][idx1],
                                   tree_by_idx( p2.trees[select], p2.trees_idx[select][idx2] ))

                        substitute(p2.trees[select], p2.trees_idx[select][idx2],
                                   tree_by_idx( p1_2.trees[select], p1_2.trees_idx[select][idx1] ))

                        tot_d = max(depth(p1_1.trees[select]), depth(p2.trees[select]))
                        n = max(n_nodes(p1_1.trees[select]), n_nodes(p2.trees[select]))

            # Update and store
            p1_1.update(); p2.update()
            res.extend([p1_1, p2])
        return res

    def pool_evaluate(inds):
        res = do_pool(inds, 'evaluate')
        return res

    def pool_mutate(inds):
        res = do_pool(inds, 'mutate')
        return res
    
    def pool_recombine(inds):
        res = do_pool(inds, 'recombine')
        return res

    def do_pool(inds, arg): #, f_function, n_procs):
        global n_procs
        # Assign
        jobs = []
        l = len(inds)
        size = l // n_procs
        res = []
        for i in range(n_procs):
            my_inds = inds[ size * i : size * (i+1)]
            if i == (n_procs - 1):
                my_inds = inds[ size * i : ]  
            jobs.append( my_inds)

        # Collect
        res = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = executor.map( shared_evaluate, jobs)
            
            if arg == 'evaluate':
                pool = executor.map( shared_evaluate, jobs)
            if arg == 'mutate':
                pool = executor.map( shared_mutate, jobs)
            if arg == 'recombine':
                pool = executor.map( shared_recombine, jobs)
            
            for r in pool:
                res.extend(r)
        if arg == 'evaluate':
            res = np.array(res).flatten()
            #print(res.shape)
            """
            print(len(res[:][0]), res[:][0][0])
            print(len(res[:][1]), res[:][1][0])
            res2 = [res[:][0], res[:][1]]
            res = res2
            """
            pass
        #if arg == 'mutate':
        #    print(res)
        return res

########################################################
# Fitness function.
########################################################
    # This variable store calls to the fitness functions
    calls = 0
    
    # Fitness function returns:
    #
    def f_function(trees, trees_idx, save_img = False, cont = []):
        global apollo, n_mc_iter, rnd_mc_iter, seed
        global pars_alpha, calls
        #global seeds, cont
        
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

        # Secure truncation #Not needed
        #F1 = np.exp(np.clip(F1, -1e2+10, 1e2-10))
        #F2 = np.exp(np.clip(F2, -1e2+10, 1e2-10))
        #F3 = -np.exp(np.clip(F3, -1e2+10, 1e2-10))

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

        # avg_u (control cost)
        mag_u =[]
        d_ref = []
        n_dif = [] 
        for t in trajs:
            u = t.u
            v = t.X[:,2]
            u_ref = []
            for cont in range(v.shape[0]):
                u_ref.append( apollo.ref_data.get_u_by_velocity( v[cont] ) )
                
            # Ref. difference:
            my_d_ref = u - u_ref   # No abs
            d_ref.append( np.mean(my_d_ref ) )

            # n difs u: d_u
            # u  =  u[:-1] - u[1:]
            # n = 0
            #for c_u in range(len(u)-1):
            #   if np.sign( u[c_u]) != np.sign( u[c_u+1])
            #       n += 1
            # n_dif.append(n)
            
            # Integrate u:
            diff_u =  u[:-1] - u[1:]
            mag_u.append( np.sum( np.abs(diff_u)) )
            
        avg_u = np.mean(mag_u)
        avg_d_ref = abs( np.mean(d_ref) ) #Absolute here
        #n_dif = np.mean(n_dif)

        # Saturation time:
        avg_sat = []
        for t in trajs:
            u = t.u
            avg_sat.append(  len( np.where(abs(u) >= (np.pi / 2.) )[0] ) / len(t.X) )
            
        avg_sat = np.mean(avg_sat)
            
        # Objetives
        nn = np.sum( [n_nodes(each) for each in trees ] )
        avg_disp = avg_disp + pars_alpha * nn
        objs = [avg_disp, avg_u, avg_sat]

     
        

        
        calls += 1

        return [avg_disp, [objs]]

########################################################
# Random initialization and random seeds.
########################################################
    np.random.seed(22)  #21
    n_runs = 5
    seeds = ((np.random.rand(n_runs)*1e6)+100).astype(np.int)

    if save: print('\n'*5 + 'SAVING RESULTS!' + '\n'*5)
    print('Multi:', use_multi)
    
    init_run = 0
    for run in runs: ###range(0, 1):
    # Reset calls in each run
        calls = 0
        print('\n Run:', str(run+1))
        print(labels_f.format(*(labels)))
        
########################################################
# Associating random seeds to runs allows
# to continue interrupted simulations.
########################################################
        seed = seeds[run]
        np.random.seed(seed)
    
# Outuput file
        if save:
            condition = 'GP_' + str(Gamma) + '_' + str(p_c) + '_' + str(run+1)
            name = output_folder + '/' + condition  + '.txt'
            f = open(name, 'w')
            cad = ''
            for each in labels[1:]:
                cad += each + '\t'
            f.write( cad + '\n')
    
########################################################
# Create population.
########################################################
        pop = Population()
# Set the fitness function
        pop.set_fitness_function(f_function)
        pop.multi_ev_f = pool_evaluate
        pop.multi_mut_f = pool_mutate
        pop.multi_recombine_f = pool_recombine

# First random generation of individuals of lenght the number of objects
        pop.first_generation()
        # Write
        cont=-1; t = 0
        if save:
            string = ''
            for each in pop.performance():
                string += "%.4g" % each + '\t'
            string += "%s" % calls + '\t'
            string += "%.4g" % t + '\n'
            f.write(string)

        # Show
        string_list = ["%3d" % (cont + 1)]
        string_list += ["%.4g" % each for each in pop.performance()]
        string_list += [str(calls)]
        string_list += ["%.4g" % t]
        print(labels_f.format(*string_list))
        
# Stopping criterion is n. iterations
        for cont in range (max_gen):
            calls = 0
            start = time.perf_counter()
        
            # Recombine
            pop.recombine()
        
            # Mutate
            pop.mutate()
        
            # Reproduce
            pop.reproduce()
        
            # Selection
            pop.selection(run, cont+1, output_folder)
            end = time.perf_counter()
            t = end - start

            # Save best until now
            if cont == 0 or pop.best()[1] < pop.best(-2)[1]:
                f_function(pop.best()[0], [tree_idx(t) for t in pop.best()[0]], save_img = True, cont = cont +1)

                # Save best til now
                
                name = output_folder + '/' +str(cont)
                f2 = open(name, 'wb')
                pickle.dump(pop.best()[0], f2)
                f2.close()
                
  
            # Write to 4 s.f.
            if save:
                string = ''
                for each in pop.performance():
                    string += "%.4g" % each + '\t'
                string += "%s" % calls + '\t'
                string += "%.4g" % t + '\n'
                f.write(string)

            # Show
            if (cont == 0 or (cont + 1) % 1 == 0):
                #print (' ', "%3d" % (cont + 1), string[:-2])
                string_list = ["%3d" % (cont + 1)]
                string_list += ["%.4g" % each for each in pop.performance()]
                string_list += [str(calls)]
                string_list += ["%.4g" % t]
                print(labels_f.format(*string_list))
        
            if save:
                """
                # Early stop to prevent overfitting
                if pop.best()[1] < stop_value:
                    f.close()
                    sys.exit(0)
                """
                # Save the best
                name = output_folder + '/' + condition + '_best' #_' + str(cont+1)
                f2 = open(name, 'wb')
                pickle.dump(pop.best()[0], f2)
                f2.close()
                
        print()
        if save: f.close()
        """
        # Save best
        if save:
            name = GA_output_f + '/' + condition + '_best_' + str(run+1)
            with open(name, 'wb') as f2: pickle.dump(pop.best()[0], f2)
            #with open(name, 'rb') as f: r = pickle.load(f)
        """

    
