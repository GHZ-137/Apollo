"""
GP classes:
    Individual
    Population

"""
########################################################
# Inclusions.
########################################################
import pickle, numpy as np
from copy import deepcopy
from functions_trees import *
from functions_cma_es import *
from functions_NSGA import *

from config_GP import *

########################################################
# Class individual. Each ind. is a list of n_func trees with
#  init_max_d, max_d, min_d,
#  method - initialization method (full, grow)
# Performs random initialization and mutation
########################################################
class Individual:
    def __init__(self, my_method = []):
        # Init trees use method GROW
        global n_func, min_d, method
        
        trees = []
        for cont in range(n_func):
            trees.append( tree(init_d, method) )
        self.trees = trees
        self.vars = self.trees
        self.trees_idx = [ tree_idx(each) for each in self.trees]
        return

    def mutate(self):
        global max_d, max_n_nodes
        # Mutate one of the functions
        #s = [np.random.randint( n_func )]
        s = range(n_func)  # << Mutate all functions
        
        for select in s:
            # Get random node from selected function
            node_idx = self.trees_idx[select][ np.random.choice( range(1, len(self.trees_idx[select] )))]

            # Choose sub-tree or value mutation
            opt = np.random.rand() < p_subtree  # % subtree mutation
            ##opt = 0

            tree_cp = deepcopy(self.trees[select])
            if True: #opt:
                # Subtree mutation
                 
                """
                #Substitute subtree in node_idx by a random tree of same depth
                sub_tree = tree_by_idx(tree_cp, node_idx)
                d = depth(sub_tree)
                new_node = tree(d, 'full')
                tree_cp = substitute(tree_cp, node_idx, new_node)
                """
                #Substitute subtree in node_idx by a random tree of min depth
                if (depth(tree_cp) + min_d) <= max_d:
                    new_node = tree(min_d) #Grow full
                    tree_cp = substitute(tree_cp, node_idx, new_node)
                    
                
            else:
                # Value substitution
                var_list = extract_tree_num(tree_cp)
                if var_list != []:
                    var_list[np.random.choice( range(len(var_list)))] += np.random.normal(0., sigma)
                    tree_cp = insert_tree_num(tree_cp, var_list)
                     
            self.trees[select] = tree_cp
            self.update()
        return
    
    def update(self):
        self.trees_idx = [ tree_idx(each) for each in self.trees]
        self.vars = self.trees
        return
    
# Tests
"""
a = Individual()
for each in a.trees: print(expr(each, dec = 2), depth(each), n_nodes(each))
print()
a.mutate()
for each in a.trees: print(expr(each, dec = 2), depth(each), n_nodes(each))
"""
########################################################
# Population class.
########################################################
class Population:
    def __init__(self):
        # Constructor method.
        # generations is initially an empty list
        self.generations = []
        # List of ind. idx ordered by fitness. First(0) -> Best 
        self.fit_order = []
        return

    def set_fitness_function(self, p_function):
        # Set fitness function pointer
        self.fitness_function = p_function
        return

    def first_generation(self):
        global Mu
        generation = []
        for cont in range(Mu):
            generation.append( Individual() )

        self.generations.append(generation)
        self.evaluate()
        return

    def evaluate(self, n_gen=-1):
        # Store fitness value of each ind. in value.
        # Applies to n_gen number (default -1, present generation)
        global optimization
        
        # Evaluate individuals
        if use_multi:
            inds = self.generations[n_gen]
            #self.multi_ev_f(inds, self.fitness_function, n_procs)
            #print(len(inds))
            res = self.multi_ev_f(inds)
            res = np.array(res)
            res = np.reshape(res, (len(res)// n_objs, n_objs))
            # Control cost from Pi
            ##res[:,1] = np.pi - res[:,1]
            
            for cont in range(len( self.generations[n_gen] )):
                self.generations[n_gen][cont].value = res[cont][0]
                self.generations[n_gen][cont].objs = res[cont]
                
        else:
            for ind in self.generations[n_gen]:
                ind.value, ind.objs = self.fitness_function(ind.vars, ind.trees_idx)
                


        fits = [ind.value for ind in self.generations[n_gen]]
        if optimization == 'Minimize':
            f_order = np.argsort(fits)
        elif optimization == 'Maximize':
            f_order = np.argsort(fits)[::-1]
        self.fit_order.append(f_order)

        #print(len(diags), diags[0].shape, f_order[0])
        return

    def best(self, n_gen=-1):
        # Best individual in a generation.
        # Return vars. [trees], fitness and index
        
        idx = self.fit_order[n_gen][0]
        best_ind = self.generations[n_gen][idx].vars
        best_val = self.generations[n_gen][idx].value
        return [best_ind, best_val, idx]

    def worst(self, n_gen=-1):
        # worst individual in a generation.
        # Return vars. (tree), fitness and index
        
        idx = self.fit_order[n_gen][-1]
        w_ind = self.generations[n_gen][idx].vars
        w_val = self.generations[n_gen][idx].value   
        return [w_ind, w_val, idx]

    def average(self, n_gen=-1):
        # Average ind. fitness in n_gen number
        avg = 0  
        for ind in self.generations[n_gen]:
            avg += ind.value
        avg /= len(self.generations[n_gen])   
        return avg

    def tournament(self, prev = -1, n = 2):
        # List of n individuals by n random tournments of size Gamma.
        # 'prev' indicates to use last (-1, default) or previous to last generation (-2).
        # 'prev = -2' should be used in sequential recombination + mutation + reproduction
        
        global Gamma
        res = []
        for cont in range(n):
            # Random indices
            rnd_idx = np.round((np.random.rand(Gamma)) * (len(self.fit_order[prev + 1])-1)).astype(np.int)
            
            # Fitness order of random indices and best of them
            my_fit_order = self.fit_order[prev + 1][rnd_idx]  
            best_of_rnd_idx = np.where(my_fit_order == np.min(my_fit_order))[0][0]

            ##print(rnd_idx, my_fit_order, best_of_rnd_idx)
            ##print( rnd_idx[best_of_rnd_idx ])
            sel = rnd_idx[best_of_rnd_idx ]
            
            best_ind = self.generations[prev][sel]
            
            res.append( best_ind )
        return res
    
    def recombine(self, cont_gen = 0):
        # Get (Mu * p_c) children by recombining two random tournament parents
        # with bloat control
        # FALSE MULTI
        
        global Mu, p_c, elitism, max_d, max_n_nodes

        # If elitism, add previous best to children
        # Except if NSGA
        if elitism == 'Yes' and not do_NSGA:
            children = [ self.generations[-1][self.best(-1)[2]] ]
        else:
            children = []

        parent_pairs = []
        for cont in range( np.round(Mu * p_c / 2.).astype(int) ):
            # 2 parents by tournament from previous generation.
            parents = self.tournament(prev = -1, n = 2)

            # unless NSGA -> random selection
            if do_NSGA:
                idx1 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                idx2 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                parents = [self.generations[-1][idx1], self.generations[-1][idx2]]

            
            # Parents need cloning and duplicating
            parent_pairs.append([deepcopy(parents[0]), deepcopy(parents[0]), deepcopy(parents[1])])

        if False: #use_multi:
            res = self.multi_recombine_f(parents)
            children.extend(res)
        else:

            for pp in parent_pairs:
                ##s = [np.random.randint( n_func )]
                s = range(n_func)           # << Cross all functions
                for select in s: #range(n_func): #One function # Mix every function
                    
                    tot_d = float('Inf')     # Bloat control
                    n = float('Inf')
                    while (tot_d > max_d or n > max_n_nodes):
                        p1_1, p1_2, p2 = deepcopy(pp)
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
                        """
                        # Interchange a pair of trees
                        p1_1.trees[select] = deepcopy(p2.trees[select])
                        p2.trees[select] = deepcopy(p1_2.trees[select])
                        """

                        
                        tot_d = max(depth(p1_1.trees[select]), depth(p2.trees[select]))
                        n = max(n_nodes(p1_1.trees[select]), n_nodes(p2.trees[select]))
                # Update and store
                
                p1_1.update(); p2.update()
                children.extend([p1_1, p2])


        self.generations.append( children )
        return

    def mutate(self, cont_gen = 0):
        # Get (Mu * p_m) children by mutation.
        # As we have stored previous recombination results,
        # parents are selected from previous to last generation (n = -2)
        
        global Mu, p_m
        parents = []
        for cont in range( np.round(Mu * p_m ).astype(int) ):
            parent = deepcopy(self.tournament(prev = -2, n = 1))[0]
            # unless NSGA -> random selection
            if do_NSGA:
                idx1 = np.round((np.random.rand()) * (len(self.generations[-1])-1)).astype(np.int)
                parent = self.generations[-1][idx1]
                    
            
            parents.append( parent )
        children = []
 
        if use_multi:
            res = self.multi_mut_f(parents)
            children.extend(res)
            
        else:
            for parent_cp in parents:
                parent_cp.mutate()
                children.append(parent_cp)

        self.generations[-1].extend( children )
        return

    def reproduce(self):
        # Get (Mu * p_r) children by selection
        # from previous to last generation (n = -2)
        global Mu, p_r, freq_cma, optimization

        children = []
        for cont in range( Mu - len(self.generations[-1]) ):
            parent = self.tournament(prev = -2, n = 1)[0]
            children.append(parent)
        self.generations[-1].extend( children )

        # Finally, do not forget evaluating present generation!
        self.evaluate()

        # cma after improvement?
        best_prev = self.best(n_gen=-2)[1]
        best = self.best(n_gen=-1)[1]
        
        if (freq_cma == -1 ) and \
        ((optimization == 'Maximize' and best > best_prev)\
           or (optimization == 'Minimize' and best < best_prev)):
            # Inds. to apply cma
            #
            my_fit_order = self.fit_order[-1]
            l_inds = [self.generations[-1][each] for each in my_fit_order][: int(np.round(Mu * frac_ind_cma))]
            print(len(l_inds))
            #l_inds = self.generations[-1]

            # For random fluctuationcs
            if True: #abs(best - best_prev) > 0.005:
                for ind in l_inds: #[self.best(-1)[2]]]: #If only the best
                    res = cma_es(ind.vars, ind.trees_idx)
                    #print(res[0])
                    #print()
                
                    ind.vars = res[0]
                    ind.tree = ind.vars
                
                self.evaluate()
        
        return

    def selection(self, run =0, cont = 0, out_folder = ''):
        # Generational substitution do nothing
        if do_NSGA: 
            # Join Mu and Lambda
            new_Mu = self.generations[-1] + self.generations[-2]
 
            objs = []
            for each in new_Mu:
                # Select which objectives
                objs.append( [each.objs[obj_1], each.objs[obj_2]])

            # Get sucessive fronts
            my_fronts = fronts(objs, optimization)
            my_fronts_idx = [front_idx(each, objs) for each in my_fronts]

            # CMA
            if do_NSGA_CMA:
                pass
                
            # Save
            if True: ###cont == 1 or cont == 100 or cont%10 == 0:
                f3 = open(out_folder + '/' + str(run+1)+ '_pareto_' + str(cont) , "wb")
                pickle.dump(my_fronts[0], f3)
                f3.close()

                p = np.array(my_fronts[0])
                yy = p[:,1]
                xx = p[:,0]
                plt.clf()
                plt.ylim(0, np.max(yy) * 1.15) # 40
                plt.xlim(0, np.max(xx) * 1.15) # 3.14
                for cont2 in range(1)[::-1]:
                    p = np.array(my_fronts[cont2])
                    yy = p[:,0]
                    xx = p[:,1]
                    plt.scatter(yy, xx)
                    plt.xlabel("Disp. [Km]")
                    plt.ylabel("Total u (º)")
                    plt.title('Pareto front. Gen:' + str(cont))
                    plt.savefig(out_folder + '/' + str(run+1)+ '_pareto_' + str(cont) + '.png') #

            # Assign fronts to new_Lambda
            new_Lambda = []
            #if elitism:
                #new_Lambda = [ self.generations[-2][self.best(-2)[2]] ]
            
            cont = 0
            while (len(new_Lambda) + len(my_fronts[cont])) <= Mu :
                to_add = [new_Mu[each] for each in my_fronts_idx[cont] ]
                new_Lambda.extend( to_add )
                ##print( len(new_Lambda) )
                cont += 1
                   
            # Truncate last front
            dist = crowd_dist(objs[cont])
            dist_idx = np.argsort(dist)[::-1]
            n_diff = Mu - len(new_Lambda)
            ##print(n_diff)

            to_add = [new_Mu[each] for each in my_fronts_idx[cont]]
            to_add2 = [to_add[each] for each in dist_idx[:n_diff]]
            new_Lambda.extend(to_add2)

            ##print(len(new_Lambda))
            
            fits = [ind.value for ind in new_Lambda ]
            new_fit_order = np.argsort(fits)
            if optimization == "maximize":
                new_fit_order = np.argsort(fits)[::-1]
                    
            self.generations[-1] = new_Lambda
            self.fit_order[-1] = new_fit_order
        return

    def performance(self, n_gen = -1):
        # Obtain best / average / worst performance in n_gen number
        # Max/ min / avg depth & number of nodes
        d_vals= []
        n_vals = []
        for i in self.generations[n_gen]:
            for each in i.trees:
                d_vals.append(depth(each))
                n_vals.append(n_nodes(each))

        return [self.best(n_gen)[1], self.average(n_gen), self.worst(n_gen)[1],
                np.min(d_vals), np.max(d_vals), np.mean(d_vals),
                np.min(n_vals), np.max(n_vals), np.mean(n_vals)
                ]


