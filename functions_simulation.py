"""
NPCGC:
Monte Carlo simulation.
"""

############################################
# Imports
############################################
import numpy as np
import pickle

from class_apollo_controller import *
from config_reference import *
from config_model import *

############################################
# Monte-Carlo simulation
############################################

# Monte carlo simulations given a controller, n_iter and random seed
# Use
#   global parameters in config_reference
#   eom in class_trajectory
# Return: list of trajectories

def mc_simulation(bank_controller, n_iter = 10, seed=0):
    global verbose

    ##Store random state
    state = np.random.get_state()

    # Random number generator
    random = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))

    # 3-sigma variations to be simulated
    beta_3sigma = 0.05 * params['beta']
    LD_3sigma = 0.08 * params['LD']
    gamma_3sigma= 0.6 * np.pi/180
    
    # Generate random spreads for parameters and initial conditions
    beta_vec = random.normal(params['beta'], beta_3sigma/3, n_iter) #n_mc_iter)
    LD_vec = random.normal(params['LD'], LD_3sigma/3, n_iter) #mc_iter)
    gamma0_vec = random.normal(gamma0, gamma_3sigma/3, n_iter) #mc_iter)

    results = []
    for i in range(n_iter):
        if verbose:
            if (i % 10) == 0:
                print('%d/%d ' % (i, n_mc_iter), end ='')

        params_i = params.copy()
        params_i['beta'] = beta_vec[i]
        params_i['LD'] = LD_vec[i]
        X0 = np.array([h0, s0, V0, gamma0_vec[i]])
        
        traj_i = simulate_entry_trajectory(
            traj_eom,
            t0, 
            tf,
            X0,
            2, # V
            600, # V_f
            params_i,
            bank_angle_fn = bank_controller,
            t_eval = tspan)
        results.append(traj_i)
    if verbose: print()

    ## Restore random state
    np.random.set_state(state)
    return results

def save_results(f_name, results):
    with open(f_name, 'wb') as f:
        pickle.dump(results, f)
    return

def footprint(results, kind = 'variance'): ##RANGE
    # Footprint: dispersion of final range position in Km
    # Get s(range, idx = 1) of last row
    S = np.array([traj.X[-1][1] for i, traj in enumerate(results)])
    if kind == 'range':
        res = abs(max(S) - min(S))/1e3
    if kind == 'variance':
        res = np.var(S * 1e-3)       
    return res

# Weighted penalization differences between trajectories and controller reference
def diff_to_ref(bank_controller, results):
    
    # reference: h, s, v, Gam 
    ref_h = bank_controller.ref_data.data[:,1]
    ref_s = bank_controller.ref_data.data[:,2]
    ref_v = bank_controller.ref_data.data[:,3]
    ref_Gam = bank_controller.ref_data.data[:,4]
    
    res = []
    # trajectories:
    hs = np.array([traj.X[:,0] for i, traj in enumerate(results)])
    ss = np.array([traj.X[:,1] for i, traj in enumerate(results)])
    vs = np.array([traj.X[:,2] for i, traj in enumerate(results)])
    Gams = np.array([traj.X[:,3] for i, traj in enumerate(results)])


    res = []
    for cont in range(len(results)):
        """
        # h weighted penalization parameters
        U = 40e3; K_0 = 1.; K_1 = 10
        ind = np.sum(d_h * (d_h < U) * K_0) + np.sum(d_h * (d_h > U) * K_1)
        """
        ind = 0.
        # s weighted penalization parameters h 5e3 s 2e3
        U = 1e3; K_0 = 1.; K_1 = 10
        # Trajectory data must be truncated by right to ref
        point = len(ref_h) - len(hs[cont])
        if point >= 0:
            d_h = np.abs(hs[cont] - ref_h[point:])
        else:
            d_h = np.abs(hs[cont][:point] - ref_h[:])
            
        #print(hs[cont].shape, ref_h.shape)

       
        #print(d_v, np.sum(d_v), '\n')
        ind += np.sum(d_h)
        #ind += np.sum(d_v * (d_v< U) * K_0) + np.sum(d_v * (d_v > U) * K_1)

        """
        # v weighted penalization parameters
        U = 1e3; K_0 = 1.; K_1 = 10
        d_v = np.abs(vs[cont] - ref_v)
        ind += np.sum(d_v * (d_v < U) * K_0) + np.sum(d_v * (d_v > U) * K_1)
        
        # Gam weighted penalization parameters
        U = .2; K_0 = 1.; K_1 = 10
        d_G = np.abs(Gams[cont] - ref_Gam)
        ind += np.sum(d_G * (d_G < U) * K_0) + np.sum(d_G * (d_G > U) * K_1)
        """
        res.append(ind)
   
    return np.sum(res)

def simulation_image(results, ref, path = './simulation.png'):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    num_iter = len(results)
    # Plot 
    plt.clf()
    colors = [ cm.Blues(x) for x in np.linspace(0.0, 1.0, num_iter)]
    for res, c in zip(results, colors):
        #plt.plot(range(res.X[:,2].shape[0]), res.X[:,0]/1e3, color=c)
        plt.plot(res.X[:,2]/1e3, np.degrees(res.u), color=c)
    ref_line = plt.plot(ref.X_and_lam[:,2]/1e3, np.degrees(ref.u), 'r-')
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Commanded Bank Angle [deg]')
    plt.ylim(-5., 185.)
    plt.savefig(path)
    return







