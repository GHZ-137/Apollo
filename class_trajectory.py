"""
Functions for equation of motion,
to simulate entry trajectory
and trajectory class to keep the results.
.
"""

############################################
# Imports
############################################
import numpy as np
from scipy.integrate import solve_ivp

# Fix Python 3's weird rounding function
# https://stackoverflow.com/a/44888699/538379
round2=lambda x,y=None: round(x+1e-15,y)

############################################
# Equation of motion of the trajectory
############################################
def traj_eom(t: float, 
             state: np.array, 
             params: dict, 
             bank_angle_fn #: Callable[[float, np.array, dict], float]
            ):
    """ Equation of motion for a state, with param and bank function"""
    h, s, v, gam = state
    u = bank_angle_fn(t, state, params)
    
    rho0 = params['rho0']
    H = params['H']
    beta = params['beta']   # m/(Cd * Aref)
    LD = params['LD']
    R_m = params['R_m']
    g = params['g']
    
    v2 = v*v
    rho = rho0 * np.exp(-h/H) 
    D_m = rho * v2 / (2 * beta)  # Drag Acceleration (D/m)
    r = R_m + h
    return np.array([v * np.sin(gam),       # dh/dt
                     v * np.cos(gam),       # ds/dt
                     -D_m - g*np.sin(gam),  # dV/dt
                     (v2 * np.cos(gam)/r + D_m*LD*np.cos(u) - g*np.cos(gam))/v] # dgam/dt
                   )

############################################
# Trajectory class and forward integration function
#
############################################
class Trajectory:
    """Data structure for holding the result of a simulation run"""
    def __init__(self, t: float, X: np.array, u: np.array, params: dict):
        self.t = t
        self.X = X  # [h, s, v, Gam]
        self.u = u
        self.params = params

def simulate_entry_trajectory(eom, #: Callable[[float, np.array], np.array], 
                              t0: float, 
                              tf: float, 
                              X0: np.array, 
                              term_var_idx: int,  # Index of ther terminal variable in X
                              term_var_val: float, # Terminal value of the terminal variable
                              params: dict,
                              bank_angle_fn, #: Callable[[float, np.array, dict], float],
                              t_eval= None): #Optional[np.array] = None)
    """ Simulate a trajectory  from initial conditions until a stop even occurs indicated
        by term_var_idx and term_var_val and evaluate at instants in t_eval.
        Return a trajectory object."""

    altitude_stop_event = lambda t, X, params, _: X[term_var_idx] - term_var_val
    altitude_stop_event.terminal = True
    altitude_stop_event.direction = -1
    
    output = solve_ivp(eom, 
                   [t0, tf], 
                   X0, 
                   args=(params, bank_angle_fn), 
                   t_eval=t_eval,
                   rtol=1e-6, events=altitude_stop_event)
    
    # Get the used bank angle in the trajectory
    # Transpose y so that each state is in a separate column and each row 
    # represents a timestep
    num_steps = len(output.t)
    u = np.zeros(num_steps)
    for i, (t, X) in enumerate(zip(output.t, output.y.T)):
        u[i] = bank_angle_fn(t, X, params)
        
    return Trajectory(output.t, output.y.T, u, params)

