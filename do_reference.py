"""
NPCGC:
Create reference trajectory (APGD)
"""

############################################
# Imports
############################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Fix Python 3's weird rounding function
# https://stackoverflow.com/a/44888699/538379
round2=lambda x,y=None: round(x+1e-15,y)

from class_trajectory import *
from class_apollo_reference_data import *
from functions_plot import *
from config_reference import *

############################################
# Variables
############################################
images_f = './reference_images'
output_f = './reference'

############################################
# Reference bank and EOM
#
############################################
def reference_bank_angle(t, X, params):
    """Generates the reference bank angle profile with arguments:
        t(time), X(state) and params(constants)"""
    v = X[2]    
    if v >= 3500:
        return np.deg2rad(75);
    elif v <= 1500:
        return np.deg2rad(50);
    else:
        return np.deg2rad(50 + (75-50)*(v-1500)/(3500-1500))

V = np.linspace(1000, 4000, 301)
bank_angle_deg = np.degrees([reference_bank_angle(0, [0, 0, v, 0], {}) for v in V], dtype=np.float64)


plt.plot(V, bank_angle_deg)
plt.xlabel('V [m/s]')
plt.ylabel('Bank Angle [deg]')
plt.grid(True)
plt.title('Reference Bank Angle')
plt.savefig(output_f + '/ref_bank.png')


############################################
# Create reference trajectory:
# 1.Forward integration:
# Return a trajectory object with: t, X [h s v G], u, params
############################################
ref_traj = simulate_entry_trajectory(traj_eom, t0, tf, X0, 2, v_f, params, reference_bank_angle, tspan)

"""
plt.plot(ref_traj.X[:,2]/1e3, ref_traj.X[:,0]/1e3)
plt.xlabel('V [km/s]')
plt.ylabel('h [km]')
plt.grid(True)
plt.savefig(images_f + '/ref_traj_V-t.png')
"""

############################################
# Co-states equation of motion and
# reverse-integration
############################################
def traj_eom_with_costates(t: float, 
             state: np.array, 
             params: dict, 
             bank_angle_fn #: Callable[[float, np.array, dict], float]
            ):
    """ Equation of motion with lambda co-states (one for each variable), to be used
        in reverse order integration using target state as initial condition
    """
        
    lamS = 1
    h, s, V, gam, lamH, lamV, lamGAM, lamU = state   

    u = bank_angle_fn(t, state, params)
    
    rho0 = params['rho0']
    H = params['H']
    beta = params['beta']
    LD = params['LD']
    R_m = params['R_m']
    g = params['g']
    
    r = R_m + h
    
    v = V
    V2 = V*V
    rho = rho0 * np.exp(-h/H) 
    D_m = rho * V2 / (2 * beta)  # Drag Acceleration (D/m)
      
    lamHdot = D_m*LD*lamGAM*np.cos(u)/(H*v) - D_m*lamV/H + lamGAM*v*np.cos(gam)/r**2
    lamVdot = D_m*LD*lamGAM*np.cos(u)/v**2 - LD*lamGAM*rho*np.cos(u)/beta - g*lamGAM*np.cos(gam)/v**2 - lamGAM*np.cos(gam)/r - lamH*np.sin(gam) - lamS*np.cos(gam) + lamV*rho*v/beta
    lamGAMdot = -g*lamGAM*np.sin(gam)/v + g*lamV*np.cos(gam) + lamGAM*v*np.sin(gam)/r - lamH*v*np.cos(gam) + lamS*v*np.sin(gam)
    lamUdot = LD*lamGAM*rho*v*np.sin(u)/(2*beta)
    
    return np.array([V * np.sin(gam),       # dh/dt
                     V * np.cos(gam),       # ds/dt
                     -D_m - g*np.sin(gam),  # dV/dt
                     (V2 * np.cos(gam)/r + D_m*LD*np.cos(u) - g*np.cos(gam))/V, # dgam/dt

                     lamHdot,
                     lamVdot,
                     lamGAMdot,
                     lamUdot]
                   )
############################################
# 2.Inverse integration
############################################
# Reverse
ref_tf = ref_traj.t[-1]
ref_tspan_rev = ref_traj.t[::-1] # Reverse the time span
Xf = np.copy(ref_traj.X[-1,:]) # Get final state

# Ensure monotonic decreasing V
def V_event(t,X,p,_):
    return X[3] - 5500

V_event.direction = 1
V_event.terminal = True

# Reverse integration with co-states
X_and_lam0 = np.concatenate((Xf, [-1/np.tan(Xf[3]), 0, 0, 0]))
output = solve_ivp(traj_eom_with_costates, # lambda t,X,p,u: -traj_eom_with_costates(t,X,p,u), 
                   [ref_tf, 0], # Reversed time span
                   X_and_lam0, 
                   t_eval=ref_traj.t[::-1], # Reversed evaluated instants
                   rtol=1e-8,
                   events=V_event,
                   args=(params, reference_bank_angle))
lam = output.y.T[:,4:][::-1] # All instants, from 4th column and time reversed
X_and_lam = output.y.T[::-1] # All instants, all columns and time reversed

############################################
# Save Apollo reference data
# Return a ApolloReferenceData with:
# X_and_Lam (8), u, tspan, params
############################################
ref = ApolloReferenceData(X_and_lam, ref_traj.u, ref_traj.t, params)
ref.save(output_f + '/apollo_data_vref.npz')

# Load data back and check that it matches the original
#ref2 = ApolloReferenceData.load(output_f + '/apollo_data_vref.npz')
#assert np.allclose(ref2.data, ref.data)


############################################
# Plot
############################################
plt.clf()
plt.plot(ref.tspan, ref.X_and_lam[:,0]/1e3)
plt.xlabel('t [s]')
plt.ylabel('h [Km]')
plt.grid(True)
plt.title('Reference')
plt.savefig(images_f + '/ref_h.png')

plt.clf()
plt.plot(ref.tspan, ref.X_and_lam[:,1]/1e3)
plt.xlabel('t [s]')
plt.ylabel('s [Km]')
plt.grid(True)
plt.title('Reference')
plt.savefig(images_f + '/ref_s.png')

plt.clf()
plt.plot(ref.tspan, ref.X_and_lam[:,2])
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.grid(True)
plt.title('Reference')
plt.savefig(images_f + '/ref_v.png')

plt.clf()
plt.plot(ref.tspan, np.rad2deg(ref.X_and_lam[:,3]))
plt.xlabel('t [s]')
plt.ylabel('Gamma [deg]')
plt.grid(True)
plt.title('Reference')
plt.savefig(images_f + '/ref_G.png')

plt.clf()
plt.plot(ref.tspan, ref.u)
plt.xlabel('t [s]')
plt.ylabel('[u]')
plt.grid(True)
plt.title('Reference')
plt.savefig(images_f + '/ref_u.png')




####
F1, F2, F3 = ref.get_gains()

plt.clf()
plt.plot(ref.X_and_lam[:,2], F1)
plt.xlabel('v')
plt.ylabel('F1')
plt.grid(True)
plt.title('F1')
plt.savefig(images_f + '/F1.png')

plt.clf()
plt.plot(ref.X_and_lam[:,2], F2)
plt.xlabel('v')
plt.ylabel('F2')
plt.grid(True)
plt.title('F2')
plt.savefig(images_f + '/F2.png')

plt.clf()
plt.plot(ref.X_and_lam[:,2], F3)
plt.xlabel('v')
plt.ylabel('F3')
plt.grid(True)
plt.title('F3')
plt.savefig(images_f + '/F3.png')


####
from functions_optimization import *
F1 = linearize(F1, 8, np.min(F1), np.max(F1)) #, np.max(F1)-np.min(F1) )
F2 = linearize(F2, 8, np.min(F2), np.max(F2)) #, np.max(F2)-np.min(F2) )
F3 = linearize(F3, 8, np.min(F3), np.max(F3)) #, np.max(F3)-np.min(F3) )
#ref.set_gains([F1, F2, F3])

plt.clf()
plt.scatter(ref.X_and_lam[:,2], F1)
plt.xlabel('v')
plt.ylabel('F1')
plt.grid(True)
plt.title('F1')
plt.savefig(images_f + '/lin_F1.png')

plt.clf()
plt.scatter(ref.X_and_lam[:,2], F2)
plt.xlabel('v')
plt.ylabel('F2')
plt.grid(True)
plt.title('F2')
plt.savefig(images_f + '/lin_F2.png')

plt.clf()
plt.scatter(ref.X_and_lam[:,2], F3)
plt.xlabel('v')
plt.ylabel('F3')
plt.grid(True)
plt.title('F3')
plt.savefig(images_f + '/lin_F3.png')

