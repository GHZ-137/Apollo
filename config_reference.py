"""
Initial conditions and model definition
"""
############################################
# Imports
############################################
import numpy as np

# Initial conditions
h0 = 120e3; # Entry altitude
V0 = 5500;  # Entry velocity
gamma0_deg = -14.5; # Entry flight path angle
s0 = 0

# Reference params
params = {'H': 11.1e3,
          'rho0': 0.020, # kg/m^3
          'beta': 120,
          'LD': 0.24,
          'R_m': 3380e3,
          'g': 3.73}

# Terminal velocity
v_f = 600

gamma0 = np.deg2rad(gamma0_deg)
X0 = np.array([h0, s0, V0, gamma0])
t0 = 0
tf = 500.
tspan = np.linspace(t0, tf, 101) #101

# Terminal altitude
h_f = 0e3
