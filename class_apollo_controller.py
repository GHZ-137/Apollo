"""
NPCGC:
Class to simulate Apollo guidance control,
with methods to load reference data and
closed loop guidance
"""

############################################
# Imports
############################################
import numpy as np
from scipy.integrate import solve_ivp

from class_trajectory import *
from class_apollo_reference_data import *
from functions_optimization import *
from functions_plot import *
############################################
# Closed-loop Apollo guidance controller class
#
############################################
class apollo_controller:   
    def __init__(self, ref_data: ApolloReferenceData, K = 5.0):
        """K: Over-control gain"""
        self.ref_data = ref_data
        self.K = K      
        self.min_D_m = 0.05 * 9.81  # 5% g of Drag acceleration is the trigger

    @staticmethod
    def from_file(filename: str, K = 5.0):
        """Loads reference data from a file and initializes a new guidance controller"""
        ref_data = ApolloReferenceData.load(filename)
        return apollo_controller(ref_data, K)

    def reference_bank_angle(self, t, state, params):
        """Bank angle function for open loop guidance that simply returns
           the reference bank angle"""
        v = state[2]
        if v >= 3500:
            return np.deg2rad(75);
        elif v <= 1500:
            return np.deg2rad(50);
        else:
            return np.deg2rad(50 + (75-50)*(v-1500)/(3500-1500))
    
    def closed_loop_guidance(self, t, state, params):
        global order_to_fit
        h, s, v, gam = state
  
        rho0 = params['rho0']
        beta = params['beta']
        H = params['H']
        
        rho = rho0 * np.exp(-h/H)
        D_m = rho * v * v / (2 * beta)  # Drag Acceleration (D/m)
        
        # Compute reference bankangle
        phi = self.reference_bank_angle(t, state, params)
        
        # Wait for sensed G force to exceed threshold before starting
        # closed loop guidance
        if abs(D_m) < self.min_D_m and h > 60e3:
            return phi

        ref_data_row = self.ref_data.get_row_by_velocity(v)

        s_ref = ref_data_row[2]
        F1, F2, F3, D_m_ref, hdot_ref = ref_data_row[5:10]
        hdot = v * np.sin(gam)
            
        
        # Add correction based on Apollo guidance algorithm
        dphi = self.K * ( -(s-s_ref) - F2*(hdot-hdot_ref) - F1*(D_m - D_m_ref ))/F3

        phi = phi + dphi
        phi = abs(phi)
        phi = max(min(phi, np.pi), 0)
        return phi
