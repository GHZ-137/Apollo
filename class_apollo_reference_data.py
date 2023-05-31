"""
Class to create the reference trajectory data
"""

############################################
# Imports
############################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from config_reference import *
from config_model import *

############################################
# Variables
############################################
output_f = './reference'

############################################
# Class for Apollo_Reference_Data
############################################
class ApolloReferenceData:
    def __init__(self, X_and_lam: np.array,
                 u: np.array, tspan: np.array, params: dict):
        """
        X_and_lam: [h, s, v, gam,  lamH, lamV, lamGAM, lamU] - 8 x n matrix
        tspan: 1 x n vector
        """
        self.X_and_lam = X_and_lam
        self.tspan = tspan
        self.params = params
        self.u = u
        
        assert len(X_and_lam.shape) == 2 and X_and_lam.shape[0] > 1, "Need at least two rows of data"
        self.num_rows = X_and_lam.shape[0]
        
        self.delta_v = abs(X_and_lam[1,2] - X_and_lam[0,2])
        assert self.delta_v > 0, "Reference trajectory has repeated velocites in different rows"
        
        self.start_v = X_and_lam[0][2]

        F1, F2, F3, D_m, hdot_ref = self._compute_gains_and_ref()
        F3[-1] = F3[-2]   # Account for F3=0 at t=tf
        # Stack the columns as follows:
        # [t, h, s, v, gam, F1, F2, F3, D/m]
        self.data = np.column_stack((tspan, X_and_lam[:,:4], F1, F2, F3, D_m, hdot_ref))

        """
        # Reorder by inceasing velocity
        self.v_dict = []
        all_v = self.get_v()
        idx = np.argsort(all_v)
        self.data = self.data[idx]

        # Compute v_dict to speed closest velocity
        all_v = self.get_v()
        for cont in range(len(all_v)-1):
            l = all_v[cont]
            h = all_v[cont + 1]
            m = 0.5 * (l + h)
            self.v_dict.append( [l, h, m])
        """
        
    def get_gains(self):
        F1 = self.data[:,5]
        F2 = self.data[:,6]
        F3 = self.data[:,7]

        res = [F1, F2, F3]
        return res

    def set_gains(self, gains):
        F1, F2, F3 = gains
        self.data[:,5] = F1
        self.data[:,6] = F2
        self.data[:,7] = F3
        return

    def get_v(self):
        #v = self.X_and_lam[:,2]
        v = self.data[:,3]
        return v

    def _compute_gains_and_ref(self):
        h = self.X_and_lam[:,0]
        v = self.X_and_lam[:,2]
        gam = self.X_and_lam[:,3]

        lamH = self.X_and_lam[:,4]
        lamGAM = self.X_and_lam[:,6]
        lamU = self.X_and_lam[:,7]
        
        rho0 = self.params['rho0']
        H = self.params['H']
        beta = self.params['beta']   # m/(Cd * Aref)

        v2 = v*v
        rho = rho0 * np.exp(-h/H) 
        D_m = rho * v2 / (2 * beta)  # Drag Acceleration (D/m)
        hdot = v * np.sin(gam)

        #AQUI
        F1 = H * lamH/D_m
        F2 = lamGAM/(v * np.cos(gam))
        F3 = lamU
        return F1, F2, F3, D_m, hdot

    def get_row_by_velocity(self, v: float):
        """
        Returns data row closest to given velocity
        """
        if not (0 <= v <= 6e3):
            if v < 0:
                v = 0.
            if v > 6e3:
                v = 6e3
                
        all_v = self.data[:,3]
        dist_to_v = np.abs(all_v - v)
        index = min(dist_to_v) == dist_to_v
        #print(index, v, dist_to_v)
        return self.data[index,:][0]

    def get_row_by_velocity2(self, v: float):
        """
        Return data row closest to a given velocity
        """
        if v > self.data[-1][3]:
            v = self.data[-1][3] - 1e-5
        if v < self.data[0][3]:
            v = self.data[0][3]    
        
        #print(v)
        #print(self.v_dict)
        
        ##idx = 0
        for cont in range( len( self.v_dict ) ):
            l, h, m = self.v_dict[cont]
            if l <= v < h:
                idx = cont
                if v > m:
                    idx = +1

        return self.data[idx][:]

    def get_u_by_velocity(self, v: float):
        """
        Returns data row closest to given velocity
        """
        if not (0 <= v <= 6e3):
            if v < 0:
                v = 0.
            if v > 6e3:
                v = 6e3
                
        all_v = self.data[:,3]
        dist_to_v = np.abs(all_v - v)
        index = min(dist_to_v) == dist_to_v
        #print(index, v, dist_to_v)
        return self.u[index]
    
    
    def save(self, filename: str):
        """Saves the reference trajectory data to a file"""
        np.savez(filename, X_and_lam=self.X_and_lam, u=self.u, tspan=self.tspan, params=self.params)

    @staticmethod
    def load(filename: str):
        """Initializes a new ApolloReferenceData from a saved data file"""
        npzdata = np.load(filename, allow_pickle=True)
        X_and_lam = npzdata.get('X_and_lam')
        u = npzdata.get('u')
        tspan = npzdata.get('tspan')
        params = npzdata.get('params').item()
        return ApolloReferenceData(X_and_lam, u, tspan, params)
