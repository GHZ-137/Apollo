"""
Auxiliary functions
"""
############################################
# Imports
############################################
import numpy as np

def nearest_idx(arr, val):
    idx = (np.abs(arr - val)).argsort()
    return (idx[0], idx[1])

def interp(var, new_x):
    """ Interpolate y values between nearests x
    """
    x, y = var[:,0], var[:,1]
    res = np.zeros((new_x.shape[0], 2))
    for cont in range( len(new_x)):
        a_x = new_x[cont]
        idx1, idx2 = nearest_idx(x, a_x)
        dtot = abs(x[idx1] - x[idx2])
        d1 = abs(a_x - x[idx1])
        d2 = abs(a_x - x[idx2])
        
        if x[idx1] <= a_x <= x[idx2] or x[idx2] <= a_x <= x[idx1]:
            a_y = y[idx1] * (d2/dtot) + y[idx2] * (d1/dtot)
        else:
            a_y = y[idx1]

        res[cont][0] = a_x
        res[cont][1] = a_y
    
    return res


def linearize(F, n_reg, l, h, new_y = [], fixed_edges = True):
    """ Piece-wise linearize a function F given:
     number of regions,
     low, high ends (range),
     discrete points (new_y)
    """
    
    # Get range
    r = h - l
    
    # Build point list
    x0 = 0
    y0 = F[0]
    p = [[x0,y0]]
    d_x = len(F) // n_reg

    x = x0
    for cont in range(n_reg):
        x += d_x
        y = F[x]
        p.append([x,y])
    p = np.array(p)

    # Use extreme discrete points from F 
    if fixed_edges and len(new_y) == p.shape[0] -2:
        p[1:-1,1] = np.array(new_y)
    if not(fixed_edges) and  len(new_y) == p.shape[0]:
        p[:,1] = np.array(new_y)
        
    # Line between discrete points
    res = []
    for cont in range( n_reg ):
        d_y = p[cont+1][1] - p[cont][1]
        m = float( d_y / d_x )
        for cont2 in range(0, d_x):
            x = p[cont][0] + cont2
            y = p[cont][1] + cont2 * m
            res.append([x,y])
    res = np.array(res)
   
    F_res = np.zeros(F.shape) #np.copy(F)
    F_res[:res.shape[0]] = res[:,1]

    # If F is not a power of 2, interpolate end points
    if len(F) > len(res):
        d_x = res[-1][0] - res[-2][0]
        d_y = res[-1][1] - res[-2][1]
        m = d_y / d_x
        x0, y0 = res[-1][0], res[-1][1]

        for cont in range(len(F) - len(res)):
            a_x = x0 + (cont + 1) * d_x
            a_y = y0 + (a_x - x0) * m
            #print(x0, y0, m, a_x, a_y)
            F_res[cont + len(res)] = a_y
    
    return F_res


